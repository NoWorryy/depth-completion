import os
from loguru import logger
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import grad

from kbnet_model import KBNetModel
from posenet_model import PoseNetModel
from net_utils import OutlierRemoval
import losses, networks, net_utils, eval_utils


class Train_net(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, device, model_params, train_params, output_params, pretrained_weights=None):
        super(Train_net, self).__init__()

        self.model_params = model_params
        self.output_params = output_params
        self.train_params = train_params
        self.pretrained_weights = pretrained_weights

        self.device = device


        # 模型初始化
        logger.info('Initializing models:')
        self.depth_model = KBNetModel(**model_params['depth_model_params'])
        self.pose_net = PoseNetModel(**model_params['pose_net_params'])
        self.outlier_removal = OutlierRemoval(**model_params['outlier_params'])

        # if pretrained_weights is not None:
        #     self.load_ckpt(pretrained_weights)

        self.depth_model.to(self.device)
        self.pose_net.to(self.device)
        self.outlier_removal.to(self.device)
        logger.info('Models initialized.')


        # 优化器初始化
        logger.info('Initializing optimizers:')
        self.optimizer_depth_model = torch.optim.Adam(self.depth_model.parameters(), 
                                                      lr=train_params['lr_depth'], betas=(0.5, 0.999))
        self.optimizer_pose_net = torch.optim.Adam(self.pose.parameters(),
                                                   lr=train_params['lr_pose'], betas=(0.5, 0.999))
        logger.info('Optimizers initialized.')


        # 学习率调整器
        logger.info('Initializing schedulers:')

        self.scheduler_depth_model = MultiStepLR(self.optimizer_depth_model, train_params['learning_schedule'],
                                                                  gamma=0.5, last_epoch=-1)
        self.scheduler_pose_net = MultiStepLR(self.optimizer_pose_net, train_params['learning_schedule'],
                                                                  gamma=0.5, last_epoch=-1)
        logger.info('Schedulers initialized.')


        # loss权重配置及模型加载
        self.loss_weights = train_params['loss_weights']

   
        self.iter = self.load_ckpt(pretrained_weights) if pretrained_weights is not None else 0

    def train(self, inputs):
        """
        :param x: dict, contain source, driving, and other
                  source: torch.Tensor,  # (bs, 3, h, w) normalized 0~1
                  driving: torch.Tensor   # (bs, 3, h, w) normalized 0~1
        :return:
        """
        # Set models to training mode
        self.depth_model.train()
        self.pose_net.train()

        self.depth_model.requires_grad_(True)
        self.pose_net.requires_grad_(True)

        self.optimizer_depth_model.zero_grad()
        self.optimizer_pose_net.zero_grad()

  
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image0, image1, image2, sparse_depth0, intrinsics = \
            inputs['image0'], inputs['image1'], inputs['image2'], inputs['sparse_depth0'], inputs['intrinsics']

        # Validity map is where sparse depth is available
        validity_map_depth0 = torch.where(sparse_depth0 > 0, torch.ones_like(sparse_depth0), sparse_depth0)

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth0, filtered_validity_map_depth0 = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth0,
                validity_map=validity_map_depth0)
        
        # Forward through the network
        output_depth0 = self.depth_model.forward(
            image=image0,
            sparse_depth=sparse_depth0,
            validity_map_depth=filtered_validity_map_depth0,
            intrinsics=intrinsics)

        pose01 = self.pose_net.forward(image0, image1)
        pose02 = self.pose_net.forward(image0, image2)

        shape = image0.shape

        # Backproject points to 3D camera coordinates
        points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics, shape)
        target_xy02 = net_utils.project_to_pixel(points, pose02, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image01 = net_utils.grid_sample(image1, target_xy01, shape)
        image02 = net_utils.grid_sample(image2, target_xy02, shape)

        generated = {
            'output_depth0': output_depth0,
            'image01': image01,
            'image02': image02
        }

        # todo: 改参数
        # Compute loss function 
        loss, loss_info = self.compute_loss(
            image0=image0,
            image1=image1,
            image2=image2,
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map_depth0=filtered_validity_map_depth0,
            intrinsics=intrinsics,
            pose01=pose01,
            pose02=pose02,
            **self.train_params['loss_weights'])
        
        loss.backward()

        self.optimizer_depth_model.step()
        self.optimizer_pose_net.step()
 
        return loss_info, generated

    # todo 改参数 删掉计算image01/02
    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map_depth0,
                     intrinsics,
                     pose01,
                     pose02,
                     w_color=0.15,
                     w_structure=0.95,
                     w_sparse_depth=0.60,
                     w_smoothness=0.04):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose01 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose02 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape
        validity_map_image0 = torch.ones_like(sparse_depth0)

        # Backproject points to 3D camera coordinates
        points = net_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy01 = net_utils.project_to_pixel(points, pose01, intrinsics, shape)
        target_xy02 = net_utils.project_to_pixel(points, pose02, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image01 = net_utils.grid_sample(image1, target_xy01, shape)
        image02 = net_utils.grid_sample(image2, target_xy02, shape)

        '''
        Essential loss terms
        '''
        # Color consistency loss function
        loss_color01 = losses.color_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0)
        loss_color02 = losses.color_consistency_loss_func(
            src=image02,
            tgt=image0,
            w=validity_map_image0)
        loss_color = loss_color01 + loss_color02

        # Structural consistency loss function
        loss_structure01 = losses.structural_consistency_loss_func(
            src=image01,
            tgt=image0,
            w=validity_map_image0)
        loss_structure02 = losses.structural_consistency_loss_func(
            src=image02,
            tgt=image0,
            w=validity_map_image0)
        loss_structure = loss_structure01 + loss_structure02

        # Sparse depth consistency loss function
        loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
            src=output_depth0,
            tgt=sparse_depth0,
            w=validity_map_depth0)

        # Local smoothness loss function
        loss_smoothness = losses.smoothness_loss_func(
            predict=output_depth0,
            image=image0)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = w_color * loss_color + \
            w_structure * loss_structure + \
            w_sparse_depth * loss_sparse_depth + \
            w_smoothness * loss_smoothness

        loss_info = {
            'loss' : loss,
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            'loss_sparse_depth' : loss_sparse_depth,
            'loss_smoothness' : loss_smoothness,
        }

        return loss, loss_info


    @torch.no_grad()
    def validate(self, inputs):

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image, sparse_depth, intrinsics, validity_map_depth, ground_truth, gt_mask = \
            inputs['image'], inputs['sparse_depth'], inputs['intrinsics'], inputs['validity_map_depth'], inputs['ground_truth'], inputs['gt_mask']
        
        self.depth_model.eval()
        
        output_depth = self.depth_model.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=validity_map_depth,
            intrinsics=intrinsics)

        # output_depth = np.squeeze(output_depth.cpu().numpy())   # (b, h, w)
        # ground_truth = np.squeeze(ground_truth.cpu().numpy())   # (b, h, w)
        # validity_map_gt = np.squeeze(validity_map_gt.cpu().numpy())

        # Select valid regions to evaluate 放在dataset里
        # validity_mask_gt = np.where(validity_map_gt > 0, 1, 0)
        # min_max_mask = np.logical_and(
        #     ground_truth > self.train_params['min_evaluate_depth'],
        #     ground_truth < self.train_params['max_evaluate_depth'])
        # mask = np.where(np.logical_and(validity_mask_gt, min_max_mask) > 0)

        output_depth = output_depth[gt_mask].cpu().numpy()
        ground_truth = ground_truth[gt_mask].cpu().numpy()

        # Compute validation metrics
        metrics = {}
        metrics['mae'] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        metrics['rmse'] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        metrics['imae'] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        metrics['irmse'] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

        generated = {}
        generated['output_depth'] = output_depth

        return metrics, generated

    def scheduler_epoch_step(self):
        self.scheduler_depth_model.step()
        self.scheduler_pose_net.step()


    def load_ckpt(self, pretrained_weights):
        # 加载预训练权重
        model_path = pretrained_weights.get('model_path', None)
        if model_path is not None:
            logger.info(f"load model checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

            # Load model states
            m, u = self.depth_model.load_state_dict(ckpt['depth_model'], strict=False)
            logger.info(f"depth_model missing keys: {len(m)}, unexpected keys: {len(u)}")
            m, u = self.pose_net.load_state_dict(ckpt['pose_net'], strict=False)
            logger.info(f"pose_net missing keys: {len(m)}, unexpected keys: {len(u)}")

            # Load optimizer states
            self.optimizer_depth_model.load_state_dict(ckpt['optimizer_depth_model'])
            self.optimizer_pose_net.load_state_dict(ckpt['optimizer_pose_net'])

            # Load scheduler states
            self.scheduler_depth_model.load_state_dict(ckpt['scheduler_depth_model'])
            self.scheduler_pose_net.load_state_dict(ckpt['scheduler_pose_net'])

            iter = ckpt.get('iter', 0)
        
        return iter


    def save_checkpoint(self, checkpoint_dir, iteration):
        """
        Save the current state of the model, optimizers, and other necessary components.

        :param checkpoint_dir: Directory where the checkpoint will be saved.
        :param epoch: Current iteration number, which can be used to name the checkpoint file.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ckpt = {
            'iter': iteration,

            'depth_model': self.depth_model.state_dict(),
            'pose_net': self.pose_net.state_dict(),

            'optimizer_depth_model': self.optimizer_depth_model.state_dict(),
            'optimizer_pose_net': self.optimizer_pose_net.state_dict(),

            'scheduler_depth_model': self.scheduler_depth_model.state_dict(),
            'scheduler_pose_net': self.scheduler_pose_net.state_dict()
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_iter_{iteration}.pth')
        torch.save(ckpt, checkpoint_path)
        logger.info(f'Checkpoint saved at {checkpoint_path}')
