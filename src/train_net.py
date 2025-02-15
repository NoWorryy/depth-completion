import os
from loguru import logger
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed


from scale_model import ScaleModel
# from net_utils import OutlierRemoval
from transforms import Transforms
from depth_anything_v2.dpt import DepthAnythingV2
import losses, eval_utils


class Train_net(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, device, model_params, train_params, pretrained_weights=None):
        super(Train_net, self).__init__()

        self.model_params = model_params
        self.train_params = train_params
        self.pretrained_weights = pretrained_weights
        self.device = device

        self.augmentation_schedule_pos = 0
        self.augmentation_probability = train_params['augmentation_probabilities'][0]

        # Initialize the Accelerator
        logger.info('Initializing the Accelerator:')
        self.accelerator = Accelerator(
            # gradient_accumulation_steps=train_params['gradient_accumulation_steps'],
            # mixed_precision=train_params['mixed_precision_training'],
            # log_with='tensorboard',
            # project_dir=os.environ['SUMMARY_DIR']
        )
        logger.info('The accelerator Initialized.')
        self.device = self.accelerator.device

        # 模型初始化
        logger.info('Initializing models:')

        model_configs = {
            'vits': {'encoder': 'vits', 'embed_dim': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'embed_dim': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'embed_dim': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'embed_dim': 1536, 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        scale_config = model_configs[model_params['depth_anything']['encoder']]

        self.depth_anything = DepthAnythingV2(**scale_config)

        self.scale_model = ScaleModel(nclass=1,
                                      in_channels=scale_config['embed_dim'],
                                      features=scale_config['features'],
                                      out_channels=scale_config['out_channels'],
                                      use_bn=False,
                                      use_clstoken=False,
                                      use_prefill=model_params['scale_model_params']['use_prefill'],
                                      output_act='sigmoid')
        
        # self.outlier_removal = OutlierRemoval(**model_params['outlier_params'])

        self.depth_anything.to(self.device)
        self.scale_model.to(self.device)
   
        logger.info('Models initialized.')


        # 优化器初始化
        logger.info('Initializing optimizers:')
        self.optimizer_scale_model = torch.optim.Adam(self.scale_model.parameters(), lr=train_params['lr_scale'], betas=(0.9, 0.999))
        logger.info('Optimizers initialized.')


        # 学习率调整器
        logger.info('Initializing schedulers:')
        self.scheduler_scale_model = MultiStepLR(self.optimizer_scale_model, train_params['learning_schedule'], gamma=0.5, last_epoch=-1)
        logger.info('Schedulers initialized.')


        # loss权重配置及模型加载
        self.loss_weights = train_params['loss_weights']
        self.iter = self.load_ckpt(pretrained_weights)
        logger.info('Ckpt loaded.')

        # self.train_transforms = Transforms(**train_params['aug_params'])

        # self.scale_model = torch.nn.DataParallel(self.scale_model)
        # self.depth_anything = torch.nn.DataParallel(self.depth_anything)


    def forward(self, inputs):
        """
        param :
        return:
        """
        # Set models to training mode
        self.depth_anything.eval()
        self.depth_anything.requires_grad_(False)

        self.scale_model.train()
        self.scale_model.requires_grad_(True)
        self.optimizer_scale_model.zero_grad()


        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image, sparse_depth, gt = \
            inputs['image'], inputs['sparse_depth'], inputs['gt']
        
        # 获取相对深度图和图像特征
        rel_depth, img_feat = self.depth_anything.module.infer_image(image, self.model_params['depth_anything']['input_size'])  # (h, w)
        rel_depth = (rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min())

        # 获取稀疏尺度
        rel_depth_invalid = (rel_depth == 0)
        rel_depth[rel_depth_invalid] = 1
        sparse_scale = torch.div(sparse_depth, rel_depth)
        sparse_scale[rel_depth_invalid] = 0
        rel_depth[rel_depth_invalid] = 0

        patch_h = 37
        patch_w = img_feat[0][0].shape[1] // 37   # (B, n, d)
        ouput_scale = self.scale_model.forward(img_feat, patch_h, patch_w, sparse_scale, img_size=image.shape[2:])    # (B, 518, W0) ---> (B, H, W)
        # ouput_scale = F.interpolate(ouput_scale[:, None], image.shape[2:], mode="bilinear", align_corners=True)     # (B, H, W)

        output_depth = torch.mul(ouput_scale, rel_depth)
        generated = {
            'ouput_scale': ouput_scale, 
            'output_depth': output_depth
        }

        # Compute loss function 
        loss, loss_info = self.compute_loss(generated, gt, rel_depth, self.train_params['loss_weights'])
        
        # loss.backward()
        self.accelerator.backward(loss)

        self.optimizer_scale_model.step()
     
 
        return loss_info, generated


    def compute_loss(self, generated, gt, rel_depth, loss_weights):

        gt_mask = (gt != 0)
        valid_points = gt_mask.sum()
        loss_gt = torch.sum(torch.abs(generated['output_depth'] - gt ) * gt_mask) / valid_points

        # todo: loss_e
        loss_e = torch.zeros_like(loss_gt)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = loss_weights['w_gt'] * loss_gt + loss_weights['w_e'] * loss_e

        loss_info = {
            'loss': loss,
            'loss_gt': loss_gt,
            'loss_e': loss_e
        }

        return loss, loss_info


    @torch.no_grad()
    def validate(self, inputs):

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image, sparse_depth, intrinsics, validity_map_depth, ground_truth, gt_mask = \
            inputs['image'], inputs['sparse_depth'], inputs['intrinsics'], inputs['validity_map_depth'], inputs['ground_truth'], inputs['gt_mask']
        
        self.scale_model.eval()
        
        output_depth_map = self.scale_model.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=validity_map_depth,
            intrinsics=intrinsics)

        output_depth = output_depth_map[gt_mask].cpu().numpy()
        ground_truth = ground_truth[gt_mask].cpu().numpy()

        # Compute validation metrics
        metrics = {}
        metrics['mae'] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        metrics['rmse'] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        metrics['imae'] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        metrics['irmse'] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

        generated = {}
        generated['output_depth'] = output_depth_map

        return metrics, generated


    def scheduler_epoch_step(self):
        self.scheduler_scale_model.step()


    def load_ckpt(self, pretrained_weights):
        # 加载预训练权重
        iter = 0
        model_path = pretrained_weights.get('model_path', None)
        if model_path is not None:
            logger.info(f"load model checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

            scale_state_dict = {}
            for key, value in ckpt['scale_model'].items():
                new_key = key.replace('module.', '')  # 去掉 'module.' 前缀
                scale_state_dict[new_key] = value
            
            # Load model states
            m, u = self.scale_model.load_state_dict(scale_state_dict, strict=False)
            logger.info(f"scale_model loaded, missing keys: {len(m)}, unexpected keys: {len(u)}")
    
            # Load optimizer states
            self.optimizer_scale_model.load_state_dict(ckpt['optimizer_scale_model'])
     
            # Load scheduler states
            self.scheduler_scale_model.load_state_dict(ckpt['scheduler_scale_model'])
  
            iter = ckpt.get('iter', 0)
            logger.info(f"iter loaded: {iter}")
        
        depth_anything_ckpt_path = pretrained_weights.get('depth_anything_ckpt', None)
        self.depth_anything.load_state_dict(torch.load(depth_anything_ckpt_path))
        
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

            'scale_model': self.accelerator.unwrap_model(self.scale_model).state_dict(),

            'optimizer_scale_model': self.optimizer_scale_model.state_dict(),

            'scheduler_scale_model': self.scheduler_scale_model.state_dict(),
    
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_iter_{iteration}.pth')
        torch.save(ckpt, checkpoint_path)
        logger.info(f'Checkpoint saved at {checkpoint_path}')
