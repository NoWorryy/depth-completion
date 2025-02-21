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

    def __init__(self, device, mode, model_params, train_params, pretrained_weights=None):
        super(Train_net, self).__init__()

        self.model_params = model_params
        self.train_params = train_params
        self.pretrained_weights = pretrained_weights
        self.device = device

        self.augmentation_schedule_pos = 0
        self.augmentation_probability = train_params['augmentation_probabilities'][0]

        self.use_dino = model_params['use_dino']
        # Initialize the Accelerator
        if mode == 'train':
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

        if self.use_dino:
            self.depth_anything = DepthAnythingV2(**scale_config)
            self.depth_anything.to(self.device)

        self.scale_model = ScaleModel(nclass=1,
                                      in_channels=scale_config['embed_dim'],
                                      features=scale_config['features'],
                                      out_channels=scale_config['out_channels'],
                                      use_bn=False,
                                      use_clstoken=False,
                                      use_dino=False,
                                      use_prefill=model_params['scale_model_params']['use_prefill'],
                                      output_act='ReLU')
                
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
        if mode == 'test':
            self.scale_model = torch.nn.DataParallel(self.scale_model)
            # self.depth_anything = torch.nn.DataParallel(self.depth_anything)


    def forward(self, inputs):
        """
        param :
        return:
        """
        # Set models to training mode
        if self.use_dino:
            self.depth_anything.eval()
            self.depth_anything.requires_grad_(False)

        self.scale_model.train()
        self.scale_model.requires_grad_(True)
        self.optimizer_scale_model.zero_grad()


        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image, sparse_depth, gt, prefill_depth, d_clear = \
            inputs['image'], inputs['sparse_depth'], inputs['gt'], inputs['prefill_depth'], inputs['d_clear']
        
        # 图像特征
        if self.use_dino:
            img_feat, (resize_h, resize_w) = self.depth_anything.module.infer_image(image, self.model_params['depth_anything']['input_size'])  # (h, w)

            patch_h = resize_h // 14
            patch_w = resize_w // 14   # (B, n, d)

            # 直接回归depth
            output_depth = self.scale_model.forward(img_feat, patch_h, patch_w, image=image, sparse_depth=sparse_depth, d_clear=d_clear, prefill_depth=prefill_depth, img_size=image.shape[2:])    # (B, 518, W0) ---> (B, H, W)
        else:
            output_depth = self.scale_model.forward(image=image, sparse_depth=sparse_depth, d_clear=d_clear, prefill_depth=prefill_depth, img_size=image.shape[2:])    # (B, 518, W0) ---> (B, H, W)

        generated = {
            'prefill_depth': prefill_depth,
            'output_depth0': output_depth[0],
            'output_depth1': output_depth[1],
            'output_depth2': output_depth[2],
            'output_depth3': output_depth[3],
        }

        # Compute loss function 
        loss, loss_info = self.compute_loss(output_depth, gt, self.train_params['loss_weights'])
        
        # loss.backward()
        self.accelerator.backward(loss)

        self.optimizer_scale_model.step()
     
 
        return loss_info, generated



    def compute_gradients(self, img):
        # 定义水平和竖直方向的卷积核
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)

        sobel_y = torch.tensor([[-1, -2, -1], 
                                [0,  0,  0], 
                                [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)

        # 扩展卷积核的维度以适应输入图像的通道数
        sobel_x = sobel_x.repeat(img.size(1), 1, 1, 1)
        sobel_y = sobel_y.repeat(img.size(1), 1, 1, 1)

        # 使用 padding=1 来保持图像尺寸
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.size(1))
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.size(1))

        return grad_x, grad_y


    def compute_loss(self, output_depth, gt, loss_weights):
        # loss_gt

        # loss_l1 = losses.l1loss(pred=output_depth, gt=gt, max_depth=90)
        # loss_l2 = losses.l2loss(pred=output_depth, gt=gt, max_depth=90)

        loss_l1 = losses.l1_multiscale(preds=output_depth, target=gt)
        loss_l2 = losses.l2_multiscale(preds=output_depth, target=gt)


        # loss_e
        # output_depth_norm = torch.zeros_like(output_depth)
        # for i, od in enumerate(output_depth):
        #     output_depth_norm[i] = (od - od.min()) / (od.max() - od.min())
        # grad_x_gen, grad_y_gen = self.compute_gradients(output_depth_norm)

        # rel_depth_norm = torch.zeros_like(rel_depth)
        # for i, rd in enumerate(rel_depth):
        #     rel_depth_norm[i] = (rd - rd.min()) / (rd.max() - rd.min())
        # grad_x_target, grad_y_target = self.compute_gradients(rel_depth_norm)

        # loss_x = torch.abs(grad_x_gen - grad_x_target)
        # loss_y = torch.abs(grad_y_gen - grad_y_target)

        # loss_e = loss_x.mean() + loss_y.mean()

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}
        loss = loss_weights['w_l1'] * loss_l1 + loss_weights['w_l2'] * loss_l2

        loss_info = {
            'loss': loss,
            'loss_l1': loss_l1,
            'loss_l2': loss_l2
        }

        return loss, loss_info


    @torch.no_grad()
    def validate(self, inputs):

        # Set models to test mode
        if self.use_dino:
            self.depth_anything.eval()
            self.depth_anything.requires_grad_(False)

        self.scale_model.eval()
        self.scale_model.requires_grad_(False)


        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image, sparse_depth, gt, prefill_depth, d_clear = \
            inputs['image'], inputs['sparse_depth'], inputs['gt'], inputs['prefill_depth'], inputs['d_clear']
        
        # 获取相对深度图和图像特征
        if self.use_dino:
            img_feat, (resize_h, resize_w) = self.depth_anything.module.infer_image(image, self.model_params['depth_anything']['input_size'])  # (h, w)

            patch_h = resize_h // 14
            patch_w = resize_w // 14   # (B, n, d)
            output_depth = self.scale_model.forward(img_feat, patch_h, patch_w, sparse_depth=sparse_depth, d_clear=d_clear, prefill_depth=prefill_depth, img_size=image.shape[2:])    # (B, 518, W0) ---> (B, H, W)

        # 获取稀疏尺度
        # rel_depth_invalid = (rel_depth == 0)
        # rel_depth[rel_depth_invalid] = 1
        # sparse_scale = torch.div(sparse_depth, rel_depth)
        # sparse_scale[rel_depth_invalid] = 0
        # rel_depth[rel_depth_invalid] = 0

        # 获取置信度图
        # confidence_map = torch.zeros_like(sparse_depth)
        # mask = (sparse_depth > 0.0001)
        # confidence_map[mask] = 1.0

        # 归一化尺度
        # B = sparse_scale.shape[0]
        # max_val = torch.quantile(
        #     sparse_scale.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        # sparse_scale = sparse_scale / max_val 
        
        else:
            output_depth = self.scale_model.forward(image=image, sparse_depth=sparse_depth, d_clear=d_clear, prefill_depth=prefill_depth, img_size=image.shape[2:])    # (B, 518, W0) ---> (B, H, W)

        generated = {
            'prefill_depth': prefill_depth,
            'output_depth0': output_depth[0],
            'output_depth1': output_depth[1],
            'output_depth2': output_depth[2],
            'output_depth3': output_depth[3],
        }
        
        # output_depth = output_depth.cpu().numpy()
        # gt = gt.cpu().numpy()

        # Compute validation metrics
        # metrics = {}
        # metrics['mae'] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * gt)
        # metrics['rmse'] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * gt)
        # metrics['imae'] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * gt)
        # metrics['irmse'] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * gt)

        metrics = eval_utils.evaluate(gt=gt, pred=output_depth[3])

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
        if self.use_dino:   
            depth_anything_ckpt_path = pretrained_weights.get('depth_anything_ckpt', None)
            m, u = self.depth_anything.load_state_dict(torch.load(depth_anything_ckpt_path), strict=False)
            logger.info(f"scale_model loaded, missing keys: {len(m)}, unexpected keys: {len(u)}")
        
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
    
    def normalize(self,
                sparse_scale: torch.Tensor):
        B, C, H, W = sparse_scale.shape
        min_val = torch.quantile(
            sparse_scale.reshape(B, -1), 0., dim=1, keepdim=True)[:, :, None, None]
        max_val = torch.quantile(
            sparse_scale.reshape(B, -1), 1., dim=1, keepdim=True)[:, :, None, None]
        # sparse_scale = (sparse_scale - min_val) / (max_val - min_val)
        sparse_scale = sparse_scale / max_val   
        return sparse_scale, min_val, max_val

    def denormalize(self,
                    scale: torch.Tensor,
                    min_val: torch.Tensor,
                    max_val: torch.Tensor):
        return scale * (max_val - min_val) + min_val
