import os
from loguru import logger
import itertools

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DeepSpeedPlugin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.optim.lr_scheduler import MultiStepLR

from src.modules.util import ImagePyramide, make_coordinate_grid_2d
from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix

from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.modules.motion_extractor import MotionExtractor
from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from src.modules.stitching_retargeting_network import StitchingRetargetingNetwork

from src.utils.helper import concat_feat, remove_ddp_dumplicate_key
from src.utils.retargeting_utils import calc_eye_close_ratio_torch, calc_lip_close_ratio_torch
from src.utils.dependencies.model import BiSeNet
from src.utils.pixelai_aliface_sdk.aliface import AliFace

SOURCE_LMK_DET_OLD = True

def get_nonshoulder_mask(images, parsing, keypoints, target_size=None):
    # 0~23 左眼睛
    # 24~47 右眼睛
    # 145～164 左眉毛
    # 165～184 右眉毛
    # images: (B, 3, H, W)
    # parsing: (b, n, 512, 512)
    b = parsing.shape[0]
    b, _, h, w = images.shape
    masks = torch.zeros(images.shape, dtype=images.dtype, device=images.device)

    for img_idx in range(images.size(0)):  # B

        parsing_img = parsing[img_idx].argmax(0)  # (512, 512)

        face_mask = torch.zeros((512, 512), device=images.device)
        for pi in range(1, 14):
            index = torch.where(parsing_img == pi)
            face_mask[index[0], index[1]] = 1  # 脸 B

        torso_mask = torch.zeros((512, 512), device=images.device)
        for pi in range(16, 17):  # 身体
            index = torch.where(parsing_img == pi)
            torso_mask[index[0], index[1]] = 1

        face_index = torch.where(face_mask)
        if len(face_index[0]) != 0 :
            x1 = face_index[1].min()
            x2 = face_index[1].max()
            y_min = face_index[0].min()
            y_max = face_index[0].max()
            height = y_max - y_min
            y3 = min(images.size(2), int(y_max + 0.2 * height))  # h
        else:
            x1 = keypoints[img_idx, 108:145, 0].min()
            x2 = keypoints[img_idx, 108:145, 0].max()
            y_max = keypoints[img_idx, 126, 1]
            y_min = keypoints[img_idx, 145:185, 1].min()    # 眉毛

            face_width = x2 - x1
            x1 = int(x1) - int(0.05 * face_width)  # 决定左边界
            x2 = int(x2) + int(0.05 * face_width)  # 决定右边界

            face_height = y_max - y_min
            y3 = int(y_max) + int(0.2 * face_height)
            y3 = min(y3, w)

        left_shoulder = torch.where(torso_mask[:, :x1]) # 返回的元组，0是水平坐标y，1是垂直坐标x
        right_shoulder = torch.where(torso_mask[:, x2:])

        y1 = left_shoulder[0].min() if len(left_shoulder[0]) != 0 else y3
        y2 = right_shoulder[0].min() if len(right_shoulder[0]) != 0 else y3

        # 裁剪图像
        masks[img_idx, :, :y1, :x1] = 1.0
        masks[img_idx, :, :y3, x1:x2] = 1.0
        masks[img_idx, :, :y2, x2:] = 1.0

    if target_size is not None:
        masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)

    return masks


def get_eyes_area_mask(images, keypoints, target_size=None):
    # 0~23 左眼睛
    # 24~47 右眼睛
    # 145～164 左眉毛
    # 165～184 右眉毛
    # images: (B, 3, H, W)
    # keypoints: (B, 203, 2)
    masks = torch.zeros_like(images)

    for img_idx in range(images.size(0)):  # B
        # 嘴巴轮廓点用来确定左右边
        eyes_region = keypoints[img_idx, 0:48, :]  # (48,2)
        brow_region = keypoints[img_idx, 145:185, :]  # (40,2)
        lip_region = torch.cat((eyes_region, brow_region), dim=0)  # (88, 2)
        lip_x_min = lip_region[:, 0].min()  # 最左边
        lip_x_max = lip_region[:, 0].max()  # 最右边

        lip_y_min = lip_region[:, 1].min()  # 最上边
        lip_y_max = lip_region[:, 1].max()  # 最下边

        # 计算裁剪区域
        lip_width = lip_x_max - lip_x_min
        lip_height = lip_y_max - lip_y_min
        x1 = int(lip_x_min) - int(0.1 * lip_width)  # 决定左边界
        x2 = int(lip_x_max) + int(0.1 * lip_width)  # 决定右边界

        y1 = int(lip_y_min) - int(0.3 * lip_height)
        y2 = int(lip_y_max) + int(0.3 * lip_height)

        # 确保裁剪区域不越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(images.size(3), x2)  # w
        y2 = min(images.size(2), y2)  # h

        # 裁剪图像
        masks[img_idx, :, y1:y2, x1:x2] = 1.0

    if target_size is not None:
        masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)

    return masks


def get_lip_area_mask(images, keypoints, target_size=None):
    masks = torch.zeros_like(images)

    for img_idx in range(images.size(0)):
        # 嘴巴轮廓点用来确定左右边
        lip_region = keypoints[img_idx, 48:84, :]  # (36,2)
        lip_x_min = lip_region[:, 0].min()  # 最左边
        lip_x_max = lip_region[:, 0].max()  # 最右边

        lip_y_min = lip_region[:, 1].min()  # 最上边
        lip_y_max = lip_region[:, 1].max()  # 最下边

        # 计算裁剪区域
        lip_width = lip_x_max - lip_x_min
        lip_height = lip_y_max - lip_y_min
        x1 = int(lip_x_min) - int(0.2 * lip_width)  # 决定左边界
        x2 = int(lip_x_max) + int(0.2 * lip_width)  # 决定右边界

        y1 = int(lip_y_min) - int(0.4 * lip_height)
        y2 = int(lip_y_max) + int(0.4 * lip_height)

        # 确保裁剪区域不越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(images.size(3), x2)
        y2 = min(images.size(2), y2)

        # 裁剪图像
        masks[img_idx, :, y1:y2, x1:x2] = 1.0

    if target_size is not None:
        masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)

    return masks


def keypoint_transformation(kp_canonical: torch.Tensor, kp_info: dict, kp_source_info: dict = None):
    """
    Transform the implicit keypoints with the pose, shift, and expression deformation.

    :param kp_canonical: canonical keypoints from source image, shape: (bs, num_kp, 3)
    :param kp_info: motion module output dict with the following keys:
                    'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'

    :return: Transformed keypoints, shape: (bs, num_kp, 3)
    """
    kp = kp_info['kp']  # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    scale = kp_source_info['scale'] if kp_source_info else kp_info['scale']
    t = kp_source_info['t'] if kp_source_info else kp_info['t']
    
    exp = kp_info['exp']

    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = kp_canonical.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


class Stage2Trainer(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, model_params, train_params, pretrained_weights=None):
        super(Stage2Trainer, self).__init__()

        self.model_params = model_params
        self.train_params = train_params
        self.pretrained_weights = pretrained_weights

        # Initialize the Accelerator
        logger.info('Initializing the Accelerator:')
        self.accelerator = Accelerator(
            gradient_accumulation_steps=train_params['gradient_accumulation_steps'],
            mixed_precision=train_params['mixed_precision_training'],
            # log_with='tensorboard',
            # project_dir=os.environ['SUMMARY_DIR']
        )
        logger.info('The accelerator Initialized.')

        self.device = self.accelerator.device

        # 模型初始化
        logger.info('Initializing models:')
        self.appearance_feature_extractor = AppearanceFeatureExtractor(
            **model_params['appearance_feature_extractor_params'])
        self.motion_extractor = MotionExtractor(**model_params['motion_extractor_params'])
        self.warping_module = WarpingNetwork(**model_params['warping_module_params'])
        self.generator = SPADEDecoder(**model_params['spade_generator_params'])
        self.parsing = BiSeNet(n_classes=19)
        self.ali_face = AliFace(prefix=train_params['lmk_det'], device=self.device)

        # Special handling for stitching and retargeting module
        config = model_params['stitching_retargeting_module_params']
        self.stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        self.retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        self.retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))

        if pretrained_weights is not None:
            self.load_cpk(pretrained_weights)
        self.parsing.load_state_dict(torch.load(train_params['parsing']))

        self.appearance_feature_extractor.to(self.device).eval()
        self.motion_extractor.to(self.device).eval()
        self.warping_module.to(self.device).eval()
        self.generator.to(self.device).eval()
        self.parsing.to(self.device).eval()
        self.appearance_feature_extractor.requires_grad_(False)
        self.motion_extractor.requires_grad_(False)
        self.warping_module.requires_grad_(False)
        self.generator.requires_grad_(False)
        self.parsing.requires_grad_(False)

        self.stitcher.to(self.device).train()
        self.retargetor_lip.to(self.device).train()
        self.retargetor_eye.to(self.device).train()
        self.stitcher.requires_grad_(True)
        self.retargetor_lip.requires_grad_(True)
        self.retargetor_eye.requires_grad_(True)

        logger.info('Models initialized.')

        # 优化器初始化
        logger.info('Initializing optimizers:')
        self.optimizer_stitcher_module = torch.optim.Adam(self.stitcher.parameters(),
                                                          lr=train_params['lr_stitcher_module'],
                                                          betas=(0.5, 0.999))
        self.optimizer_retargetor_lip_module = torch.optim.Adam(self.retargetor_lip.parameters(),
                                                                lr=train_params['lr_retargetor_lip_module'],
                                                                betas=(0.5, 0.999))
        self.optimizer_retargetor_eye_module = torch.optim.Adam(self.retargetor_eye.parameters(),
                                                                lr=train_params['lr_retargetor_eye_module'],
                                                                betas=(0.5, 0.999))
        logger.info('Optimizers initialized.')

        # 学习率调整器
        logger.info('Initializing schedulers:')
        self.start_epoch = 0
        self.scheduler_stitcher_module = MultiStepLR(self.optimizer_stitcher_module,
                                                     train_params['epoch_milestones'], gamma=0.1,
                                                     last_epoch=self.start_epoch - 1)
        self.scheduler_retargetor_lip_module = MultiStepLR(self.optimizer_retargetor_lip_module,
                                                           train_params['epoch_milestones'], gamma=0.1,
                                                           last_epoch=self.start_epoch - 1)
        self.scheduler_retargetor_eye_module = MultiStepLR(self.optimizer_retargetor_eye_module,
                                                           train_params['epoch_milestones'], gamma=0.1,
                                                           last_epoch=self.start_epoch - 1)

        logger.info('Schedulers initialized.')

        # loss权重配置及模型加载
        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """
        feat_stiching = concat_feat(kp_source, kp_driving)  # (b, -1)
        delta = self.stitcher(feat_stiching)

        return delta

    def retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)
        delta = self.retargetor_eye(feat_eye)
        return delta.reshape(-1, kp_source.shape[1], 3)

    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)
        delta = self.retargetor_lip(feat_lip)
        return delta.reshape(-1, kp_source.shape[1], 3)

    def forward(self, x):
        """
        :param x: dict, contain source, driving, and other
                  source: torch.Tensor,  # (bs, 3, h, w) normalized 0~1
                  driving: torch.Tensor   # (bs, 3, h, w) normalized 0~1
        :return:
        """
        self.optimizer_stitcher_module.zero_grad()
        self.optimizer_retargetor_lip_module.zero_grad()
        self.optimizer_retargetor_eye_module.zero_grad()

        source = x['source'].to(self.device)
        driving = x['driving'].to(self.device)
        source_lmk = x['source_lmk'].to(self.device)

        batch_size = source.size(0)
        c_d_eye = (torch.rand(batch_size, 1) * 0.8).to(self.device)
        c_d_lip = (torch.rand(batch_size, 1) * 0.8).to(self.device)

        # get source 3d feature
        feature_3d = self.appearance_feature_extractor(source)
        kp_info_source = self.motion_extractor(source)
        kp_info_driving = self.motion_extractor(driving)

        kp_canonical = kp_info_source['kp']
        kp_source = keypoint_transformation(kp_canonical=kp_canonical, kp_info=kp_info_source)  # (b, 21, 3)
        kp_driving = keypoint_transformation(kp_canonical=kp_canonical, kp_info=kp_info_driving, kp_source_info=kp_info_source) # stiching时用source的s t

        # stitching_retargeting_module
        st_delta = self.stitching(kp_source, kp_driving)
        bs, num_kp = kp_source.shape[:2]
        delta_exp = st_delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x21x3
        delta_tx_ty = st_delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2
        kp_st = kp_driving + delta_exp
        kp_st[..., :2] += delta_tx_ty

        # retargeting
        if SOURCE_LMK_DET_OLD:
            c_s_eyes = calc_eye_close_ratio_torch(source_lmk, is_lmk106=False).float().to(self.device)  # (b, 2)
            c_s_lip = calc_lip_close_ratio_torch(source_lmk, is_lmk106=False).float().to(self.device)   # (b, 1)
        else:
            source_512 = F.interpolate(source, size=(512, 512), mode='bilinear', align_corners=False)
            source_bboxes = self.ali_face.dl_face_detection_batch(source_512, img_dim=512)  # (b, num_of_faces, 5)
            source_lmk = self.ali_face.dl_face_alignment_batch(source, source_bboxes)  # (b, 106, 2)
            c_s_lip = calc_lip_close_ratio_torch(source_lmk, is_lmk106=True).float().to(self.device)   # (b, 1)
            c_s_eyes = calc_eye_close_ratio_torch(source_lmk, is_lmk106=True).float().to(self.device)  # (b, 2)

        combined_eye_ratio_tensor = torch.cat([c_s_eyes, c_d_eye], dim=1)    # (b, 3)
        eyes_delta = self.retarget_eye(kp_source, combined_eye_ratio_tensor)
        kp_eye = kp_source + eyes_delta
        
        combined_lip_ratio_tensor = torch.cat([c_s_lip, c_d_lip], dim=1)  # bx2
        lip_delta = self.retarget_lip(kp_source, combined_lip_ratio_tensor)
        kp_lip = kp_source + lip_delta

        # get decoder input
        ret_dct_recon = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_source)
        ret_dct_st = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_st)
        ret_dct_eye = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_eye)
        ret_dct_lip = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_lip)

        # decode
        gen_out_recon = self.generator(feature=ret_dct_recon['out'])
        gen_out_st = self.generator(feature=ret_dct_st['out'])
        gen_out_eye = self.generator(feature=ret_dct_eye['out'])
        gen_out_lip = self.generator(feature=ret_dct_lip['out'])

        # cal eye/lip close ratio
        try:
            gen_out_eye_bboxes = self.ali_face.dl_face_detection_batch(gen_out_eye, img_dim=512)  # (b, num_of_faces, 5)
            gen_out_eye_lmk = self.ali_face.dl_face_alignment_batch(gen_out_eye, gen_out_eye_bboxes)  # (b, 106, 2)
            p_s_eye = calc_eye_close_ratio_torch(gen_out_eye_lmk, is_lmk106=True)   # (b, 2)
            p_s_eye = torch.mean(p_s_eye, dim=1).unsqueeze(1)   # (b, 1)
            gen_out_eye_lmk = gen_out_eye_lmk[:, [72, 73, 52, 55, 75, 76, 58, 61], :]
        except Exception as e:
            print(f"face detection error: {type(e).__name__}: {e}")
            p_s_eye = c_d_eye.clone()
            gen_out_eye_lmk = None

        try:
            gen_out_lip_bboxes = self.ali_face.dl_face_detection_batch(gen_out_lip, img_dim=512)  # (b, num_of_faces, 5)
            gen_out_lip_lmk = self.ali_face.dl_face_alignment_batch(gen_out_lip, gen_out_lip_bboxes)  # (b, 106, 2)
            p_s_lip = calc_lip_close_ratio_torch(gen_out_lip_lmk, is_lmk106=True)   # (b, 1)
            gen_out_lip_lmk = gen_out_lip_lmk[:, [98, 102, 84, 90], :]
        except Exception as e:
            print(f"face detection error: {type(e).__name__}: {e}")
            p_s_lip = c_d_lip.clone()
            gen_out_lip_lmk = None

        generated = {
            'gen_out_recon': gen_out_recon,
            'gen_out_st': gen_out_st,
            'gen_out_eye': gen_out_eye,
            'gen_out_lip': gen_out_lip,
            'st_delta': st_delta,
            'eyes_delta': eyes_delta,
            'lip_delta': lip_delta,
            'c_d_eye': c_d_eye,
            'c_d_lip': c_d_lip,
            'c_s_eye': c_s_eyes,
            'c_s_lip': c_s_lip,
            'p_s_eye': p_s_eye,
            'p_s_lip': p_s_lip,
            'gen_out_eye_lmk': gen_out_eye_lmk,
            'gen_out_lip_lmk': gen_out_lip_lmk
        }

        losses, mask = self.compute_loss(x, generated)
        generated['mask'] = mask

        loss_values = [val.mean() for val in losses.values()]
        loss = sum(loss_values)

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            all_params = itertools.chain(*[self.stitcher.parameters(), self.retargetor_lip.parameters(),
                                           self.retargetor_eye.parameters()])
            self.accelerator.clip_grad_norm_(all_params, self.train_params['max_grad_norm'])

        self.optimizer_stitcher_module.step()
        self.optimizer_retargetor_lip_module.step()
        self.optimizer_retargetor_eye_module.step()

        return losses, generated

    def compute_loss(self, x, generated):
        loss_values = {}
        mask = {}
        source_lmk = x['source_lmk'].to(self.device)
        driving_512 = x['driving_512'].to(self.device)
        source = x['source'].to(self.device)

        # stitcher loss
        source_img = F.interpolate(source, size=(512, 512), mode='bilinear',
                                   align_corners=False)  # (b, 3, 512, 512)
        source_parsing = self.parsing(source_img)  # (b, n, 512, 512)

        nonshoulder_mask = get_nonshoulder_mask(driving_512, source_parsing, source_lmk * 2)  # 512的mask
        st_delta_reg = torch.norm(generated['st_delta'], p=1, dim=-1).mean()  # L∆
        loss_values['st_loss'] = (torch.mean(
            torch.abs(generated['gen_out_st'] - generated['gen_out_recon']) * (1 - nonshoulder_mask)) +
                                  self.loss_weights['st_reg'] * st_delta_reg)
        mask['nonshoulder_mask'] = nonshoulder_mask

        # eye loss
        eyes_mask = get_eyes_area_mask(driving_512, source_lmk * 2)  # lmk转到512
        eyes_delta_reg = torch.mean(torch.abs(generated['eyes_delta']))
        eyes_cond_loss = torch.mean(torch.abs(generated['p_s_eye'] - generated['c_d_eye']))
        loss_values['eye_img_loss'] = torch.mean(torch.abs(generated['gen_out_eye'] - generated['gen_out_recon']) * (1 - eyes_mask))
        loss_values['eye_reg_loss'] = self.loss_weights['eye_reg'] * eyes_delta_reg
        loss_values['eye_cond_loss'] = self.loss_weights['eye_cond'] * eyes_cond_loss
        mask['eye_mask'] = eyes_mask

        # lip loss
        lip_mask = get_lip_area_mask(driving_512, source_lmk * 2)  # lmk转到512
        lip_delta_reg = torch.mean(torch.abs(generated['lip_delta']))
        lip_cond_loss = torch.mean(torch.abs(generated['p_s_lip'] - generated['c_d_lip']))
        loss_values['lip_img_loss'] = torch.mean(torch.abs(generated['gen_out_lip'] - generated['gen_out_recon']) * (1 - lip_mask))
        loss_values['lip_reg_loss'] = self.loss_weights['lip_reg'] * lip_delta_reg
        loss_values['lip_cond_loss'] = self.loss_weights['lip_cond'] * lip_cond_loss
        mask['lip_mask'] = lip_mask

        return loss_values, mask

    def scheduler_epoch_step(self):
        self.scheduler_stitcher_module.step()
        self.scheduler_retargetor_lip_module.step()
        self.scheduler_retargetor_eye_module.step()

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def load_cpk(self, pretrained_weights):
        # 加载预训练权重
        appearance_feature_extractor_weights_path = pretrained_weights.get('appearance_feature_extractor', None)
        if appearance_feature_extractor_weights_path is not None:
            logger.info(f"load appearance_feature_extractor checkpoint: {appearance_feature_extractor_weights_path}")
            appearance_feature_extractor_weights = torch.load(appearance_feature_extractor_weights_path,
                                                              map_location=lambda storage, loc: storage)
            m, u = self.appearance_feature_extractor.load_state_dict(appearance_feature_extractor_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        motion_extractor_weights_path = pretrained_weights.get('motion_extractor', None)
        if motion_extractor_weights_path is not None:
            logger.info(f"load motion_extractor checkpoint: {motion_extractor_weights_path}")
            motion_extractor_weights = torch.load(motion_extractor_weights_path,
                                                  map_location=lambda storage, loc: storage)
            m, u = self.motion_extractor.load_state_dict(motion_extractor_weights, strict=False)
            logger.info(f"load missing keys: {len(m)}, unexpected keys: {len(u)}")

        warping_module_weights_path = pretrained_weights.get('warping_module', None)
        if warping_module_weights_path is not None:
            logger.info(f"load warping_module checkpoint: {warping_module_weights_path}")
            warping_module_weights = torch.load(warping_module_weights_path,
                                                map_location=lambda storage, loc: storage)
            m, u = self.warping_module.load_state_dict(warping_module_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        spade_generator_weights_path = pretrained_weights.get('spade_generator', None)
        if spade_generator_weights_path is not None:
            logger.info(f"load spade_generator checkpoint: {spade_generator_weights_path}")
            spade_generator_weights = torch.load(spade_generator_weights_path,
                                                 map_location=lambda storage, loc: storage)
            m, u = self.generator.load_state_dict(spade_generator_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        stitching_retargeting_module_weights_path = pretrained_weights.get('stitching_retargeting_module', None)
        if stitching_retargeting_module_weights_path is not None:
            checkpoint = torch.load(stitching_retargeting_module_weights_path,
                                    map_location=lambda storage, loc: storage)
            self.stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
            self.retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
            self.retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))

    def save_checkpoint(self, checkpoint_dir, iteration):
        """
        Save the current state of the model, optimizers, and other necessary components.

        :param checkpoint_dir: Directory where the checkpoint will be saved.
        :param epoch: Current iteration number, which can be used to name the checkpoint file.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_data = {
            'iteration': iteration,
            'retarget_shoulder': self.accelerator.unwrap_model(
                self.stitcher).state_dict(),
            'retarget_mouth': self.accelerator.unwrap_model(self.retargetor_lip).state_dict(),
            'retarget_eye': self.accelerator.unwrap_model(self.retargetor_eye).state_dict(),

            'optimizer_stitcher_module': self.optimizer_stitcher_module.state_dict(),
            'optimizer_retargetor_eye_module': self.optimizer_retargetor_eye_module.state_dict(),
            'optimizer_retargetor_lip_module': self.optimizer_retargetor_lip_module.state_dict(),

            'scheduler_stitcher_module': self.scheduler_stitcher_module.state_dict(),
            'scheduler_retargetor_eye_module': self.scheduler_retargetor_eye_module.state_dict(),
            'scheduler_retargetor_lip_module': self.scheduler_retargetor_lip_module.state_dict(),
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
        self.accelerator.save(checkpoint_data, checkpoint_path)
        logger.info(f'Checkpoint saved at {checkpoint_path}')
