import os
from loguru import logger
import itertools


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import grad


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']),
                                                          type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def crop_and_resize_face_area(images, keypoints, target_size=(128, 128)):
    cropped_images = []

    for img_idx in range(images.size(0)):
        # 轮廓点用来确定左右边
        left_contour_x = keypoints[img_idx, 108:145, 0].min()  # 左轮廓的最左边
        right_contour_x = keypoints[img_idx, 108:145, 0].max()  # 右轮廓的最右边

        # 下巴点
        chin_x, chin_y = keypoints[img_idx, 126]

        # 眉毛点
        lb_y_min = keypoints[img_idx, 145:165, 1].min()  # 左眉毛的最小 y 坐标
        rb_y_min = keypoints[img_idx, 165:185, 1].min()  # 右眉毛的最小 y 坐标

        # 计算裁剪区域
        face_width = right_contour_x - left_contour_x
        x1 = int(left_contour_x) - int(0.05 * face_width)  # 决定左边界
        x2 = int(right_contour_x) + int(0.05 * face_width)  # 决定右边界

        # 根据最小的眉毛 y 坐标和下巴 y 坐标设置上下边界
        y1 = int(min(lb_y_min, rb_y_min))  # 向上取一点
        y2 = int(chin_y)  # 向下取一点，确保完全包含下巴
        face_height = y2 - y1
        y1 = y1 - int(0.1 * face_height)
        y2 = y2 + int(0.05 * face_height)

        # 确保裁剪区域不越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(images.size(3), x2)
        y2 = min(images.size(2), y2)

        # 裁剪图像
        cropped_img = images[img_idx, :, y1:y2, x1:x2]

        # 调整为目标大小
        resize_transform = transforms.Resize(target_size)
        resized_img = resize_transform(cropped_img)

        cropped_images.append(resized_img)

    return torch.stack(cropped_images)


def crop_and_resize_lip_area(images, keypoints, target_size=(64, 128)):
    cropped_images = []

    for img_idx in range(images.size(0)):
        # 嘴巴轮廓点用来确定左右边
        lip_left_x = keypoints[img_idx, 48:85, 0].min()  # 左轮廓的最左边
        lip_right_x = keypoints[img_idx, 48:85, 0].max()  # 右轮廓的最右边

        lip_y_min = keypoints[img_idx, 48:85, 1].min()  # 最小 y 坐标
        lip_y_max = keypoints[img_idx, 48:85, 1].max()

        # 计算裁剪区域
        lip_width = lip_right_x - lip_left_x
        x1 = int(lip_left_x) - int(0.15 * lip_width)  # 决定左边界
        x2 = int(lip_right_x) + int(0.15 * lip_width)  # 决定右边界

        y1 = int(lip_y_min) - int(0.15 * lip_width)
        y2 = int(lip_y_max) + int(0.15 * lip_width)

        # 确保裁剪区域不越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(images.size(3), x2)
        y2 = min(images.size(2), y2)

        # 裁剪图像
        cropped_img = images[img_idx, :, y1:y2, x1:x2]

        # 调整为目标大小
        resize_transform = transforms.Resize(target_size)
        resized_img = resize_transform(cropped_img)

        cropped_images.append(resized_img)

    return torch.stack(cropped_images)


def keypoint_transformation(kp_canonical: torch.Tensor, kp_info: dict):
    """
    Transform the implicit keypoints with the pose, shift, and expression deformation.

    :param kp_canonical: canonical keypoints from source image, shape: (bs, num_kp, 3)
    :param kp_info: motion module output dict with the following keys:
                    'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'

    :return: Transformed keypoints, shape: (bs, num_kp, 3)
    """
    kp = kp_info['kp']  # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']

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


class Train_net(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, model_params, discriminator_params, discriminator_face_params, discriminator_lip_params,
                 train_params, pretrained_weights=None):
        super(Trainer, self).__init__()

        self.model_params = model_params
        self.discriminator_params = discriminator_params
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
        self.stitching_retargeting_module = None  # todo: stage2 training
        self.warping_module = WarpingNetwork(**model_params['warping_module_params'])
        self.generator = SPADEDecoder(**model_params['spade_generator_params'])
        self.discriminator = MultiScaleDiscriminator(**discriminator_params)
        self.discriminator_face = MultiScaleDiscriminator(**discriminator_face_params)
        self.discriminator_lip = MultiScaleDiscriminator(**discriminator_lip_params)

        if pretrained_weights is not None:
            self.load_cpk(pretrained_weights)

        self.appearance_feature_extractor.to(self.device)
        self.motion_extractor.to(self.device)
        self.warping_module.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.discriminator_face.to(self.device)
        self.discriminator_lip.to(self.device)

        # 转换为 SyncBatchNorm
        self.appearance_feature_extractor = nn.SyncBatchNorm.convert_sync_batchnorm(self.appearance_feature_extractor)
        self.warping_module = nn.SyncBatchNorm.convert_sync_batchnorm(self.warping_module)  # bn3d

        logger.info('Models initialized.')

        # 优化器初始化
        logger.info('Initializing optimizers:')
        self.optimizer_appearance_feature_extractor = torch.optim.Adam(
            self.appearance_feature_extractor.parameters(), lr=train_params['lr_appearance_feature_extractor'],
            betas=(0.5, 0.999))
        self.optimizer_motion_extractor = torch.optim.Adam(self.motion_extractor.parameters(),
                                                           lr=train_params['lr_motion_extractor'], betas=(0.5, 0.999))
        self.optimizer_warping_module = torch.optim.Adam(self.warping_module.parameters(),
                                                         lr=train_params['lr_warping_module'], betas=(0.5, 0.999))
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=train_params['lr_generator'],
                                                    betas=(0.5, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
        self.optimizer_discriminator_face = torch.optim.Adam(self.discriminator_face.parameters(),
                                                             lr=train_params['lr_discriminator_face'],
                                                             betas=(0.5, 0.999))
        self.optimizer_discriminator_lip = torch.optim.Adam(self.discriminator_lip.parameters(),
                                                            lr=train_params['lr_discriminator_lip'], betas=(0.5, 0.999))
        logger.info('Optimizers initialized.')

        # 学习率调整器
        logger.info('Initializing schedulers:')
        self.start_epoch = 0
        self.scheduler_appearance_feature_extractor = MultiStepLR(
            self.optimizer_appearance_feature_extractor,
            train_params['epoch_milestones'], gamma=0.1,
            last_epoch=-1 + self.start_epoch * (train_params['lr_appearance_feature_extractor'] != 0))
        self.scheduler_motion_extractor = MultiStepLR(self.optimizer_motion_extractor, train_params['epoch_milestones'],
                                                      gamma=0.1,
                                                      last_epoch=-1 + self.start_epoch * (
                                                              train_params['lr_motion_extractor'] != 0))
        self.scheduler_warping_module = MultiStepLR(self.optimizer_warping_module, train_params['epoch_milestones'],
                                                    gamma=0.1,
                                                    last_epoch=-1 + self.start_epoch * (
                                                            train_params['lr_warping_module'] != 0))
        self.scheduler_generator = MultiStepLR(self.optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                               last_epoch=self.start_epoch - 1)
        self.scheduler_discriminator = MultiStepLR(self.optimizer_discriminator, train_params['epoch_milestones'],
                                                   gamma=0.1, last_epoch=self.start_epoch - 1)
        self.scheduler_discriminator_face = MultiStepLR(self.optimizer_discriminator_face, train_params['epoch_milestones'],
                                                        gamma=0.1, last_epoch=self.start_epoch - 1)
        self.scheduler_discriminator_lip = MultiStepLR(self.optimizer_discriminator_lip, train_params['epoch_milestones'],
                                                       gamma=0.1, last_epoch=self.start_epoch - 1)

        logger.info('Schedulers initialized.')

        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        # 抗锯齿的二维插值下采样，旨在更好地保留输入信号。其主要工作是通过应用一个高斯核来低通滤波输入，从而在下采样时减少频率混叠效应。
        self.pyramid = ImagePyramide(self.scales, num_channels=3)
        self.pyramid.to(self.device)

        self.disc_pyramid = ImagePyramide(self.disc_scales, num_channels=3)
        self.disc_pyramid.to(self.device)

        # loss权重配置及模型加载
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            logger.info('Loading vgg19')
            self.vgg = vgg19.Vgg19().to(self.device)
            self.vgg.eval()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            logger.info('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            self.hopenet = self.hopenet.to(self.device)
            self.hopenet.eval()
        
        if self.loss_weights['face_id'] != 0:
            self.idnet = iresnet50(False, fp16=False)
            self.idnet.load_state_dict(torch.load(train_params['resnet50']))
            self.idnet = self.idnet.to(self.device)
            self.idnet.eval()

        self.zero_tensor = None

    def gen_update(self, x):
        """
        :param x: dict, contain source, driving, and other
                  source: torch.Tensor,  # (bs, 3, h, w) normalized 0~1
                  driving: torch.Tensor   # (bs, 3, h, w) normalized 0~1
        :return:
        """
        # Set models to training mode
        self.motion_extractor.train()
        self.appearance_feature_extractor.train()
        self.warping_module.train()
        self.generator.train()
        self.discriminator.eval()
        self.discriminator_face.eval()
        self.discriminator_lip.eval()

        self.motion_extractor.requires_grad_(True)
        self.appearance_feature_extractor.requires_grad_(True)
        self.warping_module.requires_grad_(True)
        self.generator.requires_grad_(True)
        self.discriminator.requires_grad_(False)
        self.discriminator_face.requires_grad_(False)
        self.discriminator_lip.requires_grad_(False)

        self.optimizer_appearance_feature_extractor.zero_grad()
        self.optimizer_motion_extractor.zero_grad()
        self.optimizer_warping_module.zero_grad()
        self.optimizer_generator.zero_grad()

        # kp_info:
        #    ret_dct = {
        #         'pitch': pitch, 'yaw': yaw, 'roll': roll, 't': t,
        #         'exp': exp, 'scale': scale, 'kp': kp,  # canonical keypoint
        #    }

        source = x['source'].to(self.device)
        driving = x['driving'].to(self.device)
        kp_info_source = self.motion_extractor(source)
        kp_info_driving = self.motion_extractor(driving)

        kp_canonical = kp_info_source['kp']
        kp_source = keypoint_transformation(kp_canonical=kp_canonical, kp_info=kp_info_source)
        kp_driving = keypoint_transformation(kp_canonical=kp_canonical, kp_info=kp_info_driving)

        # get source 3d feature
        feature_3d = self.appearance_feature_extractor(source)
        # get decoder input
        ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
        # decode
        gen_out = self.generator(feature=ret_dct['out'])

        generated = {
            'kp_canonical': kp_canonical,
            'kp_info_source': kp_info_source,
            'kp_info_driving': kp_info_driving,
            'kp_source': kp_source,
            'kp_driving': kp_driving,
            'prediction': gen_out
        }

        losses_generator = self.compute_loss(x, generated)

        loss_values = [val.mean() for val in losses_generator.values()]
        g_loss = sum(loss_values)

        self.accelerator.backward(g_loss)
        if self.accelerator.sync_gradients:
            all_params = itertools.chain(*[model.parameters() for model in
                                           [self.appearance_feature_extractor, self.motion_extractor,
                                            self.warping_module, self.generator]])
            self.accelerator.clip_grad_norm_(all_params, self.train_params['max_grad_norm'])

        self.optimizer_appearance_feature_extractor.step()
        self.optimizer_motion_extractor.step()
        self.optimizer_warping_module.step()
        self.optimizer_generator.step()

        return losses_generator, generated

    def compute_loss(self, x, generated):
        loss_values = {}
        driving = x['driving_512'].to(self.device)
        pyramide_real = self.pyramid(driving)
        pyramide_generated = self.pyramid(generated['prediction'])

        # Equivariance loss L_E
        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0],
                                  **self.train_params['transform_params'])  # x['driving']: (bs, 3, h, w) normalized 0~1
            transformed_frame = transform.transform_frame(x['driving'])

            kp_info_driving_transformed = self.motion_extractor(transformed_frame)

            transformed_kp = keypoint_transformation(kp_canonical=generated['kp_canonical'],
                                                     kp_info=kp_info_driving_transformed)

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = generated['kp_driving'][:, :, :2]
                transformed_kp_2d = transformed_kp[:, :, :2]
                value = torch.abs(transformed_kp_2d - transform.warp_coordinates(kp_driving_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

        # L keypiont loss
        if self.loss_weights['keypoint'] != 0:
            value_total = 0
            for i in range(generated['kp_driving'].shape[1]):
                for j in range(generated['kp_driving'].shape[1]):
                    dist = F.pairwise_distance(generated['kp_driving'][:, i, :], generated['kp_driving'][:, j, :], p=2,
                                               keepdim=True) ** 2
                    dist = 0.1 - dist  # set Dt = 0.1
                    dd = torch.gt(dist, 0)
                    value = (dist * dd).mean()
                    value_total += value
            kp_mean_depth = generated['kp_driving'][:, :, -1].mean(-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()  # set Zt = 0.33
            value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        # Lp_cascade: a cascaded perceptual loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                with torch.no_grad():  # Turn off gradients to save memory and computations
                    x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        # LG_cascade: a cascaded GAN loss
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])  # 生成器的训练目标使判别器以为是真实的
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                discriminator_maps_real = self.discriminator(pyramide_real)
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean() # 训练生成器使得生成的和真实的尽可能相似
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        driving_face = crop_and_resize_face_area(x['driving_512'], x['driving_lmk'] * 2.0)
        generated_face = crop_and_resize_face_area(generated['prediction'], x['driving_lmk'] * 2.0)
        pyramide_generated_face = self.pyramid(generated_face)
        pyramide_real_face = self.pyramid(driving_face)
        if self.loss_weights['generator_gan_face'] != 0:
            discriminator_face_maps_generated = self.discriminator_face(pyramide_generated_face)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_face_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_face_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))
                value_total += self.loss_weights['generator_gan_face'] * value
            loss_values['gen_gan_face'] = value_total
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                discriminator_face_maps_real = self.discriminator_face(pyramide_real_face)
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_face_maps_real[key], discriminator_face_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching_face'] = value_total

        driving_lip = crop_and_resize_lip_area(x['driving_512'], x['driving_lmk'] * 2.0)
        generated_lip = crop_and_resize_lip_area(generated['prediction'], x['driving_lmk'] * 2.0)
        pyramide_generated_lip = self.pyramid(generated_lip)
        pyramide_real_lip = self.pyramid(driving_lip)
        if self.loss_weights['generator_gan_lip'] != 0:
            discriminator_lip_maps_generated = self.discriminator_lip(pyramide_generated_lip)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_lip_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_lip_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))
                value_total += self.loss_weights['generator_gan_lip'] * value
            loss_values['gen_gan_lip'] = value_total
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                discriminator_lip_maps_real = self.discriminator_lip(pyramide_real_lip)
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_lip_maps_real[key], discriminator_lip_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching_lip'] = value_total

        #  LH: head pose loss
        if self.loss_weights['headpose'] != 0:
            transform_hopenet = transforms.Compose([transforms.Resize(size=(224, 224)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
            driving_224 = transform_hopenet(x['driving'])
            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = (generated['kp_info_driving']['yaw'], generated['kp_info_driving']['pitch'],
                                generated['kp_info_driving']['roll'])
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)
            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(
                roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        # L∆: deformation prior loss
        if self.loss_weights['expression'] != 0:
            value = torch.norm(generated['kp_info_driving']['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value

        # Landmark-guided wing loss 
        if self.loss_weights['guide'] != 0:
            # source_lmk = self.cropper.calc_lmk_from_cropped_image(x['source'])      # (b, 203, 2)
            # driving_lmk = self.cropper.calc_lmk_from_cropped_image(x['driving'])    # (b, 203, 2)

            source_lmk = x['source_lmk'][:, [6, 197, 18, 30, 198, 42, 48, 66, 102, 90], :]
            driving_lmk = x['driving_lmk'][:, [6, 197, 18, 30, 198, 42, 48, 66, 102, 90], :]
            xs = (generated['kp_source'][:, 11:21, :2] + 1) / 2 * 256
            xd = (generated['kp_driving'][:, 11:21, :2] + 1) / 2 * 256
            value = (self.wing_loss(xs, source_lmk) + self.wing_loss(xd, driving_lmk)) / 2.

            loss_values['guide'] = self.loss_weights['guide'] * value
        
        # face-id loss 
        if self.loss_weights['face_id'] != 0:
            # source: torch.Tensor,  # (bs, 3, h, w) normalized 0~1
            source_112 = F.interpolate(x['source'], size=(112, 112), mode='bilinear', align_corners=False)  # (bs, 3, 112, 112)
            pred_112 = F.interpolate(generated['prediction'], size=(112, 112), mode='bilinear', align_corners=False)
            feat1 = self.idnet(source_112.mul_(2).sub_(1))     #(b, 512)
            feat2 = self.idnet(pred_112.mul_(2).sub_(1)) # (b, 512)
            value = torch.abs(feat1-feat2).mean()
            
            loss_values['face_id'] = self.loss_weights['face_id'] * value

        return loss_values

    def wing_loss(self, landmarks, labels, w=10.0, epsilon=2.0):
        """
        Arguments:
            landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
            w, epsilon: a float numbers.
        Returns:
            a float tensor with shape [].
        """
        batch_size = landmarks.size(0)
        num_landmarks = landmarks.size(1)

        x = landmarks - labels

        # 确保 w 和 epsilon 是张量
        w_tensor = torch.tensor(w, dtype=landmarks.dtype, device=landmarks.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=landmarks.dtype, device=landmarks.device)

        c = w * (1.0 - torch.log(1.0 + w_tensor / epsilon_tensor))
        absolute_x = torch.abs(x)

        losses = torch.where(
            w > absolute_x,
            w * torch.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )

        loss = torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)
        return loss

    def dis_update(self, x, generated):
        # Ensure models are in the correct training/eval mode
        self.motion_extractor.eval()
        self.appearance_feature_extractor.eval()
        self.warping_module.eval()
        self.generator.eval()
        self.discriminator.train()
        self.discriminator_face.train()
        self.discriminator_lip.train()

        self.motion_extractor.requires_grad_(False)
        self.appearance_feature_extractor.requires_grad_(False)
        self.warping_module.requires_grad_(False)
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)
        self.discriminator_face.requires_grad_(True)
        self.discriminator_lip.requires_grad_(True)

        self.discriminator.zero_grad()
        self.discriminator_face.zero_grad()
        self.discriminator_lip.zero_grad()

        driving = x['driving_512'].to(self.device)
        pyramide_real = self.disc_pyramid(driving)
        pyramide_generated = self.disc_pyramid(generated['prediction'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.disc_scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key] - 1,
                                              self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(
                    torch.min(-discriminator_maps_generated[key] - 1,
                              self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        driving_face = crop_and_resize_face_area(x['driving_512'], x['driving_lmk'] * 2.0)
        generated_face = crop_and_resize_face_area(generated['prediction'], x['driving_lmk'] * 2.0)
        pyramide_real_face = self.disc_pyramid(driving_face)
        pyramide_generated_face = self.disc_pyramid(generated_face.detach())
        discriminator_face_maps_generated = self.discriminator_face(pyramide_real_face)
        discriminator_face_maps_real = self.discriminator_face(pyramide_generated_face)
        value_total = 0
        for scale in self.disc_scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_face_maps_real[key] - 1,
                                              self.get_zero_tensor(discriminator_face_maps_real[key]))) - torch.mean(
                    torch.min(-discriminator_face_maps_generated[key] - 1,
                              self.get_zero_tensor(discriminator_face_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_face_maps_real[key]) ** 2 + discriminator_face_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan_face'] * value
        loss_values['disc_gan_face'] = value_total

        driving_lip = crop_and_resize_lip_area(x['driving_512'], x['driving_lmk'] * 2.0)
        generated_lip = crop_and_resize_lip_area(generated['prediction'], x['driving_lmk'] * 2.0)
        pyramide_real_lip = self.disc_pyramid(driving_lip)
        pyramide_generated_lip = self.disc_pyramid(generated_lip.detach())
        discriminator_lip_maps_generated = self.discriminator_lip(pyramide_generated_lip)
        discriminator_lip_maps_real = self.discriminator_lip(pyramide_real_lip)
        value_total = 0
        for scale in self.disc_scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_lip_maps_real[key] - 1,
                                              self.get_zero_tensor(discriminator_lip_maps_real[key]))) - torch.mean(
                    torch.min(-discriminator_lip_maps_generated[key] - 1,
                              self.get_zero_tensor(discriminator_lip_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_lip_maps_real[key]) ** 2 + discriminator_lip_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan_lip'] * value
        loss_values['disc_gan_lip'] = value_total

        dis_loss = [val.mean() for val in loss_values.values()]
        d_loss = sum(dis_loss)

        self.accelerator.backward(d_loss)
        if self.accelerator.sync_gradients:
            all_params = itertools.chain(*[model.parameters() for model in
                                           [self.discriminator, self.discriminator_face, self.discriminator_lip]])
            self.accelerator.clip_grad_norm_(all_params, self.train_params['max_grad_norm'])

        self.optimizer_discriminator.step()
        self.optimizer_discriminator_face.step()
        self.optimizer_discriminator_lip.step()

        return loss_values

    def scheduler_epoch_step(self):
        self.scheduler_appearance_feature_extractor.step()
        self.scheduler_motion_extractor.step()
        self.scheduler_warping_module.step()
        self.scheduler_generator.step()
        self.scheduler_discriminator.step()
        self.scheduler_discriminator_face.step()
        self.scheduler_discriminator_lip.step()

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

        # todo： stage2
        # stitching_retargeting_module_weights_path = pretrained_weights.get('stitching_retargeting_module', None)

        discriminator_weights_path = pretrained_weights.get('discriminator', None)
        if discriminator_weights_path is not None:
            logger.info(f"load discriminator checkpoint: {discriminator_weights_path}")
            discriminator_weights = torch.load(discriminator_weights_path,
                                               map_location=lambda storage, loc: storage)
            m, u = self.discriminator.load_state_dict(discriminator_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        discriminator_face_weights_path = pretrained_weights.get('discriminator_face', None)
        if discriminator_face_weights_path is not None:
            logger.info(f"load discriminator_face checkpoint: {discriminator_face_weights_path}")
            discriminator_face_weights = torch.load(discriminator_face_weights_path,
                                                    map_location=lambda storage, loc: storage)
            m, u = self.discriminator_face.load_state_dict(discriminator_face_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        discriminator_lip_weights_path = pretrained_weights.get('discriminator_lip', None)
        if discriminator_lip_weights_path is not None:
            logger.info(f"load discriminator_face checkpoint: {discriminator_lip_weights_path}")
            discriminator_lip_weights = torch.load(discriminator_lip_weights_path,
                                                   map_location=lambda storage, loc: storage)
            m, u = self.discriminator_lip.load_state_dict(discriminator_lip_weights, strict=False)
            logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

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
            'appearance_feature_extractor': self.accelerator.unwrap_model(
                self.appearance_feature_extractor).state_dict(),
            'motion_extractor': self.accelerator.unwrap_model(self.motion_extractor).state_dict(),
            'warping_module': self.accelerator.unwrap_model(self.warping_module).state_dict(),
            'generator': self.accelerator.unwrap_model(self.generator).state_dict(),
            'discriminator': self.accelerator.unwrap_model(self.discriminator).state_dict(),
            'discriminator_face': self.accelerator.unwrap_model(self.discriminator_face).state_dict(),
            'discriminator_lip': self.accelerator.unwrap_model(self.discriminator_lip).state_dict(),

            'optimizer_appearance_feature_extractor': self.optimizer_appearance_feature_extractor.state_dict(),
            'optimizer_motion_extractor': self.optimizer_motion_extractor.state_dict(),
            'optimizer_warping_module': self.optimizer_warping_module.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
            'optimizer_discriminator_face': self.optimizer_discriminator_face.state_dict(),
            'optimizer_discriminator_lip': self.optimizer_discriminator_lip.state_dict(),

            'scheduler_appearance_feature_extractor': self.scheduler_appearance_feature_extractor.state_dict(),
            'scheduler_motion_extractor': self.scheduler_motion_extractor.state_dict(),
            'scheduler_warping_module': self.scheduler_warping_module.state_dict(),
            'scheduler_generator': self.scheduler_generator.state_dict(),
            'scheduler_discriminator': self.scheduler_discriminator.state_dict(),
            'scheduler_discriminator_face': self.scheduler_discriminator_face.state_dict(),
            'scheduler_discriminator_lip': self.scheduler_discriminator_lip.state_dict(),
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
        self.accelerator.save(checkpoint_data, checkpoint_path)
        logger.info(f'Checkpoint saved at {checkpoint_path}')
