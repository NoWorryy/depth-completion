import os
import torch
import torchvision
import matplotlib
import cv2
from tqdm.auto import tqdm
import datetime
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import datasets
from train_net import Train_net
from log_utils import log, colorize

EPSILON = 1e-8

def write_loss(iteration, writer, losses_dict):
    for key, value in losses_dict.items():
        writer.add_scalar(key, value.item(), iteration)
    writer.flush()


def main(device: str,
        mode: str,
        dataset_train_params: dict,
        dataset_val_params: dict,
        model_params: dict,
        train_params: dict,
        pretrained_weights: dict,
        global_seed: int = 42 ):
    
    rank = int(os.environ.get('RANK', '0'))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = global_seed + int(rank)
    torch.manual_seed(seed)


    # Preparing dataset
    train_dataset = datasets.KBNetTrainingDataset(**dataset_train_params)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_params['n_batch'],
        shuffle=True,
        num_workers=train_params['n_thread'],
        drop_last=False)
    
    val_dataset = datasets.KBNetInferenceDataset(**dataset_val_params)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_params['n_batch'],
        shuffle=True,
        num_workers=train_params['n_thread'],
        drop_last=False)


    trainer = Train_net(device = device,
                        mode=mode,
                        model_params = model_params,
                        train_params = train_params,
                        pretrained_weights = pretrained_weights)

    # trainer = torch.nn.DataParallel(trainer)
    trainer.depth_anything, trainer.scale_model, trainer.optimizer_scale_model, dataloader, val_dataloader = trainer.accelerator.prepare(
        trainer.depth_anything, trainer.scale_model, trainer.optimizer_scale_model, dataloader, val_dataloader)
    # trainer.accelerator.register_for_checkpointing(trainer.scheduler_scale_model)

    if torch.cuda.device_count() > 1:
        trainer.depth_anything = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.depth_anything)
        trainer.scale_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.scale_model)

    # output setting
    folder_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(train_params['output_dir'], folder_name)
    event_path = os.path.join(output_dir, 'events')
    log_path = os.path.join(output_dir, 'results.txt')

    # training
    train_data_length = len(dataloader)     # 72400 --> 3017
    # train_data_length_bs7 = 85898 // 21 + 1
    start_epoch = trainer.iter // train_data_length
    max_epoch = train_params['learning_schedule'][-1]
    max_train_steps = max_epoch * train_data_length
    iteration = trainer.iter

    if trainer.accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/output_image", exist_ok=True)
        os.makedirs(f"{output_dir}/input_image", exist_ok=True)
        OmegaConf.save(args, os.path.join(output_dir, 'config.yaml'))

        tb_writer = SummaryWriter(log_dir = event_path)

        progress_bar = tqdm(range(iteration, max_train_steps))
        progress_bar.set_description("Steps")

        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    else:
        tb_writer = None
    

    logger.info(f'Begin training ------ train data length:{train_data_length}, max epoch:{max_epoch}, start_epoch:{start_epoch}')
    first_epoch = True
    for epoch in range(start_epoch, max_epoch):
        
        # Set augmentation schedule
        if epoch == train_params['augmentation_schedule'][trainer.augmentation_schedule_pos]:    # [50, 55, 60]
            trainer.augmentation_schedule_pos = trainer.augmentation_schedule_pos + 1
            trainer.augmentation_probability = train_params['augmentation_probabilities'][trainer.augmentation_schedule_pos]

        logger.info(f'===========================> current epoch: {epoch}, augmentation_probability: {trainer.augmentation_probability}')

        for step, inputs in enumerate(dataloader):
            
            if trainer.accelerator.is_main_process and first_epoch and step < 10:
                first_epoch = False
                for idx, (image, sparse_depth, prefill_depth, gt) in \
                    enumerate(zip(inputs['image'], inputs['sparse_depth'], inputs['prefill_depth'], inputs['gt'])):
                    # concat source and driving image
                    sparse_depth = (sparse_depth - sparse_depth.min()) / (sparse_depth.max() - sparse_depth.min())
                    gt = (gt - gt.min()) / (gt.max() - gt.min())
                    train_data_pair = torch.cat([image, sparse_depth.repeat(3,1,1), gt.repeat(3,1,1)], dim=1)
                    torchvision.utils.save_image(train_data_pair, f"{output_dir}/input_image/{f'train_data_pair-step{step}-{idx}'}.png")
                    
            loss_info, generated = trainer(inputs)

            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_info.items()}
            generated = {key: value.detach().cpu() for key, value in generated.items()}
            
            iteration = epoch * train_data_length + step
            
            if trainer.accelerator.sync_gradients and trainer.accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(**{k: f"{v:.3f}" for k, v in losses.items()})
                

                tb_writer.add_scalar('lr', trainer.optimizer_scale_model.param_groups[0]['lr'], iteration)

                # write to tensorboard
                write_loss(iteration, tb_writer, losses)
                
                if (iteration % train_params['n_ckpt']) == 0:
                    trainer.save_checkpoint(os.path.join(output_dir, 'checkpoints'), iteration)

                if (iteration % train_params['n_img']) == 0:

                    # 保存图像 这里都是(b, c, h, w)
                    img = inputs['image'][0].permute(1,2,0).detach().cpu().numpy() * 255.0

                    sd = inputs['sparse_depth'][0].squeeze(0).detach().cpu().numpy()
                    sd = (sd - sd.min()) / (sd.max() - sd.min()) * 255.0
                    sd = (cmap(sd.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    # rel_depth = generated['rel_depth'][0].squeeze(0).detach().cpu().numpy()
                    # rel_depth = (rel_depth - rel_depth.min()) / (rel_depth.max() - rel_depth.min()) * 255.0
                    # rel_depth = (cmap(rel_depth.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    output_depth = generated['output_depth'][0].squeeze(0).detach().cpu().numpy()
                    output_depth = (output_depth - output_depth.min()) / (output_depth.max() - output_depth.min()) * 255.0
                    output_depth = (cmap(output_depth.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    # output_scale = generated['output_scale'][0].squeeze(0).detach().cpu().numpy()
                    # output_scale = (output_scale - output_scale.min()) / (output_scale.max() - output_scale.min()) * 255.0
                    # output_scale = (cmap(output_scale.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    gt = inputs['gt'][0].squeeze(0).detach().cpu().numpy()
                    gt = (gt - gt.min()) / (gt.max() - gt.min()) * 255.0
                    gt = (cmap(gt.astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    train_data_pair = np.concatenate((img, sd, output_depth, gt), axis=0)
                    cv2.imwrite(f"{output_dir}/output_image/{f'train_img_sd_gt-{iteration}'}.png", train_data_pair)
                
                for name, param in trainer.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm()
                        if grad_norm < 1e-6:
                            print(f"Gradient vanishing detected at layer {name}")
                        if grad_norm > 1e6:
                            print(f"Gradient explosion detected at layer {name}")
            
        
        trainer.scheduler_epoch_step()
        trainer.accelerator.wait_for_everyone()
    
    trainer.accelerator.end_training()


if __name__ == '__main__':

    cfg_path = '/home/sbq/codes/depth-completion/configs/kitti_train.yaml'


    args = OmegaConf.load(cfg_path)


    if torch.cuda.is_available() and args.train_params.device in ['cuda', 'gpu', 'CUDA', 'GPU']:
        device = 'cuda'
    else:
        device = 'cpu'

    main(device, 'train', **args)
