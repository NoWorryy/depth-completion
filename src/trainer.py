import os
import torch
import torchvision
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

        dataset_train_params: dict,
        dataset_val_params: dict,
        model_params: dict,
        train_params: dict,
        pretrained_weights: dict):


    # output setting
    folder_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(train_params['output_dir'], folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/output_image", exist_ok=True)
    OmegaConf.save(args, os.path.join(output_dir, 'config.yaml'))
    
    event_path = os.path.join(output_dir, 'events')
    log_path = os.path.join(output_dir, 'results.txt')
    
    # Set up tensorboard summary writers
    tb_writer = SummaryWriter(log_dir = event_path)

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
                        model_params = model_params,
                        train_params = train_params,
                        pretrained_weights = pretrained_weights)

    trainer = torch.nn.DataParallel(trainer)

    # training
    train_data_length = len(dataloader)     # 72400 --> 3017
    start_epoch = trainer.module.iter // train_data_length
    max_epoch = train_params['learning_schedule'][-1]
    max_train_steps = max_epoch * train_data_length
    global_step = trainer.module.iter
    
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    logger.info(f'Begin training ------ train data length:{train_data_length}, max epoch:{max_epoch}, start_epoch:{start_epoch}')
    for epoch in range(start_epoch, max_epoch):
        logger.info(f'================> current epoch: {epoch}')
        # Set augmentation schedule
        if epoch > train_params['augmentation_schedule'][trainer.module.augmentation_schedule_pos]:    # [50, 55, 60]
            trainer.module.augmentation_schedule_pos = trainer.module.augmentation_schedule_pos + 1
            trainer.module.augmentation_probability = train_params['augmentation_probabilities'][trainer.module.augmentation_schedule_pos]

        for step, inputs in enumerate(dataloader):
            
            loss_info, generated = trainer.module.train(inputs)

            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_info.items()}
            generated = {key: value.detach().cpu() for key, value in generated.items()}
            
            progress_bar.update(1)
            global_step += 1
            iteration = epoch * train_data_length + step

            # write to tensorboard
            write_loss(iteration, tb_writer, losses)


            if (iteration % train_params['n_summary']) == 0:
                trainer.module.save_checkpoint(os.path.join(output_dir, 'checkpoints'), iteration)

                # 保存图像 这里都是(b, c, h, w)
                sparse_depth0_color = colorize((inputs['sparse_depth0'] / model_params['depth_model_params']['max_predict_depth']).cpu(), colormap='viridis')
                output_depth0_color = colorize((generated['output_depth0'] / model_params['depth_model_params']['max_predict_depth']).cpu(), colormap='viridis')
                
                train_data_pair = torch.cat([
                    sparse_depth0_color, inputs['image0'], output_depth0_color, generated['image01'], generated['image02']],
                    dim = 2)
                torchvision.utils.save_image(train_data_pair[:5, ...], f"{output_dir}/output_image/{f'train_data_pair-{iteration}'}.png")
                tb_writer.add_images("train_sparse_image_dense_image01_image02", train_data_pair[:5, ...], global_step=iteration)


                mae, rmse, imae, irmse = [], [], [], []

                for idx, inputs in enumerate(val_dataloader):
                    # inputs = {key: value.to(device) for key, value in inputs.items()}
                    metrics, generated_val = trainer.module.validate(inputs)
                    generated_val = {key: value.detach().cpu() for key, value in generated_val.items()}

                    mae.append(metrics['mae'])
                    rmse.append(metrics['rmse'])
                    imae.append(metrics['imae'])
                    irmse.append(metrics['irmse'])

                    if idx == 0:    # 保存第一个batch的图片
                        sparse_depth_color = colorize((inputs['sparse_depth'] / model_params['depth_model_params']['max_predict_depth']).cpu(), colormap='viridis')
                        output_depth_color = colorize((generated_val['output_depth'] / model_params['depth_model_params']['max_predict_depth']).cpu(), colormap='viridis')
                        
                        gt_error_abs = torch.abs(generated_val['output_depth'] - inputs['ground_truth'])
                        gt_error_rel = torch.where(
                            inputs['gt_mask'],
                            (gt_error_abs + EPSILON) / (inputs['ground_truth'] + EPSILON),
                            torch.zeros_like(gt_error_abs))
                        
                        gt_error_color = colorize(gt_error_rel.cpu(), colormap='inferno')

                        val_data_pair = torch.cat([
                            sparse_depth_color, inputs['image'], output_depth_color, gt_error_color],
                            dim = 2)
                        torchvision.utils.save_image(val_data_pair[:5, ...], f"{output_dir}/output_image/{f'val_data_pair-{iteration}'}.png")
                        tb_writer.add_images("val_sparse_image_dense_error", val_data_pair[:5, ...], global_step=iteration)

                mae   = np.mean(mae)
                rmse  = np.mean(rmse)
                imae  = np.mean(imae)
                irmse = np.mean(irmse)
                
                log(f'epoch:{epoch} iter:{iteration} ====> mae:{mae:.3f} rmse:{rmse:.3f} imae:{imae:.3f} irmse:{irmse:.3f}', log_path)

            progress_bar.set_postfix(**{k: f"{v:.3f}" for k, v in losses.items()})
        
        trainer.module.scheduler_epoch_step()


if __name__ == '__main__':

    cfg_path = '/media/data2/libihan/codes/calibrated-backprojection-network/configs/kitti_train.yaml'
    # with open(cfg_path, 'r') as fp:
    #     args = yaml.safe_load(fp)

    args = OmegaConf.load(cfg_path)


    if torch.cuda.is_available() and args.train_params.device in ['cuda', 'gpu', 'CUDA', 'GPU']:
        device = 'cuda'
    else:
        device = 'cpu'

    main(device, **args)
