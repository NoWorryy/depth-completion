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
                        model_params = model_params,
                        train_params = train_params,
                        pretrained_weights = pretrained_weights)

    # trainer = torch.nn.DataParallel(trainer)
    trainer.depth_anything, trainer.scale_model, trainer.optimizer_scale_modell, dataloader, val_dataloader = trainer.accelerator.prepare(
        trainer.depth_anything, trainer.scale_model, trainer.optimizer_scale_model, dataloader, val_dataloader)
    # trainer.accelerator.register_for_checkpointing(trainer.scheduler_scale_model)

    # output setting
    folder_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(train_params['output_dir'], folder_name)
    event_path = os.path.join(output_dir, 'events')
    log_path = os.path.join(output_dir, 'results.txt')

    # training
    train_data_length = len(dataloader)     # 72400 --> 3017
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
                for idx, (image, sparse_depth, gt) in \
                    enumerate(zip(inputs['image'], inputs['sparse_depth'], inputs['gt'])):

                    # concat source and driving image
                    train_data_pair = torch.cat([image, sparse_depth.repeat(3,1,1), gt.repeat(3,1,1)], dim=1)
                    torchvision.utils.save_image(train_data_pair, f"{output_dir}/input_image/{f'train_data_pair-step{step}-{idx}'}.png")
                    
            loss_info, generated = trainer(inputs)

            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_info.items()}
            generated = {key: value.detach().cpu() for key, value in generated.items()}
            
            iteration = epoch * train_data_length + step
            
            if trainer.accelerator.sync_gradients and trainer.accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(**{k: f"{v:.3f}" for k, v in losses.items()})
                





if __name__ == '__main__':

    cfg_path = '/home/sbq/codes/depth-completion/configs/kitti_train.yaml'
    # with open(cfg_path, 'r') as fp:
    #     args = yaml.safe_load(fp)

    args = OmegaConf.load(cfg_path)


    if torch.cuda.is_available() and args.train_params.device in ['cuda', 'gpu', 'CUDA', 'GPU']:
        device = 'cuda'
    else:
        device = 'cpu'

    main(device, **args)
