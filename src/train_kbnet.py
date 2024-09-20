import os
import argparse
import yaml
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


import datetime
import datasets
from kbnet import train
from kbnet_model import KBNetModel
from posenet_model import PoseNetModel
from train_net import Train_net

from log_utils import log
from loguru import logger

def write_loss(iteration, writer, losses_dict):
    for key, value in losses_dict.items():
        writer.add_scalar(key, value.item(), iteration)
    writer.flush()


def main(device: str,

        dataset_params: dict,
        dataset_val_params: dict,
        model_params: dict,
        train_params: dict,
        output_params: dict):


    # output setting
    folder_name = datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_params.output_dir, folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    OmegaConf.save(args, os.path.join(output_dir, 'config.yaml'))
    
    depth_model_checkpoint_path = os.path.join(output_dir, 'checkpoints', 'depth_model-{}.pth')
    pose_model_checkpoint_path = os.path.join(output_dir, 'checkpoints', 'pose_model-{}.pth')
    log_path = os.path.join(output_dir, 'results.txt')
    event_path = os.path.join(output_dir, 'events')
    
    # Set up tensorboard summary writers
    tb_writer = SummaryWriter(log_dir = event_path)

    # Preparing dataset
    train_dataset = datasets.KBNetTrainingDataset(**dataset_params)
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
        shuffle=False,
        num_workers=train_params['n_thread'],
        drop_last=False)


    # # Build KBNet (depth) network
    # depth_model = KBNetModel(device=device, **model_params)


    # # Bulid PoseNet (only needed for training) network
    # pose_model = PoseNetModel(
    #     encoder_type='resnet18',
    #     rotation_parameterization='axis',
    #     weight_initializer=weight_initializer,
    #     activation_func='relu',
        # device=device)
    
    trainer = Train_net(model_params=model_params,
                        train_params = train_params,
                        output_params = output_params)

    # training
    train_data_length = len(dataloader)
    max_train_steps = train_params['learning_schedule'][-1] * train_data_length
    global_step = 0
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    logger.info('Begin training...')
    for epoch in range(0, train_params['learning_schedule'][-1]):
        for step, inputs in enumerate(dataloader):
            
            losses_generator, generated = trainer(inputs)

            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
            
            progress_bar.update(1)
            global_step += 1
            iteration = epoch * train_data_length + step

            # write to tensorboard
            write_loss(iteration, tb_writer, losses)


            if (train_step % n_summary) == 0:
                image01 = loss_info.pop('image01')
                image02 = loss_info.pop('image02')

       

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:


                if train_step >= validation_start_step and validation_available:
                    # Switch to validation mode
                    depth_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            depth_model=depth_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            outlier_removal=outlier_removal,
                            ground_truths=ground_truths,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=n_summary_display,
                            log_path=log_path)

                    # Switch back to training
                    depth_model.train()

                # Save checkpoints
                depth_model.save_model(
                    depth_model_checkpoint_path.format(train_step), train_step, optimizer)

                pose_model.save_model(
                    pose_model_checkpoint_path.format(train_step), train_step, optimizer)

    # Save checkpoints
    depth_model.save_model(
        depth_model_checkpoint_path.format(train_step), train_step, optimizer)

    pose_model.save_model(
        pose_model_checkpoint_path.format(train_step), train_step, optimizer)


if __name__ == '__main__':

    cfg_path = '/media/data2/libihan/codes/calibrated-backprojection-network/configs/kitti_train.yaml'
    # with open(cfg_path, 'r') as fp:
    #     args = yaml.safe_load(fp)

    args = OmegaConf.load(cfg_path)

    assert len(args.train_params.learning_rates) == len(args.train_params.learning_schedule)

    if torch.cuda.is_available() and args.train_params.device in ['cuda', 'gpu', 'CUDA', 'GPU']:
        device = 'cuda'
    else:
        device = 'cpu'

    main(device, **args)
