try:
    import mdl_utils  # For ppu queue, bugfix: transformers install error
except ImportError:
    pass

import copy
import os
import math
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import cv2
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple
from contextlib import nullcontext
from tqdm import trange
from loguru import logger

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from src.train.train_net import Trainer
from src.train.train_net2 import Stage2Trainer
from src.data.datasets import FramesDataset, DatasetRepeater
from src.config.crop_config import CropConfig


def write_loss(iteration, writer, losses_dict):
    for key, value in losses_dict.items():
        writer.add_scalar(key, value.item(), iteration)
    writer.flush()


def main(
        name: str,
        output_dir: str,

        model_params: dict,
        pretrained_weights: dict,
        train_params: dict,
        dataset_params: dict,
        discriminator_params=None,
        discriminator_face_params: dict = None,
        discriminator_lip_params: dict = None,

        eval_and_save_freq: int = 10000,
        global_seed: int = 42,
        is_debug: bool = False,
        train_stage: int = 1
):
    # 环境设置
    rank = int(os.environ.get('RANK', '0'))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = global_seed + int(rank)
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    insightface_root = pretrained_weights.get('insightface_root', None)
    if insightface_root is not None:
        CropConfig.insightface_root = insightface_root
    landmark_ckpt_path = pretrained_weights.get('landmark_ckpt_path', None)
    if landmark_ckpt_path is not None:
        CropConfig.landmark_ckpt_path = landmark_ckpt_path

    # 训练数据
    train_dataset = FramesDataset(is_train=True, crop_cfg=CropConfig, **dataset_params)
    logging.info(f"num_workers: {train_params['num_workers']}")

    # DataLoaders creation:
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        train_dataset = DatasetRepeater(train_dataset, train_params['num_repeats'])
    dataloader = DataLoader(train_dataset,
                            batch_size=train_params['batch_size'],
                            shuffle=True,
                            num_workers=train_params['num_workers'],
                            pin_memory=True,
                            drop_last=True)

    if train_stage == 1:
        trainer = Trainer(model_params=model_params,
                          discriminator_params=discriminator_params,
                          discriminator_face_params=discriminator_face_params,
                          discriminator_lip_params=discriminator_lip_params,
                          train_params=train_params, pretrained_weights=pretrained_weights)

        # 使用accelerator.prepare替换所有数据加载器、模型和优化器
        prepared_elements = trainer.accelerator.prepare(trainer.appearance_feature_extractor,
                                                        trainer.motion_extractor,
                                                        trainer.warping_module,
                                                        trainer.generator,
                                                        trainer.discriminator,
                                                        trainer.discriminator_face,
                                                        trainer.discriminator_lip,
                                                        trainer.optimizer_appearance_feature_extractor,
                                                        trainer.optimizer_motion_extractor,
                                                        trainer.optimizer_warping_module,
                                                        trainer.optimizer_generator,
                                                        trainer.optimizer_discriminator,
                                                        trainer.optimizer_discriminator_face,
                                                        trainer.optimizer_discriminator_lip,
                                                        trainer.scheduler_appearance_feature_extractor,
                                                        trainer.scheduler_motion_extractor,
                                                        trainer.scheduler_warping_module,
                                                        trainer.scheduler_generator,
                                                        trainer.scheduler_discriminator,
                                                        trainer.scheduler_discriminator_face,
                                                        trainer.scheduler_discriminator_lip,
                                                        dataloader)

        (trainer.appearance_feature_extractor,
         trainer.motion_extractor,
         trainer.warping_module,
         trainer.generator,
         trainer.discriminator,
         trainer.discriminator_face,
         trainer.discriminator_lip,
         trainer.optimizer_appearance_feature_extractor,
         trainer.optimizer_motion_extractor,
         trainer.optimizer_warping_module,
         trainer.optimizer_generator,
         trainer.optimizer_discriminator,
         trainer.optimizer_discriminator_face,
         trainer.optimizer_discriminator_lip,
         trainer.scheduler_appearance_feature_extractor,
         trainer.scheduler_motion_extractor,
         trainer.scheduler_warping_module,
         trainer.scheduler_generator,
         trainer.scheduler_discriminator,
         trainer.scheduler_discriminator_face,
         trainer.scheduler_discriminator_lip,
         dataloader) = prepared_elements
    else:
        trainer = Stage2Trainer(model_params=model_params,
                                train_params=train_params,
                                pretrained_weights=pretrained_weights)

        trainer.stitcher, trainer.retargetor_lip, trainer.retargetor_eye = trainer.accelerator.prepare(
            trainer.stitcher, trainer.retargetor_lip, trainer.retargetor_eye)

    # Handle the output folder creation
    if trainer.accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if trainer.accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir=output_dir, max_queue=30, flush_secs=120)
    else:
        tb_writer = None

    train_data_length = len(dataloader)
    max_train_steps = train_params['num_epochs'] * train_data_length
    global_step = 0
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not trainer.accelerator.is_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(0, train_params['num_epochs']):
        for step, x in enumerate(dataloader):
            if trainer.accelerator.is_main_process and epoch == 0 and step < 10:
                for idx, (source_image, driving_image) in enumerate(zip(x['source'], x['driving'])):
                    # concat source and driving image
                    train_data_pair = torch.cat([source_image, driving_image], dim=2)
                    torchvision.utils.save_image(train_data_pair,
                                                 f"{output_dir}/sanity_check/{f'train_data_pair-{idx}-{step}'}.png")

            if train_stage == 1:
                losses_generator, generated = trainer.gen_update(x)
                losses_discriminator = trainer.dis_update(x, generated)
                losses_generator.update(losses_discriminator)
            else:
                losses_generator, generated = trainer(x)

            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}

            if trainer.accelerator.sync_gradients and trainer.accelerator.is_main_process:
                progress_bar.update(1)
                global_step += 1
                iteration = epoch * train_data_length + step

                # write to tensorboard
                write_loss(iteration, tb_writer, losses)

                if iteration % eval_and_save_freq == 0:
    
                    gen_out_sts = F.interpolate(generated['gen_out_st'].detach(), size=(256, 256), mode='bilinear', align_corners=False)
                    gen_out_eyes = F.interpolate(generated['gen_out_eye'].detach(), size=(256, 256), mode='bilinear', align_corners=False)
                    gen_out_lips = F.interpolate(generated['gen_out_lip'].detach(), size=(256, 256), mode='bilinear', align_corners=False)

                                            
                    for idx, (source_image, driving_image, gen_out_st, gen_out_eye, gen_out_lip, c_d_eye, c_d_lip) in enumerate(
                            zip(x['source'], x['driving'], gen_out_sts, gen_out_eyes, gen_out_lips, generated['c_d_eye'], generated['c_d_lip'])):
                        # concat source and driving image
                        gen_out_eye_np = cv2.cvtColor(255 * gen_out_eye.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        gen_out_eye_np_text = cv2.putText(gen_out_eye_np, f'c_d_eye={c_d_eye.item():.4f}', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                     
                        gen_out_lip_np = cv2.cvtColor(255 * gen_out_lip.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        gen_out_lip_np_text = cv2.putText(gen_out_lip_np, f'c_d_lip={c_d_lip.item():.4f}', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        source_image_np = cv2.cvtColor(255 * source_image.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        source_image_np_text = cv2.putText(source_image_np, 'source', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        driving_image_np = cv2.cvtColor(255 * driving_image.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        driving_image_np_text = cv2.putText(driving_image_np, 'driving', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        gen_out_st_np = cv2.cvtColor(255 * gen_out_st.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        gen_out_st_np_text = cv2.putText(gen_out_st_np, f'stiching', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        concatenated_img = np.hstack((source_image_np_text, driving_image_np_text, gen_out_st_np_text, gen_out_eye_np_text, gen_out_lip_np_text))
                        cv2.imwrite(f"{output_dir}/samples/{f'val-{idx}-{iteration}'}.png", concatenated_img)
                        
                    trainer.save_checkpoint(f"{output_dir}/checkpoints", iteration)

            progress_bar.set_postfix(**{k: f"{v:.3f}" for k, v in losses.items()})

        trainer.scheduler_epoch_step()
        trainer.accelerator.wait_for_everyone()

    trainer.accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    model_config = OmegaConf.load("src/config/models.yaml")
    config.update(model_config)

    main(name=config_name, **config)
