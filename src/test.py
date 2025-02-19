import os
import torch
import matplotlib
import cv2
from tqdm.auto import tqdm
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

import datasets
from train_net import Train_net
from log_utils import log

def main(device: str,
        mode: str,
        dataset_train_params: dict,
        dataset_val_params: dict,
        model_params: dict,
        train_params: dict,
        pretrained_weights: dict,
        global_seed: int = 42 ):
    
    seed = global_seed
    torch.manual_seed(seed)

    # Preparing dataset
    val_dataset = datasets.KBNetInferenceDataset(**dataset_val_params)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_params['n_batch_test'],
        shuffle=True,
        num_workers=train_params['n_thread'],
        drop_last=False)

    trainer = Train_net(device = device,
                        mode=mode,
                        model_params = model_params,
                        train_params = train_params,
                        pretrained_weights = pretrained_weights)

    # output setting
    output_dir = os.path.dirname(pretrained_weights['model_path']).replace('checkpoints', 'test')
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'results.txt')

    # training
    test_data_length = len(val_dataloader)     # 72400 --> 3017

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    logger.info(f'Begin test ------ train data length:{test_data_length}')
    mae, rmse, imae, irmse = [], [], [], []

    for idx, inputs in enumerate(tqdm(val_dataloader, desc="Testing")):

        metrics, generated = trainer.validate(inputs)
        generated = {key: value.detach().cpu() for key, value in generated.items()}

        mae.append(metrics['mae'])
        rmse.append(metrics['rmse'])
        imae.append(metrics['imae'])
        irmse.append(metrics['irmse'])
  
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

        test_data_pair = np.concatenate((img, sd, output_depth, gt), axis=0)
        cv2.imwrite(f"{output_dir}/{f'test_img_sd_output_gt-{idx}'}.png", test_data_pair)


    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    
    log(f'test result ====> mae:{mae:.3f} rmse:{rmse:.3f} imae:{imae:.6f} irmse:{irmse:.6f}', log_path)
    
            
            



if __name__ == '__main__':

    cfg_path = '/home/sbq/codes/depth-completion/configs/kitti_test.yaml'

    args = OmegaConf.load(cfg_path)

    if torch.cuda.is_available() and args.train_params.device in ['cuda', 'gpu', 'CUDA', 'GPU']:
        device = 'cuda'
    else:
        device = 'cpu'

    main(device, 'test', **args)
