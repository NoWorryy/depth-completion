'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import numpy as np
import torch

def evaluate(gt, pred, mode=None):
    t_valid = 0.0001

    pred = pred.detach()
    gt = gt.detach()

    pred_inv = 1.0 / (pred + 1e-8)
    gt_inv = 1.0 / (gt + 1e-8)

    # For numerical stability
    mask = gt > t_valid
    num_valid = mask.sum()

    pred = pred[mask] * 1000.0
    gt = gt[mask] * 1000.0

    pred_inv = pred_inv[mask] * 1000
    gt_inv = gt_inv[mask] * 1000

    pred_inv[pred <= t_valid*1000] = 0.0
    gt_inv[gt <= t_valid*1000] = 0.0

    # RMSE / MAE
    diff = pred - gt
    diff_abs = torch.abs(diff)
    diff_sqr = torch.pow(diff, 2)

    rmse = diff_sqr.sum() / (num_valid + 1e-8)
    rmse = torch.sqrt(rmse)

    mae = diff_abs.sum() / (num_valid + 1e-8)

    # iRMSE / iMAE
    diff_inv = pred_inv - gt_inv
    diff_inv_abs = torch.abs(diff_inv)
    diff_inv_sqr = torch.pow(diff_inv, 2)

    irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
    irmse = torch.sqrt(irmse)

    imae = diff_inv_abs.sum() / (num_valid + 1e-8)

    # Rel
    rel = diff_abs / (gt + 1e-8)
    rel = rel.sum() / (num_valid + 1e-8)

    # delta
    r1 = gt / (pred + 1e-8)
    r2 = pred / (gt + 1e-8)
    ratio = torch.max(r1, r2)

    del_1 = (ratio < 1.25).type_as(ratio)
    del_2 = (ratio < 1.25**2).type_as(ratio)
    del_3 = (ratio < 1.25**3).type_as(ratio)
    del_102 = (ratio < 1.02).type_as(ratio)
    del_105 = (ratio < 1.05).type_as(ratio)
    del_110 = (ratio < 1.10).type_as(ratio)

    del_1 = del_1.sum() / (num_valid + 1e-8)
    del_2 = del_2.sum() / (num_valid + 1e-8)
    del_3 = del_3.sum() / (num_valid + 1e-8)
    del_102 = del_102.sum() / (num_valid + 1e-8)
    del_105 = del_105.sum() / (num_valid + 1e-8)
    del_110 = del_110.sum() / (num_valid + 1e-8)

    result = {'rmse': rmse, 'mae': mae, 'irmse': irmse, 'imae': imae,
              'rel': rel, 'del_1': del_1, 'del_2':  del_2, 'del_3': del_3, 'del_102': del_102, 'del_105': del_105, 'del_110': del_110}

    # result = torch.unsqueeze(result, dim=0).detach()
    result = {key: value.detach().cpu() for key, value in result.items()}

    return result
