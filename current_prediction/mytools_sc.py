"""
流场预测工具函数
复用 SSH 预测中的地转流计算和 MSE loss，并添加流场专用工具
"""
from ssh_prediction.mytools import (
    set_all_seeds,
    convert_configs,
    compute_geostrophic_current,
    compute_gradients_exact,
    compute_gradients_sobel,
    compute_f_and_sigmoid_weight,
    MaskPearsonCorr,
    MaskPearsonCorrNP,
    unpatchify_with_batch,
    reverse_schedule_sampling,
)
from ssh_prediction.mytools import MSELossIgnoreNaN
import torch
import torch.nn as nn
import numpy as np
import math


class MSELossIgnoreNaNCurrent(nn.Module):
    """
    流场预测专用 MSE Loss，支持 2 通道 (uo, vo) 的 NaN 忽略
    """
    def __init__(self, config, mask_valid: torch.Tensor = None):
        super().__init__()
        self.mse_func = nn.MSELoss(reduction="sum")
        if mask_valid is not None:
            self.mask_valid = mask_valid[None, None, None, :, :].to(device=config.device)

    def forward(self, pred, target):
        if hasattr(self, "mask_valid"):
            mask_valid = self.mask_valid.expand_as(pred)
        else:
            mask_valid = ~(torch.isnan(target) | torch.isinf(target))

        valid_count = mask_valid.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(mask_valid, target, pred)
        loss = self.mse_func(pred, target) / valid_count
        return loss


def compute_geostrophic_current_gpu(pred, lon, lat, if_solid_f=True):
    """
    GPU 版地转流计算，直接在 GPU 上执行梯度计算
    pred: (B, T, 2, H, W) - [true_ssh, pred_ssh] concatenated along time dim
    返回 u, v (B, T, 2, H, W), w (1, 1, 1, H, W)
    """
    g = 9.81
    f, f_weight = compute_f_and_sigmoid_weight(lat, if_solid_f=if_solid_f)
    f = f.to(pred.device)
    f_weight = f_weight.to(pred.device)

    grad_x, grad_y = compute_gradients_sobel(pred, lon, lat, R_E=6.371e6)

    u_geo = - (g / f) * grad_y
    v_geo = (g / f) * grad_x

    return u_geo, v_geo, f_weight
