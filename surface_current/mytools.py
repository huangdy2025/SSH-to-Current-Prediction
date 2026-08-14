import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

import random
import os
from torch_npu.contrib import transfer_to_npu
from torch_npu.npu import amp
from torch_npu.npu.amp import autocast
import math
from configs import get_my_config, parse_args


class ConfigObject:
    """将字典转换为对象的包装类"""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def convert_configs(configs):
    """检查configs是否为字典，如果是则转换为对象"""
    if isinstance(configs, dict):
        return ConfigObject(configs)
    return configs


def set_all_seeds(seed):
    """
    设置所有随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unpatchify_with_batch(patched_tensor, patch_size, original_channels):
    """
    对带 Batch 的 patchified tensor 进行还原，恢复为原始 (B, T, C, H, W) 格式。
    """
    B, T, Cp2, H_, W_ = patched_tensor.shape
    p = patch_size
    C = original_channels

    assert Cp2 == C * p * p, f"通道数不匹配：{Cp2} != {C} * {p} * {p}"

    x = patched_tensor.reshape(B, T, C, p, p, H_, W_)
    x = x.permute(0, 1, 2, 5, 3, 6, 4)
    x = x.reshape(B, T, C, H_ * p, W_ * p)

    return x


class MSELossIgnoreNaN(nn.Module):
    def __init__(self, config, mask_valid=None, patched=False):
        super().__init__()
        self.mse_func = nn.MSELoss(reduction="sum")

        if mask_valid is not None:
            self.mask_valid = mask_valid[None, None, None, :, :].to(
                device=config.device
            )
            H, W = mask_valid.shape
            print(f"mask_valid: {self.mask_valid.shape}")
            if patched:
                C = config.output_channels
                p = config.patch_size
                self.mask_valid = self.mask_valid.expand(1, 1, C, H, W)
                self.mask_valid = self.mask_valid.reshape(C, H // p, p, W // p, p)
                self.mask_valid = self.mask_valid.permute(0, 2, 4, 1, 3).reshape(
                    1, 1, C * p * p, H // p, W // p
                )
                print(f"after patched: {self.mask_valid.shape}")

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


class MaskPearsonCorr(nn.Module):
    def __init__(self, mask_valid: torch.Tensor = None):
        super().__init__()
        self.mask_valid = mask_valid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        if self.mask_valid is not None:
            mask = self.mask_valid.view(-1)
        else:
            mask = ~torch.isnan(target_flat)
        pred_masked = pred_flat[mask]
        target_masked = target_flat[mask]

        if pred_masked.numel() == 0:
            return torch.tensor(float("nan"))

        pred_mean = pred_masked.mean()
        target_mean = target_masked.mean()

        pred_centered = pred_masked - pred_mean
        target_centered = target_masked - target_mean

        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(
            torch.sum(pred_centered**2) * torch.sum(target_centered**2)
        )

        epsilon = 1e-8
        corr = numerator / (denominator + epsilon)
        return corr


def compute_gradients_sobel(sla, lon, lat, R_E=6371000.0, is_real_gradient=True):
    """
    使用Sobel算子计算梯度，输入sla尺寸为 [B, T, C, H, W]
    """
    if (
        lon.ndim == 2
        and np.all(lon == lon[0, :][None, :])
        and np.all(lat == lat[:, 0][:, None])
    ):
        lon = lon[0, :]
        lat = lat[:, 0]
    B, T, C, H, W = sla.shape
    x = sla.reshape(B * T, C, H, W)

    kernel_sobel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
        )
        / 8.0
    )
    kernel_sobel_y = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
        )
        / 8.0
    )

    kernel_sobel_x = kernel_sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    kernel_sobel_y = kernel_sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    x_padded = F.pad(x, pad=(1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(x_padded, kernel_sobel_x, groups=C)
    grad_y = F.conv2d(x_padded, kernel_sobel_y, groups=C)

    if is_real_gradient:
        delta_lat = lat[1] - lat[0]
        delta_lon = lon[1] - lon[0]
        dy = delta_lat * (np.pi / 180) * R_E
        lat_rad = np.deg2rad(lat)
        dx_per_row = (
            delta_lon * (np.pi / 180) * R_E * np.cos(lat_rad)
        )
        dx_tensor = torch.tensor(dx_per_row, dtype=x.dtype, device=x.device).view(
            1, 1, H, 1
        )

        grad_x_phys = grad_x / dx_tensor
        grad_y_phys = grad_y / dy

        grad_x_phys = grad_x_phys.view(B, T, C, H, W)
        grad_y_phys = grad_y_phys.view(B, T, C, H, W)

        return grad_x_phys, grad_y_phys
    else:
        return grad_x, grad_y


def compute_f_and_sigmoid_weight(lat, k=2, phi0=5, if_solid_f=False):
    """
    基于纬度计算科氏参数 f 和 sigmoid 型的 f_weight
    """
    Omega = 7.2921e-5
    lat_t = torch.from_numpy(lat).float()
    phi = torch.abs(lat_t)

    if if_solid_f:
        f = 2 * Omega * torch.sin(torch.deg2rad(torch.mean(lat_t)))
        f_weight = 1.0
        return f, f_weight

    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))
    f = f[None, None, None, :, :]

    f_weight = 1 / (1.0 + torch.exp(-k * (phi - phi0)))
    f_weight = f_weight[None, None, None, :, :]

    return f, f_weight


def compute_f_and_gaussian_weight(lat, theta=2.2):
    Omega = 7.2921e-5
    lat_t = torch.from_numpy(lat).float()
    phi = torch.abs(lat_t)

    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))
    f = f[None, None, None, :, :]

    f_weight = 1 - torch.exp(-((phi / theta) ** 2))
    f_weight = f_weight[None, None, None, :, :]

    return f, f_weight


def compute_geostrophic_current(pred, lon, lat, if_solid_f=False):
    """
    基于空间 f 计算地转流速度（物理单位 m/s）。
    """
    g = 9.81

    f, f_weight = compute_f_and_sigmoid_weight(lat, if_solid_f)
    f = f.to(pred.device)

    grad_x, grad_y = compute_gradients_sobel(pred, lon, lat, R_E=6.371e6)

    u_geo = -(g / f) * grad_y
    v_geo = (g / f) * grad_x
    return u_geo, v_geo, f_weight


def compute_wind_stress(u10, v10, rho_air=1.225, Cd=1.2e-3):
    """
    从10m风速计算风应力

    参数:
        u10, v10: 风速分量 (m/s), tensor [B, T, 1, H, W]
        rho_air: 空气密度 (kg/m^3), 默认1.225
        Cd: 拖曳系数, 默认1.2e-3

    返回:
        tau_x, tau_y: 风应力分量 (N/m^2), tensor [B, T, 1, H, W]
    """
    tau_x = rho_air * Cd * u10 * torch.sqrt(u10**2 + v10**2)
    tau_y = rho_air * Cd * v10 * torch.sqrt(u10**2 + v10**2)
    return tau_x, tau_y


def compute_ekman_depth(f, rho_water=1025, K_v=0.1):
    """
    计算Ekman深度

    参数:
        f: 科氏参数 [1, 1, 1, H, W] 或标量
        rho_water: 海水密度 (kg/m^3), 默认1025
        K_v: 垂直涡动粘性系数 (m^2/s), 默认0.1

    返回:
        D_E: Ekman深度 (m), 与f同形状
    """
    D_E = torch.sqrt(2 * K_v / (rho_water * torch.abs(f + 1e-10)))
    return D_E


def compute_ekman_transport(tau_x, tau_y, f, rho_water=1025):
    """
    计算Ekman输送

    参数:
        tau_x, tau_y: 风应力分量 (N/m^2)
        f: 科氏参数
        rho_water: 海水密度

    返回:
        U_ek, V_ek: Ekman输送分量 (m^2/s)
    """
    U_ek = tau_y / (rho_water * f + 1e-10)
    V_ek = -tau_x / (rho_water * f + 1e-10)
    return U_ek, V_ek


def compute_ekman_current_from_wind(u10, v10, lat, omega=7.2921e-5):
    """
    从风场估算Ekman流（简化公式）

    参数:
        u10, v10: 风速分量 (m/s), tensor [B, T, 1, H, W]
        lat: 纬度网格

    返回:
        u_ek, v_ek: Ekman流估算 (m/s)
    """
    rho_air = 1.225
    Cd = 1.2e-3
    rho_water = 1025
    K_v = 0.1

    tau_x, tau_y = compute_wind_stress(u10, v10, rho_air, Cd)

    lat_t = torch.from_numpy(lat).float().to(u10.device)
    f = 2 * omega * torch.sin(torch.deg2rad(lat_t))
    f = f[None, None, None, :, :]

    D_E = compute_ekman_depth(f, rho_water, K_v)

    u_ek = tau_y / (rho_water * f * D_E + 1e-10)
    v_ek = -tau_x / (rho_water * f * D_E + 1e-10)

    return u_ek, v_ek


def reverse_schedule_sampling(
    itr, total_length, input_length, img_shape, args, reverse=True, mode="train"
):
    """
    PyTorch版本 - 支持训练和测试的reverse schedule sampling
    """
    C, H, W = img_shape

    if mode == "test":
        real_input_flag = torch.ones((total_length - 2, C, H, W), device=args.device)
        real_input_flag[input_length - 1 :] = 0
        return real_input_flag

    elif mode == "train":
        r_eta_st, eta_st = 0.5, 0.5
        if reverse:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")

            elif itr < args.r_sampling_step_2:
                r_eta = r_eta_st + (1 - r_eta_st) * (
                    1.0
                    - math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
                )
                eta = eta_st * (
                    1
                    - (
                        (itr - args.r_sampling_step_1)
                        / (args.r_sampling_step_2 - args.r_sampling_step_1)
                    )
                )
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
        else:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")
            elif itr < args.r_sampling_step_2:
                r_eta = 1.0
                eta = eta_st - eta_st * (
                    1 / (args.r_sampling_step_2 - args.r_sampling_step_1)
                ) * (itr - args.r_sampling_step_1)
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
                print(f"r_eta: {r_eta}, eta: {eta}")

        r_mask = torch.bernoulli(
            torch.full((input_length - 1, C, H, W), r_eta, device=args.device)
        )

        pred_mask = torch.bernoulli(
            torch.full(
                (total_length - input_length - 1, C, H, W), eta, device=args.device
            )
        )

        real_input_flag = torch.cat([r_mask, pred_mask], dim=0)

        return real_input_flag

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'test'.")
