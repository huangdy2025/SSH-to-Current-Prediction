import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch_npu
import numpy as np
from torch_npu.contrib import transfer_to_npu
import argparse
from pathlib import Path
from models import SimVP_Model
from mytools import (
    compute_geostrophic_current,
    compute_f_and_sigmoid_weight,
    unpatchify_with_batch,
)
import xarray as xr


class SurfaceCurrentInference:
    """
    表层流组合推理: SSH模型(地转流) + Ekman模型(Ekman流) = 总表层流

    流程:
    1. SSH模型 → SSH_pred
    2. geostrophic(SSH_pred) → (ug, vg)
    3. Ekman模型(SSH_pred, u10, v10) → (ue, ve)
    4. 最终: u = ug + ue, v = vg + ve
    """

    def __init__(self, ssh_model_path, ekman_model_path, args, device='npu:0'):
        self.args = args
        self.device = device

        # 加载SSH模型
        self.ssh_model = self._load_model(ssh_model_path, args)

        # 加载Ekman模型
        self.ekman_model = self._load_model(ekman_model_path, args)

        # 加载经纬度
        ssh_files = [p for p in args.datapath if 'ssh' in str(p)]
        ssh_data = xr.open_mfdataset(ssh_files, combine='nested', concat_dim='time')
        self.lat = ssh_data.latitude.values
        self.lon = ssh_data.longitude.values
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)

    def _load_model(self, model_path, args):
        model = SimVP_Model(
            in_shape=[args.input_length, args.input_channels, args.height, args.width],
            C_out=args.output_channels,
            hid_S=16, hid_T=128, N_S=4, N_T=6,
            model_type='tau', drop=0.0, drop_path=0.0,
            spatio_kernel_enc=3, spatio_kernel_dec=3,
            mlp_ratio=2.0, gated=False
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def compute_geostrophic_from_ssh(self, ssh_pred):
        """
        从SSH预测计算地转流

        参数:
            ssh_pred: SSH预测 (B, T, 1, H, W)，已归一化

        返回:
            u_geo, v_geo: 地转流 (B, T, 1, H, W)，物理单位 m/s
        """
        # 反归一化
        ssh_mean = np.load(self.args.path_means)
        ssh_std = np.load(self.args.path_stds)
        if ssh_std.ndim == 0:
            ssh_phys = ssh_pred * ssh_std + ssh_mean
        else:
            ssh_phys = ssh_pred * ssh_std[0] + ssh_mean[0]

        u_geo, v_geo, _ = compute_geostrophic_current(
            ssh_phys.cpu(), self.lon_grid, self.lat_grid, if_solid_f=False
        )
        return u_geo.to(self.device), v_geo.to(self.device)

    def predict(self, ssh_input, wind_input):
        """
        组合推理

        参数:
            ssh_input: SSH输入 (B, T, 1, H, W)
            wind_input: 风场输入 (B, T, 2, H, W) [u10, v10]

        返回:
            u_total, v_total: 总表层流 (B, T, 1, H, W)
            u_geo, v_geo: 地转流分量
            u_ek, v_ek: Ekman流分量
        """
        # 1. SSH模型预测SSH
        with torch.no_grad():
            ssh_pred = self.ssh_model(ssh_input)

        # 2. 计算地转流
        u_geo, v_geo = self.compute_geostrophic_from_ssh(ssh_pred)

        # 3. Ekman模型预测Ekman残差
        # 输入: SSH_pred + wind
        ekman_input = torch.cat([ssh_pred, wind_input], dim=2)
        with torch.no_grad():
            ekman_pred = self.ekman_model(ekman_input)

        u_ek = ekman_pred[:, :, 0:1, :, :]
        v_ek = ekman_pred[:, :, 1:2, :, :]

        # 4. 组合: 总流 = 地转流 + Ekman流
        u_total = u_geo + u_ek
        v_total = v_geo + v_ek

        return u_total, v_total, u_geo, v_geo, u_ek, v_ek

    def predict_from_file(self, ssh_nc_path, wind_nc_path, output_path=None):
        """
        从NetCDF文件读取数据并推理

        参数:
            ssh_nc_path: SSH NetCDF文件路径
            wind_nc_path: 风场NetCDF文件路径
            output_path: 输出NetCDF路径 (可选)
        """
        ssh_data = xr.open_dataset(ssh_nc_path)
        wind_data = xr.open_dataset(wind_nc_path)

        ssh = ssh_data.zos.values.astype(np.float32)
        u10 = wind_data.u10.values.astype(np.float32)
        v10 = wind_data.v10.values.astype(np.float32)

        # 归一化
        ssh_mean = np.load(self.args.path_means)
        ssh_std = np.load(self.args.path_stds)
        if ssh_std.ndim == 0:
            ssh_norm = (ssh - ssh_mean) / ssh_std
        else:
            ssh_norm = (ssh - ssh_mean[0]) / ssh_std[0]

        # 转tensor
        ssh_tensor = torch.from_numpy(ssh_norm).float().unsqueeze(0).unsqueeze(2).to(self.device)
        u10_tensor = torch.from_numpy(u10).float().unsqueeze(0).unsqueeze(2).to(self.device)
        v10_tensor = torch.from_numpy(v10).float().unsqueeze(0).unsqueeze(2).to(self.device)
        wind_tensor = torch.cat([u10_tensor, v10_tensor], dim=2)

        # 推理
        u_total, v_total, u_geo, v_geo, u_ek, v_ek = self.predict(ssh_tensor, wind_tensor)

        # 保存结果
        if output_path is not None:
            result = {
                'u_total': u_total.cpu().numpy(),
                'v_total': v_total.cpu().numpy(),
                'u_geo': u_geo.cpu().numpy(),
                'v_geo': v_geo.cpu().numpy(),
                'u_ek': u_ek.cpu().numpy(),
                'v_ek': v_ek.cpu().numpy(),
            }
            np.savez(output_path, **result)
            print(f"Results saved to {output_path}")

        return u_total, v_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh_model', type=str, required=True, help='Path to SSH model weights')
    parser.add_argument('--ekman_model', type=str, required=True, help='Path to Ekman model weights')
    parser.add_argument('--ssh_nc', type=str, required=True, help='Path to SSH NetCDF file')
    parser.add_argument('--wind_nc', type=str, required=True, help='Path to wind NetCDF file')
    parser.add_argument('--output', type=str, default=None, help='Output path (.npz)')
    parser.add_argument('--device', type=str, default='npu:0')
    parser.add_argument('--base', type=str, default='/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=240)
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--output_length', type=int, default=10)
    parser.add_argument('--need_ssh', action='store_true', default=True)
    parser.add_argument('--need_wind', action='store_true', default=True)
    parser.add_argument('--need_ekman', action='store_true', default=True)
    args = parser.parse_args()

    # 设置args
    args.input_channels = 1
    args.output_channels = 1
    args.model_config = {'patch_size': 8}

    inferencer = SurfaceCurrentInference(
        args.ssh_model, args.ekman_model, args, device=args.device
    )
    inferencer.predict_from_file(args.ssh_nc, args.wind_nc, args.output)
