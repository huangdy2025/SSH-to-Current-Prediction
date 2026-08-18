"""
端到端流场预测数据集模块
加载 SSH(zos)、风场(u10,v10)、流场(uo,vo)数据
构建端到端模型的输入和目标

利用 configs.py 的 E2EModelConfig 统一管理通道配置,
通过 _build_field() 构建输入 field
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from e2e_current_prediction.configs import CHANNEL_MAP, _parse_channels


class E2ECurrentDataset(Dataset):
    def __init__(self, args, mode='train', norm=False):
        self.args = args
        self.mode = mode
        self.norm = norm

        # 确定时间范围
        if mode == 'train':
            time_span = slice(args.start_time_train, args.end_time_train)
        elif mode == 'eval':
            time_span = slice(args.start_time_val, args.end_time_val)
        elif mode == 'test':
            time_span = slice(args.start_time_test, args.end_time_test)
        elif mode == 'all':
            time_span = slice(args.start_time_train, args.end_time_test)
        else:
            raise ValueError('mode must be train, eval, test or all')

        # ============= 加载原始数据 =============
        with xr.open_dataset(args.ssh_path) as ssh_data_ds:
            ssh_data = ssh_data_ds.sel(time=time_span)
            self.lat = ssh_data.latitude.values
            self.lon = ssh_data.longitude.values
            self.dates = ssh_data.time.values
            self.ssh = ssh_data.zos.values.astype(np.float32)

        with xr.open_dataset(args.wind_path) as wind_ds:
            wind_data = wind_ds.sel(time=time_span)
            self.u10 = wind_data.u10.values.astype(np.float32)
            self.v10 = wind_data.v10.values.astype(np.float32)

        with xr.open_dataset(args.uv_path) as uv_ds:
            uv_data = uv_ds.sel(time=time_span)
            self.uo = uv_data.uo.values.astype(np.float32)
            self.vo = uv_data.vo.values.astype(np.float32)

        T, H, W = self.ssh.shape

        # ============= 构建模型输入 field =============
        self.field = self._build_field(args.e2e_input, T, H, W)

        # ============= 构建 target (流场) =============
        if norm:
            self.target_uo = (self.uo - args.u_c_mu) / args.u_c_std
            self.target_vo = (self.vo - args.v_c_mu) / args.v_c_std
        else:
            self.target_uo = self.uo
            self.target_vo = self.vo
        self.target = np.stack([self.target_uo, self.target_vo], axis=1)  # (T, 2, H, W)

        # 异常值与 NaN 处理
        for arr in [self.field, self.target]:
            arr[abs(arr) > 100] = np.nan
            np.nan_to_num(arr, copy=False)

        self.field = torch.from_numpy(self.field)
        self.target = torch.from_numpy(self.target)

        self.input_length = args.input_length
        self.output_length = args.output_length

        print(f"[E2ECurrentDataset/{mode}] field: {self.field.shape}, "
              f"target: {self.target.shape}")

    def _build_field(self, input_spec: str, T: int, H: int, W: int) -> np.ndarray:
        """
        根据 input_spec 统一构建输入 field
        通道顺序: ssh, wind(u,v), mask
        """
        parts = input_spec.split('_')
        field_list = []

        for part in parts:
            if part == 'ssh':
                ssh_field = self.ssh.copy()
                if self.norm:
                    ssh_field = (ssh_field - self.args.ssh_mean) / self.args.ssh_std
                field_list.append(ssh_field[:, None, :, :])  # (T, 1, H, W)

            elif part == 'wind':
                u10 = (self.u10 - self.args.u10_mu) / self.args.u10_std if self.norm else self.u10
                v10 = (self.v10 - self.args.v10_mu) / self.args.v10_std if self.norm else self.v10
                field_list.append(u10[:, None, :, :])
                field_list.append(v10[:, None, :, :])

            elif part == 'mask':
                mask = self.args.mask_land  # (H, W) True=陆地
                mask_tiled = np.tile(mask[None, None, :, :], (T, 1, 1, 1))
                field_list.append(mask_tiled.astype(np.float32))

            else:
                raise ValueError(f"Unknown input component '{part}' in '{input_spec}'")

        return np.concatenate(field_list, axis=1)  # (T, C, H, W)

    def __len__(self):
        return len(self.dates) - self.input_length - self.output_length + 1

    @property
    def start_dates(self):
        return self.dates[:len(self)]

    def __getitem__(self, idx):
        datax = self.field[idx:idx + self.input_length]
        datay = self.target[idx + self.input_length:idx + self.input_length + self.output_length]
        return datax, datay
