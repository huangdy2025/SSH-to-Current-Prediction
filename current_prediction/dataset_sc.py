"""
流场预测数据集模块
加载 SSH(zos)、风场(u10,v10)、流场(uo,vo)数据
构建非地转流模型的输入和目标

利用 configs_sc.py 的 ModelConfig 统一管理通道配置,
通过 _build_field() 复用 SSH/ageo 的 field 构建逻辑
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from current_prediction.configs_sc import CHANNEL_MAP, _parse_channels


class CurrentDataset(Dataset):
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
        # 非地转模型输入 (ageo) 和 SSH 模型输入, 复用同一构建逻辑
        self.field = self._build_field(args.ageo_config.input_spec, T, H, W)
        self.ssh_field = self._build_field(args.ssh_config.input_spec, T, H, W)

        # ============= 构建 target (流场) =============
        if norm:
            self.target_uo = (self.uo - args.u_c_mu) / args.u_c_std
            self.target_vo = (self.vo - args.v_c_mu) / args.v_c_std
        else:
            self.target_uo = self.uo
            self.target_vo = self.vo
        self.target = np.stack([self.target_uo, self.target_vo], axis=1)  # (T, 2, H, W)

        # 异常值与 NaN 处理
        for arr in [self.field, self.ssh_field, self.target]:
            arr[abs(arr) > 100] = np.nan
            np.nan_to_num(arr, copy=False)

        self.field = torch.from_numpy(self.field)
        self.ssh_field = torch.from_numpy(self.ssh_field)
        self.target = torch.from_numpy(self.target)

        self.input_length = args.input_length
        self.output_length = args.output_length

        print(f"[CurrentDataset/{mode}] field: {self.field.shape}, "
              f"ssh_field: {self.ssh_field.shape}, target: {self.target.shape}")

    def _build_field(self, input_spec: str, T: int, H: int, W: int) -> np.ndarray:
        """
        根据 input_spec 统一构建输入 field (复用于 SSH 和 ageo 模型)
        通道顺序: ssh, wind(u,v), mask, lonlat(lon,lat)
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

            elif part == 'lonlat':
                # lon/lat 可能是 1D (H,) 或 2D (H,W)
                if self.lon.ndim == 1:
                    lon_2d = np.broadcast_to(self.lon[None, :], (H, W))
                    lat_2d = np.broadcast_to(self.lat[:, None], (H, W))
                else:
                    lon_2d = self.lon
                    lat_2d = self.lat
                lon_grid = np.tile(lon_2d[None, None, :, :], (T, 1, 1, 1)).astype(np.float32)
                lat_grid = np.tile(lat_2d[None, None, :, :], (T, 1, 1, 1)).astype(np.float32)
                lon_norm = 2 * (lon_grid - lon_grid.min()) / (lon_grid.max() - lon_grid.min()) - 1
                lat_norm = 2 * (lat_grid - lat_grid.min()) / (lat_grid.max() - lat_grid.min()) - 1
                field_list.append(lon_norm)
                field_list.append(lat_norm)

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
        ssh_input = self.ssh_field[idx:idx + self.input_length]
        datay = self.target[idx + self.input_length:idx + self.input_length + self.output_length]
        return datax, ssh_input, datay
