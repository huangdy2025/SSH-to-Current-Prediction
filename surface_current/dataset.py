import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CurrentDataset(Dataset):
    """
    表层流数据集：输入SSH+wind，标签=Ekman残差(ue, ve)

    归一化方式 (来自 min_max_stat.npz):
      - SSH: z-score (mean.npy/std.npy)
      - Wind (u10, v10): min-max
      - Current (uo, vo): min-max

    输入通道: [SSH(1), u10(1), v10(1)] = 3 channels (归一化后)
    标签通道: [ue(1), ve(1)] = 2 channels  (Ekman残差，归一化后)

    训练时预计算:
      1. 从原始SSH通过地转平衡计算(ug, vg)
      2. 原始(uo, vo) - 地转流(ug, vg) = Ekman残差(ue_true, ve_true)
      3. 对Ekman残差进行min-max归一化
    """

    def __init__(self, args, mode='train', norm=False):
        self.args = args
        self.norm = norm

        # 1. 加载SSH数据
        ssh_files = [p for p in self.args.datapath if 'ssh' in str(p)]
        self.ssh_data = xr.open_mfdataset(ssh_files, combine='nested', concat_dim='time')

        if mode == 'train':
            time_span = slice(args.start_time_train, args.end_time_train)
        elif mode == 'eval':
            time_span = slice(args.start_time_val, args.end_time_val)
        elif mode == 'test':
            time_span = slice(args.start_time_test, args.end_time_test)
        else:
            raise ValueError('mode must be train, eval or test')

        self.ssh_data = self.ssh_data.sel(time=time_span)
        self.lat = self.ssh_data.latitude.values
        self.lon = self.ssh_data.longitude.values

        # 2. 加载SSH (原始值)
        self.ssh = np.expand_dims(self.ssh_data.zos.values.astype(np.float32), axis=1)
        print(f"SSH shape: {self.ssh.shape}")

        # 3. 加载wind (u10, v10) (原始值)
        self.wind_data = xr.open_dataset(args.wind_path).sel(time=time_span)
        self.u10 = np.expand_dims(self.wind_data.u10.values.astype(np.float32), axis=1)
        self.v10 = np.expand_dims(self.wind_data.v10.values.astype(np.float32), axis=1)
        print(f"Wind shape: u10={self.u10.shape}, v10={self.v10.shape}")

        # 4. 加载总表层流 (uo, vo) (原始值)
        self.current_data = xr.open_dataset(args.uv_current_path).sel(time=time_span)
        self.uo = np.expand_dims(self.current_data.uo.values.astype(np.float32), axis=1)
        self.vo = np.expand_dims(self.current_data.vo.values.astype(np.float32), axis=1)
        print(f"Current shape: uo={self.uo.shape}, vo={self.vo.shape}")

        # 5. 加载归一化统计量
        self._load_stats()

        # 6. 计算地转流(ug, vg)并获取Ekman残差标签 (在原始空间计算)
        self.ue_label, self.ve_label = self._compute_ekman_labels()

        # 7. 归一化所有变量
        if norm:
            self._normalize()

        # 8. 组装输入字段: [SSH(归一化), u10(归一化), v10(归一化)]
        self.input_field = np.concatenate([self.ssh, self.u10, self.v10], axis=1)

        # 9. 组装标签字段: [ue(归一化), ve(归一化)]
        self.label_field = np.concatenate([self.uo, self.vo], axis=1)

        # 10. 处理异常值
        self.input_field[abs(self.input_field) > 100] = np.nan
        self.label_field[abs(self.label_field) > 100] = np.nan
        self.input_field = np.nan_to_num(self.input_field)
        self.label_field = np.nan_to_num(self.label_field)

        # 11. 加载land mask
        if args.need_mask:
            mask = np.load(args.path_land_mask)
            self.mask_land = torch.from_numpy(mask).float()
        else:
            self.mask_land = torch.zeros(self.ssh.shape[2], self.ssh.shape[3])

        self.input_length = args.input_length
        self.output_length = args.output_length

        # 12. Patchify (if needed for PredRNN)
        if args.patched:
            p = self.args.model_config['patch_size']
            T_in, C_in = self.input_field.shape[:2]
            T_out, C_out = self.label_field.shape[:2]
            H, W = self.input_field.shape[2:]

            self.input_field = self.input_field.reshape(T_in, C_in, H // p, p, W // p, p)
            self.input_field = self.input_field.transpose(0, 1, 3, 5, 2, 4)
            self.input_field = self.input_field.reshape(T_in, C_in * p * p, H // p, W // p)

            self.label_field = self.label_field.reshape(T_out, C_out, H // p, p, W // p, p)
            self.label_field = self.label_field.transpose(0, 1, 3, 5, 2, 4)
            self.label_field = self.label_field.reshape(T_out, C_out * p * p, H // p, W // p)

        self.input_field = torch.from_numpy(self.input_field)
        self.label_field = torch.from_numpy(self.label_field)

        print(f"Input field: {self.input_field.shape}, Label field: {self.label_field.shape}")

    def _load_stats(self):
        """加载归一化统计量"""
        # SSH: z-score
        self.ssh_mean = np.load(self.args.path_means)
        self.ssh_std = np.load(self.args.path_stds)

        # Wind + Current: min-max (来自 min_max_stat.npz)
        stat_path = Path(self.args.min_max_stat_path) if hasattr(self.args, 'min_max_stat_path') else self.args.base / "min_max_stat.npz"
        if stat_path.exists():
            stat = np.load(stat_path)
            self.u_w_min = float(stat['u_w_min'])
            self.u_w_max = float(stat['u_w_max'])
            self.v_w_min = float(stat['v_w_min'])
            self.v_w_max = float(stat['v_w_max'])
            self.u_c_min = float(stat['u_c_min'])
            self.u_c_max = float(stat['u_c_max'])
            self.v_c_min = float(stat['v_c_min'])
            self.v_c_max = float(stat['v_c_max'])
        else:
            # 回退: 从数据计算
            print(f"Warning: {stat_path} not found, computing stats from data")
            self.u_w_min = float(np.nanmin(self.u10))
            self.u_w_max = float(np.nanmax(self.u10))
            self.v_w_min = float(np.nanmin(self.v10))
            self.v_w_max = float(np.nanmax(self.v10))
            self.u_c_min = float(np.nanmin(self.uo))
            self.u_c_max = float(np.nanmax(self.uo))
            self.v_c_min = float(np.nanmin(self.vo))
            self.v_c_max = float(np.nanmax(self.vo))

        print(f"Stats loaded: ssh_mean={self.ssh_mean:.4f}, ssh_std={self.ssh_std:.4f}")
        print(f"  wind: u10=[{self.u_w_min:.2f}, {self.u_w_max:.2f}], v10=[{self.v_w_min:.2f}, {self.v_w_max:.2f}]")
        print(f"  current: uo=[{self.u_c_min:.4f}, {self.u_c_max:.4f}], vo=[{self.v_c_min:.4f}, {self.v_c_max:.4f}]")

    def _minmax_norm(self, arr, vmin, vmax):
        """min-max归一化到 [0, 1]"""
        return (arr - vmin) / (vmax - vmin + 1e-8)

    def _minmax_denorm(self, arr, vmin, vmax):
        """min-max反归一化"""
        return arr * (vmax - vmin + 1e-8) + vmin

    def _normalize(self):
        """归一化各变量 (原始空间 → 归一化空间)"""
        # SSH: z-score
        self.ssh = (self.ssh - self.ssh_mean) / (self.ssh_std + 1e-8)

        # Wind: min-max
        self.u10 = self._minmax_norm(self.u10, self.u_w_min, self.u_w_max)
        self.v10 = self._minmax_norm(self.v10, self.v_w_min, self.v_w_max)

        # Current: min-max
        self.uo = self._minmax_norm(self.uo, self.u_c_min, self.u_c_max)
        self.vo = self._minmax_norm(self.vo, self.v_c_min, self.v_c_max)

    def _compute_ekman_labels(self):
        """
        从原始SSH计算地转流(ug, vg)，然后计算Ekman残差(ue, ve)
        ue = uo - ug, ve = vo - vg (在原始空间计算)

        返回的ue/ve是原始物理值，后续归一化由 _normalize() 处理
        """
        # 计算地转流需要原始SSH值 (m)
        ssh_phys = self.ssh.copy()  # 此时还未归一化，就是原始值

        T = ssh_phys.shape[0]
        H, W = ssh_phys.shape[2], ssh_phys.shape[3]

        # 创建网格
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        lat_tensor = torch.from_numpy(lat_grid).float()

        # 科氏参数
        Omega = 7.2921e-5
        f = 2 * Omega * torch.sin(torch.deg2rad(lat_tensor))
        f = f[None, None, :, :]  # [1, 1, H, W]

        # Sobel梯度
        kernel_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        kernel_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0

        R_E = 6.371e6
        delta_lat = self.lat[1] - self.lat[0]
        delta_lon = self.lon[1] - self.lon[0]
        dy = delta_lat * (np.pi / 180) * R_E
        lat_rad = np.deg2rad(self.lat)
        dx_per_row = delta_lon * (np.pi / 180) * R_E * np.cos(lat_rad)
        dx_tensor = torch.from_numpy(dx_per_row).float().view(1, 1, H, 1)

        g = 9.81
        ug_list = []
        vg_list = []

        for t in range(T):
            ssh_t = torch.from_numpy(ssh_phys[t, 0]).float().unsqueeze(0).unsqueeze(0)

            x_padded = torch.nn.functional.pad(ssh_t, pad=(1, 1, 1, 1), mode='replicate')
            grad_x = torch.nn.functional.conv2d(x_padded, kernel_sobel_x.view(1, 1, 3, 3))
            grad_y = torch.nn.functional.conv2d(x_padded, kernel_sobel_y.view(1, 1, 3, 3))

            grad_x_phys = grad_x / dx_tensor
            grad_y_phys = grad_y / dy

            u_geo = -(g / f) * grad_y_phys
            v_geo = (g / f) * grad_x_phys

            ug_list.append(u_geo.squeeze().numpy())
            vg_list.append(v_geo.squeeze().numpy())

        ug = np.stack(ug_list, axis=0)[:, None, :, :]  # (T, 1, H, W), m/s
        vg = np.stack(vg_list, axis=0)[:, None, :, :]

        # Ekman残差 = 总流(原始) - 地转流 (原始空间)
        ue = self.uo - ug
        ve = self.vo - vg

        return ue.astype(np.float32), ve.astype(np.float32)

    def __len__(self):
        return len(self.ssh_data.time) - self.input_length - self.output_length + 1

    def __getitem__(self, idx):
        if self.args.one_seq:
            data = self.field[idx:idx+self.input_length+self.output_length]
            return data
        else:
            datax = self.input_field[idx:idx+self.input_length]
            datay = self.label_field[idx+self.input_length:idx+self.input_length+self.output_length]
            return datax, datay


if __name__ == '__main__':
    from configs import parse_args, get_my_config
    args = parse_args()
    args = get_my_config(args)
    dataset = CurrentDataset(args, mode='test', norm=args.norm)
    print(f"Dataset length: {len(dataset)}")
    datax, datay = dataset[0]
    print(f"Input shape: {datax.shape}")
    print(f"Label shape: {datay.shape}")
