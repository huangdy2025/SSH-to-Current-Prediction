"""
流场预测训练器 (GPU版)
架构: 总流 = 地转流(解析, from SSH) + 非地转流(网络学习)
SSH 模型冻结，只训练非地转流模型
"""
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast

from ssh_prediction.tools.base_method import BaseMethod
from current_prediction.mytools_sc import (
    MSELossIgnoreNaNCurrent,
    compute_geostrophic_current_gpu,
    set_all_seeds,
)


class CurrentModel(BaseMethod):
    """
    流场预测模型
    - SSH 模型 (冻结): 输入 SSH(+mask/wind) → 预测未来 SSH → 地转流
    - 非地转模型 (训练): 输入 SSH+wind+mask → 预测非地转流 (u_ageo, v_ageo)
    - 最终: 总流 = 地转流 + 非地转流
    """
    def __init__(self, model, ssh_model, config,
                 loss_func_test=nn.MSELoss(), loss_func_train=nn.MSELoss(),
                 log_dir=None, optimizer=None, scheduler=None, mode='train',
                 lon=None, lat=None, stds=None,
                 mask_land: torch.Tensor = None,
                 loss_func_pinn=None,
                 ssh_mean=0.0, ssh_std=1.0,
                 u_c_mu=0.0, u_c_std=1.0,
                 v_c_mu=0.0, v_c_std=1.0):
        self.loss_name = loss_func_train.__class__.__name__
        super().__init__(model, config, loss_func_test, log_dir, optimizer, scheduler, mode=mode)
        self.loss_func_train = loss_func_train
        self.ssh_model = ssh_model.to(self.device)
        self.ssh_model.eval()
        # 冻结 SSH 模型参数
        for param in self.ssh_model.parameters():
            param.requires_grad = False
        print(f"SSH model frozen, params: {sum(p.numel() for p in self.ssh_model.parameters()):,}")

        # 地转流计算参数
        self.lon, self.lat = lon, lat
        self.stds = stds
        self.mask_land = mask_land[None, None, None]
        # 海洋掩膜 (掩膜感知梯度用: 海岸线处的海洋格点只用海洋邻居)
        self.ocean_mask = ~mask_land

        # 归一化参数 (用于反归一化后计算地转流)
        self.ssh_mean = ssh_mean
        self.ssh_std = ssh_std
        self.u_c_mu = u_c_mu
        self.u_c_std = u_c_std
        self.v_c_mu = v_c_mu
        self.v_c_std = v_c_std

        if config.is_pinn and config.pinn_lambda > 0:
            self.loss_func_pinn = loss_func_pinn

        # 记录 SSH 模型信息到日志
        self.logger.info(f"  SSH model input: {getattr(config, 'ssh_input', 'N/A')}")
        self.logger.info(f"  SSH model path: {getattr(config, 'ssh_model_path', 'N/A')}")

    def _compute_loss(self, train_data, step, mask=None, test=False):
        """
        train_data: (datax, ssh_input, datay)
            datax: (B, T_in, C, H, W) 非地转模型输入
            ssh_input: (B, T_in, C_ssh, H, W) SSH模型输入
            datay: (B, T_out, 2, H, W) 目标流场 (uo, vo)
        """
        datax, ssh_input, datay = train_data
        B = datax.shape[0]

        datax = datax.to(self.device, non_blocking=True)
        ssh_input = ssh_input.to(self.device, non_blocking=True)
        datay = datay.to(self.device, non_blocking=True)

        with autocast('cuda'):

            # 1. SSH 模型预测未来 SSH (冻结, no_grad)
            with torch.no_grad():
                pred_ssh = self.ssh_model(ssh_input)  # (B, T_out, 1, H, W)

            # 2. 从预测的 SSH 计算地转流
            pred_ssh_norm = pred_ssh  # 归一化的 SSH
            # 反归一化 SSH 到物理量 (米) 用于地转流计算
            pred_ssh_phys = pred_ssh_norm * self.ssh_std + self.ssh_mean

            # 计算地转流: 空间变化的 f + sigmoid 权重抑制赤道附近地转贡献
            # if_solid_f=False: f 随纬度变化, 赤道附近 f→0 会被 sigmoid 权重抑制
            u_geo, v_geo, f_weight = compute_geostrophic_current_gpu(
                pred_ssh_phys, self.lon, self.lat,
                if_solid_f=getattr(self.config, 'if_solid_f', True),
                ocean_mask=self.ocean_mask
            )
            # u_geo, v_geo: (B, T_out, C_ssh, H, W), 取第 0 通道
            u_geo = u_geo[:, :, 0:1]  # (B, T_out, 1, H, W)
            v_geo = v_geo[:, :, 0:1]

            # 3. 非地转模型预测非地转流
            pred_ageo = self.model(datax)  # (B, T_out, 2, H, W)

            # 4. 总流 = 地转流 + 非地转流
            # 地转流换算到 target 归一化坐标系: 只除 std 不减 mu。
            # sigma 是线性缩放、对加法可分配 (各分量各自除再相加); mu 是平移、
            # 不可分配 —— 总流的 -mu 已由网络的归一化目标 (datay) 扣除一次,
            # 地转项再减 mu 会重复扣均值, 迫使网络学一个 +mu/std 的补偿偏置。
            u_geo_norm = u_geo / self.u_c_std
            v_geo_norm = v_geo / self.v_c_std

            # 用 f_weight 加权地转流: 赤道附近 (低纬) 权重→0, 网络负责全部流;
            # 中高纬权重→1, 地转流占主导
            f_weight = f_weight.to(self.device)
            pred_total = pred_ageo.clone()
            pred_total[:, :, 0:1] = pred_ageo[:, :, 0:1] + f_weight * u_geo_norm
            pred_total[:, :, 1:2] = pred_ageo[:, :, 1:2] + f_weight * v_geo_norm

            # 5. 计算 loss
            loss = self.loss_func_train(pred_total, datay) if not test else self.loss_func_test(pred_total, datay)

            if not test and self.config.is_pinn and self.config.pinn_lambda > 0:
                pinn_loss = self._compute_pinn_loss(pred_ssh_phys, datay)
                loss = loss + self.config.pinn_lambda * pinn_loss

        return loss, B

    def _compute_pinn_loss(self, pred_ssh, targets):
        """
        地转流 PINN loss: 预测 SSH 的地转流 vs 从目标流场反推的地转流
        """
        mask = self.mask_land.expand_as(targets[:, :, :1])
        # 用预测 SSH 计算地转流
        u_geo_pred, v_geo_pred, w = compute_geostrophic_current_gpu(
            pred_ssh, self.lon, self.lat,
            if_solid_f=getattr(self.config, 'if_solid_f', True),
            ocean_mask=self.ocean_mask
        )

        # 标准化
        if isinstance(self.stds, (int, float)) or (hasattr(self.stds, 'ndim') and self.stds.ndim == 0):
            ssh_std = float(self.stds) if isinstance(self.stds, (int, float)) else self.stds.item()
            u_std, v_std = self.u_c_std, self.v_c_std
        else:
            ssh_std, u_std, v_std = self.stds

        u_norm_pred = u_geo_pred * ssh_std / u_std
        v_norm_pred = v_geo_pred * ssh_std / v_std

        w = w.to(self.device)

        loss_u = self.loss_func_pinn(u_norm_pred * torch.sqrt(w), torch.zeros_like(u_norm_pred))
        loss_v = self.loss_func_pinn(v_norm_pred * torch.sqrt(w), torch.zeros_like(v_norm_pred))
        return loss_u + loss_v

    def _create_mask(self, epoch):
        return None


if __name__ == '__main__':
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from current_prediction.configs_sc import parse_args, get_my_config
    from current_prediction.dataset_sc import CurrentDataset
    from current_prediction.mytools_sc import MSELossIgnoreNaNCurrent
    from models import SimVP_Model
    import time
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args_ = parse_args()
    args = get_my_config(args_)

    set_all_seeds(args.SEED)
    mask_land = torch.from_numpy(args.mask_land)  # True=陆地

    # ============= 构建冻结的 SSH 模型 =============
    # 使用 configs_sc 统一构建的 ssh_model_config
    ssh_model = SimVP_Model(**args.ssh_model_config)
    if args.ssh_model_path and os.path.exists(args.ssh_model_path):
        ssh_state = torch.load(args.ssh_model_path, map_location=args.device)
        ssh_model.load_state_dict(ssh_state)
        print(f"SSH model loaded from {args.ssh_model_path}")
    else:
        print(f"Warning: SSH model not found at '{args.ssh_model_path}', using untrained SSH model")
    print(f"SSH model params: {sum(p.numel() for p in ssh_model.parameters()):,}")

    # ============= 构建非地转流模型 =============
    model = SimVP_Model(**args.model_config)
    print(f"Current model params: {sum(p.numel() for p in model.parameters()):,}")

    # ============= 数据集 =============
    train_dataset = CurrentDataset(args, mode='train', norm=args.norm)
    eval_dataset = CurrentDataset(args, mode='eval', norm=args.norm)
    test_dataset = CurrentDataset(args, mode='test', norm=args.norm)

    lon, lat = eval_dataset.lon, eval_dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # ============= loss =============
    mask_ocean = ~mask_land
    loss_func = MSELossIgnoreNaNCurrent(args, mask_ocean)
    loss_func_pinn = MSELossIgnoreNaNCurrent(args)
    loss_func_test = loss_func
    loss_func_train = loss_func if args.loss_ignore_nan else nn.MSELoss()

    # ============= optimizer =============
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # ============= log dir =============
    log_dir = rf"{args.model_savepath}/{model.__class__.__name__}_seed{args.SEED}/{args.file_name}_{time.strftime('%Y%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    # ============= trainer =============
    trainer = CurrentModel(
        model, ssh_model, args,
        loss_func_test, loss_func_train,
        log_dir, optimizer, scheduler,
        mode='train',
        lon=lon, lat=lat, stds=args.ssh_std,
        mask_land=mask_land, loss_func_pinn=loss_func_pinn,
        ssh_mean=args.ssh_mean, ssh_std=args.ssh_std,
        u_c_mu=args.u_c_mu, u_c_std=args.u_c_std,
        v_c_mu=args.v_c_mu, v_c_std=args.v_c_std,
    )

    trainer.train_model(train_dataset, eval_dataset, test_dataset)
