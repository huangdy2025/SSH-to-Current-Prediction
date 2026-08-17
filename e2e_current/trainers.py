"""
端到端流场预测训练器 (GPU版)
架构: 单一模型直接学习 SSH+wind+mask → u,v 总流
无地转/非地转分解, 无冻结 SSH 模型
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


class E2ECurrentModel(BaseMethod):
    """
    端到端流场预测模型
    - 单一模型 (训练): 输入 SSH+wind+mask → 直接预测总流 (u, v)
    - 无地转/非地转分解
    """
    def __init__(self, model, config,
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

        print(f"E2E model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # 地转流计算参数 (仅用于 PINN 软约束)
        self.lon, self.lat = lon, lat
        self.stds = stds
        self.mask_land = mask_land[None, None, None] if mask_land is not None else None

        # 归一化参数 (用于 PINN 软约束中反归一化)
        self.ssh_mean = ssh_mean
        self.ssh_std = ssh_std
        self.u_c_mu = u_c_mu
        self.u_c_std = u_c_std
        self.v_c_mu = v_c_mu
        self.v_c_std = v_c_std

        if config.is_pinn and config.pinn_lambda > 0:
            self.loss_func_pinn = loss_func_pinn

    def _compute_loss(self, train_data, step, mask=None, test=False):
        """
        train_data: (datax, datay)
            datax: (B, T_in, C, H, W) 端到端模型输入
            datay: (B, T_out, 2, H, W) 目标流场 (uo, vo)
        """
        datax, datay = train_data
        B = datax.shape[0]

        datax = datax.to(self.device, non_blocking=True)
        datay = datay.to(self.device, non_blocking=True)

        with autocast('cuda'):

            # 1. 端到端模型直接预测总流
            pred_total = self.model(datax)  # (B, T_out, 2, H, W)

            # 2. 计算 loss
            loss = self.loss_func_train(pred_total, datay) if not test else self.loss_func_test(pred_total, datay)

            if not test and self.config.is_pinn and self.config.pinn_lambda > 0:
                pinn_loss = self._compute_e2e_pinn_loss(datax, pred_total)
                loss = loss + self.config.pinn_lambda * pinn_loss

        return loss, B

    def _compute_e2e_pinn_loss(self, datax, pred_total):
        """
        E2E PINN loss: 软约束预测流场接近地转+风驱平衡
        """
        ssh_input_phys = datax[:, -1, 0:1, :, :] * self.ssh_std + self.ssh_mean
        u_geo_ref, v_geo_ref, f_weight = compute_geostrophic_current_gpu(
            ssh_input_phys.unsqueeze(1), self.lon, self.lat,
            if_solid_f=getattr(self.config, 'if_solid_f', True)
        )

        pred_u_phys = pred_total[:, :, 0:1] * self.u_c_std + self.u_c_mu
        pred_v_phys = pred_total[:, :, 1:2] * self.v_c_std + self.v_c_mu

        f_weight = f_weight.to(self.device)
        res_u = pred_u_phys - f_weight * u_geo_ref[:, :, 0:1]
        res_v = pred_v_phys - f_weight * v_geo_ref[:, :, 0:1]

        if self.mask_land is not None:
            m = (~self.mask_land).to(self.device)
            m = m.expand_as(res_u)
            count = m.sum().clamp(min=1)
            return ((res_u**2 + res_v**2) * m).sum() / count
        return torch.mean(res_u**2 + res_v**2)

    def _create_mask(self, epoch):
        return None


if __name__ == '__main__':
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from e2e_current.configs import parse_args, get_my_config
    from e2e_current.dataset import E2ECurrentDataset
    from current_prediction.mytools_sc import MSELossIgnoreNaNCurrent
    from models import SimVP_Model
    import time
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args_ = parse_args()
    args = get_my_config(args_)

    set_all_seeds(args.SEED)
    mask_land = torch.from_numpy(args.mask_land)  # True=陆地

    # ============= 构建端到端流模型 =============
    model = SimVP_Model(**args.model_config)
    print(f"E2E model params: {sum(p.numel() for p in model.parameters()):,}")

    # ============= 数据集 =============
    train_dataset = E2ECurrentDataset(args, mode='train', norm=args.norm)
    eval_dataset = E2ECurrentDataset(args, mode='eval', norm=args.norm)
    test_dataset = E2ECurrentDataset(args, mode='test', norm=args.norm)

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
    trainer = E2ECurrentModel(
        model, args,
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
