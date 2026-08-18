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
                 log_dir=None, optimizer=None, scheduler=None, mode='train'):
        self.loss_name = loss_func_train.__class__.__name__
        super().__init__(model, config, loss_func_test, log_dir, optimizer, scheduler, mode=mode)
        self.loss_func_train = loss_func_train

        print(f"E2E model params: {sum(p.numel() for p in self.model.parameters()):,}")

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

        return loss, B

    def _create_mask(self, epoch):
        return None


if __name__ == '__main__':
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from e2e_current_prediction.configs import parse_args, get_my_config
    from e2e_current_prediction.dataset import E2ECurrentDataset
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

    # ============= loss =============
    mask_ocean = ~mask_land
    loss_func = MSELossIgnoreNaNCurrent(args, mask_ocean)
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
    )

    trainer.train_model(train_dataset, eval_dataset, test_dataset)
