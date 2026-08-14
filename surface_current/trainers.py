import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.base_method import BaseMethod
import torch
import torch_npu
import numpy as np
from torch_npu.contrib import transfer_to_npu
from torch_npu.npu import amp
from torch_npu.npu.amp import autocast
from mytools import (
    reverse_schedule_sampling,
    convert_configs,
    MSELossIgnoreNaN,
    compute_geostrophic_current,
    unpatchify_with_batch,
    compute_wind_stress,
)
import torch.nn as nn


class Model(BaseMethod):
    def __init__(
        self,
        model,
        config,
        loss_func_test=nn.MSELoss(),
        loss_func_train=nn.MSELoss(),
        log_dir=None,
        optimizer=None,
        scheduler=None,
        mode="train",
        lon=None,
        lat=None,
        stds=None,
        mask_land: torch.Tensor = None,
        loss_func_pinn=None,
    ):
        self.loss_name = loss_func_train.__class__.__name__
        super().__init__(
            model, config, loss_func_test, log_dir, optimizer, scheduler, mode=mode
        )
        self.loss_func_train = loss_func_train

        if config.pinn_lambda > 0:
            self.lon, self.lat = lon, lat
            self.stds = stds
            self.loss_func_pinn = loss_func_pinn
            self.mask_land = mask_land[None, None, None]

    def _compute_loss(self, train_data, step, mask=None, test=False):
        inputs, targets = train_data

        B = inputs.shape[0]
        with autocast(True):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            preds = self.model(inputs)
            loss = (
                self.loss_func_train(preds, targets)
                if not test
                else self.loss_func_test(preds, targets)
            )
            if not test and self.config.is_pinn:
                pinn_loss = self._compute_pinn_loss_sigmoid_weight(preds, targets)
                loss = loss + self.config.pinn_lambda * pinn_loss
        return loss, B

    def _create_mask(self, epoch):
        return None

    def _compute_pinn_loss_sigmoid_weight(self, preds, targets):
        mask = self.mask_land.expand_as(targets)
        targets[mask] = torch.nan
        ssh_concate = torch.cat([targets[:, :, :1], preds[:, :, :1]], dim=2)
        ssh_concate_cpu = ssh_concate.cpu()
        u, v, w = compute_geostrophic_current(
            ssh_concate_cpu, self.lon, self.lat, if_solid_f=True
        )
        u = u.to(self.config.device)
        v = v.to(self.config.device)
        w = w.to(self.config.device)
        u_true, u_pred = u[:, :, 0], u[:, :, 1]
        v_true, v_pred = v[:, :, 0], v[:, :, 1]
        if self.stds.ndim == 0:
            ssh_std, u_std, v_std = self.stds.item(), 0.23, 0.23
        else:
            ssh_std, u_std, v_std = self.stds
        u_norm_pred = u_pred * ssh_std / u_std
        v_norm_pred = v_pred * ssh_std / v_std
        u_norm_true = u_true * ssh_std / u_std
        v_norm_true = v_true * ssh_std / v_std
        w = w.to(self.config.device)

        loss_u = self.loss_func_pinn(
            u_norm_pred * torch.sqrt(w), u_norm_true * torch.sqrt(w)
        )
        loss_v = self.loss_func_pinn(
            v_norm_pred * torch.sqrt(w), v_norm_true * torch.sqrt(w)
        )
        return loss_u + loss_v


class ReModel(Model):
    def __init__(
        self,
        model,
        config,
        loss_func_test=nn.MSELoss(),
        loss_func_train=nn.MSELoss(),
        log_dir=None,
        optimizer=None,
        scheduler=None,
        mode="train",
        lon=None,
        lat=None,
        stds=None,
        mask_land: torch.Tensor = None,
        loss_func_pinn=None,
    ):
        super().__init__(
            model,
            config,
            loss_func_test=loss_func_test,
            loss_func_train=loss_func_train,
            log_dir=log_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            mode=mode,
            lon=lon,
            lat=lat,
            stds=stds,
            mask_land=mask_land,
            loss_func_pinn=loss_func_pinn,
        )
        self.model_config = convert_configs(self.model_config)

        self.img_shape = (
            self.config.output_channels * self.model_config.patch_size**2,
            self.config.height // self.model_config.patch_size,
            self.config.width // self.model_config.patch_size,
        )

    def _compute_loss(self, train_data, step, mask=None, test=False):
        B = train_data.shape[0]
        with autocast(True):
            train_data = train_data.to(self.device)
            var_pred, loss = self.model(
                train_data,
                mask,
                loss_func=self.loss_func_test if test else self.loss_func_train,
            )
            if not test and self.config.is_pinn:
                target = unpatchify_with_batch(
                    train_data, self.model_config.patch_size, self.config.input_channels
                )[:, 1:, 0 : self.config.output_channels]
                pred = unpatchify_with_batch(
                    var_pred, self.model_config.patch_size, self.config.output_channels
                )[:, :, 0 : self.config.output_channels]
                pinn_loss = self._compute_pinn_loss_sigmoid_weight(pred, target)
                loss = loss + self.config.pinn_lambda * pinn_loss
        return loss, B

    def _create_mask(self, epoch):
        mask_true_train = reverse_schedule_sampling(
            epoch + 1,
            self.model_config.total_length,
            self.config.input_length,
            self.img_shape,
            self.model_config,
            reverse=self.model_config.reverse_schedule,
        )
        print(f"mask: {mask_true_train.shape}")

        return mask_true_train


class EkmanModel(Model):
    """
    Ekman残差模型 — 联合训练模式

    训练流程:
      1. SSH模型(冻结) → SSH_pred
      2. geostrophic(SSH_pred) → (ug, vg)
      3. Ekman模型(SSH_pred, wind) → (ue_pred, ve_pred)
      4. u_total = ug + ue_pred, v_total = vg + ve_pred
      5. Loss = MSE(u_total, uo_true)

    Dataset输出:
      inputs:  (B, T, 3, H, W) — [SSH, u10, v10]
      targets: (B, T, 4, H, W) — [uo_true, vo_true, ug_true, vg_true]
        其中 uo/vo 是总流真值, ug/vg 是从真值SSH计算的地转流（用于对比）
    """

    def __init__(
        self,
        model,
        ssh_model,
        config,
        loss_func_test=nn.MSELoss(),
        loss_func_train=nn.MSELoss(),
        log_dir=None,
        optimizer=None,
        scheduler=None,
        mode="train",
        lon=None,
        lat=None,
        stds=None,
        mask_land: torch.Tensor = None,
        loss_func_pinn=None,
    ):
        super().__init__(
            model,
            config,
            loss_func_test=loss_func_test,
            loss_func_train=loss_func_train,
            log_dir=log_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            mode=mode,
            lon=lon,
            lat=lat,
            stds=stds,
            mask_land=mask_land,
            loss_func_pinn=loss_func_pinn,
        )
        self.ssh_model = ssh_model
        self.ssh_model.eval()
        for p in self.ssh_model.parameters():
            p.requires_grad = False

    def _compute_loss(self, train_data, step, mask=None, test=False):
        inputs, targets = train_data
        B = inputs.shape[0]

        with autocast(True):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            ssh_input = inputs[:, :, 0:1, :, :]
            wind_input = inputs[:, :, 1:3, :, :]
            uo_true = targets[:, :, 0:1, :, :]
            vo_true = targets[:, :, 1:2, :, :]

            with torch.no_grad():
                ssh_pred = self.ssh_model(ssh_input)

            u_geo, v_geo, _ = compute_geostrophic_current(
                ssh_pred, self.lon, self.lat, if_solid_f=False
            )

            ekman_input = torch.cat([ssh_pred, wind_input], dim=2)
            ekman_pred = self.model(ekman_input)

            u_ek_pred = ekman_pred[:, :, 0:1, :, :]
            v_ek_pred = ekman_pred[:, :, 1:2, :, :]

            u_total_pred = u_geo + u_ek_pred
            v_total_pred = v_geo + v_ek_pred

            if test:
                loss = self.loss_func_test(u_total_pred, uo_true) + \
                       self.loss_func_test(v_total_pred, vo_true)
            else:
                loss = self.loss_func_train(u_total_pred, uo_true) + \
                       self.loss_func_train(v_total_pred, vo_true)

        return loss, B


if __name__ == "__main__":
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from configs import parse_args, get_my_config
    from models import PredFormer_Model, Mask_PredFormer_Model, SimVP_Model, RNN
    from dataset import CurrentDataset
    import time
    from mytools import set_all_seeds

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args_ = parse_args()
    args = get_my_config(args_)

    set_all_seeds(args.SEED)
    mask_land = torch.from_numpy(
        np.load(args.path_land_mask)
    )

    if args.model_name == "predformer":
        if args.mask_predformer:
            model = Mask_PredFormer_Model(args.model_config, mask_land)
        else:
            model = PredFormer_Model(args.model_config)
    elif args.model_name in ("gsta", "tau", "ekman"):
        model = SimVP_Model(**args.model_config)
    elif args.model_name == "predrnn":
        model = RNN(args.model_config)

    train_dataset = CurrentDataset(args, mode="train", norm=args.norm)
    eval_dataset = CurrentDataset(args, mode="eval", norm=args.norm)
    test_dataset = CurrentDataset(args, mode="test", norm=args.norm)

    lon, lat = eval_dataset.lon, eval_dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    stds = np.load(args.path_stds)

    log_dir = rf"{args.model_savepath}/{model.__class__.__name__}_seed{args.SEED}/{args.file_name}_{time.strftime('%Y%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    weight_decay = 1e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    loss_func_unpatched = MSELossIgnoreNaN(args, ~mask_land, patched=False)
    loss_func_pinn = MSELossIgnoreNaN(args)

    if args.model_name == "ekman":
        from models import SimVP_Model as SSHModel
        ssh_model = SSHModel(
            in_shape=[args.input_length, 1, args.height, args.width],
            C_out=1,
            hid_S=16, hid_T=128, N_S=4, N_T=6,
            model_type='tau', drop=0.0, drop_path=0.0,
            spatio_kernel_enc=3, spatio_kernel_dec=3,
            mlp_ratio=2.0, gated=False
        )
        if hasattr(args, 'ssh_model_path') and os.path.exists(args.ssh_model_path):
            ssh_state = torch.load(args.ssh_model_path, map_location=args.device)
            ssh_model.load_state_dict(ssh_state)
            print(f"SSH model loaded from {args.ssh_model_path}")
        else:
            print(f"Warning: SSH model not found at {args.ssh_model_path}, using untrained SSH model")
        ssh_model.to(args.device)

        loss_func_test = loss_func_unpatched
        loss_func_train = loss_func_unpatched if args.loss_ignore_nan else nn.MSELoss()
        trainer = EkmanModel(
            model,
            ssh_model,
            args,
            loss_func_test,
            loss_func_train,
            log_dir,
            optimizer,
            scheduler,
            mode="train",
            lon=lon,
            lat=lat,
            stds=stds,
            mask_land=mask_land,
            loss_func_pinn=loss_func_pinn,
        )
    elif args.model_name == "predrnn":
        loss_func_patched = MSELossIgnoreNaN(args, ~mask_land, patched=True)
        loss_func_test = loss_func_patched
        loss_func_train = loss_func_patched if args.loss_ignore_nan else nn.MSELoss()
        trainer = ReModel(
            model,
            args,
            loss_func_test,
            loss_func_train,
            log_dir,
            optimizer,
            scheduler,
            mode="train",
            lon=lon,
            lat=lat,
            stds=stds,
            mask_land=mask_land,
            loss_func_pinn=loss_func_pinn,
        )
    else:
        loss_func_test = loss_func_unpatched
        loss_func_train = loss_func_unpatched if args.loss_ignore_nan else nn.MSELoss()
        trainer = Model(
            model,
            args,
            loss_func_test,
            loss_func_train,
            log_dir,
            optimizer,
            scheduler,
            mode="train",
            lon=lon,
            lat=lat,
            stds=stds,
            mask_land=mask_land,
            loss_func_pinn=loss_func_pinn,
        )

    trainer.train_model(train_dataset, eval_dataset, test_dataset)
