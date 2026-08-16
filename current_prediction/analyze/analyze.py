"""
表层流预测核心评估代码
支持双模型架构: SSH模型(冻结) + 非地转流模型(可训练)
评估指标: RMSE (by lead/date), Correlation, 空间MAE, Persistence基准
"""
import os
import re
import torch
import numpy as np
from pathlib import Path

from current_prediction.configs_sc import parse_args, get_my_config, VALID_SSH_INPUTS, VALID_AGEO_INPUTS
from current_prediction.dataset_sc import CurrentDataset
from current_prediction.mytools_sc import (
    MSELossIgnoreNaNCurrent,
    compute_geostrophic_current_gpu,
    set_all_seeds,
)
from ssh_prediction.mytools import MaskPearsonCorr
from models import SimVP_Model


# SSH 预训练模型路径映射
SSH_MODEL_PATHS = {
    'ssh_mask': '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_mask_20260811_1116/model_paras.pkl',
    'ssh_wind_mask': '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_wind_mask_20260813_0114/model_paras.pkl',
}


def extract_model_info(parent_dir):
    """从父目录中提取所有 model_paras.pkl 文件的路径及相关信息"""
    parent_path = Path(parent_dir)
    pattern = re.compile(r'^(.*?)(?:_\d{8}_\d{4})?$')
    results = []

    for folder in parent_path.iterdir():
        if folder.is_dir():
            pkl_file = folder / "model_paras.pkl"
            if pkl_file.exists():
                name = folder.name
                match = pattern.match(name)
                simple_name = match.group(1) if match else name
                results.append((str(pkl_file), simple_name, str(folder)))
    return results


def parse_folder_name(folder_name):
    """
    从文件夹名解析配置
    格式: current_{ageo_input}[_pinn{lambda}]_norm_{timestamp}
    返回: dict with ageo_input, pinn, norm, ssh_input
    """
    info = {'ageo_input': None, 'pinn': None, 'norm': False, 'ssh_input': None}

    name = re.sub(r'_\d{8}_\d{4}$', '', folder_name)
    if name.startswith('current_'):
        name = name[len('current_'):]

    if name.endswith('_norm'):
        info['norm'] = True
        name = name[:-len('_norm')]

    m = re.search(r'_pinn([\d.]+)$', name)
    if m:
        info['pinn'] = float(m.group(1))
        name = name[:m.start()]

    info['ageo_input'] = name
    # ssh_input 默认与 ageo_input 一致 (不含 lonlat 时)
    if name in VALID_SSH_INPUTS:
        info['ssh_input'] = name
    elif name in VALID_AGEO_INPUTS:
        # 可能含 lonlat, ssh_input 取不含 lonlat 的部分
        parts = name.split('_')
        ssh_parts = [p for p in parts if p != 'lonlat']
        info['ssh_input'] = '_'.join(ssh_parts)

    return info


class CurrentModelEvaluator:
    """表层流预测模型评估器"""

    def __init__(self, parent_dir, save_dir):
        self.parent_dir = parent_dir
        self.save_dir = save_dir
        self.results = {}

    # ---------------------------------------------------------
    # 1. 构造配置
    # ---------------------------------------------------------
    def build_args(self, folder_name, season='full'):
        info = parse_folder_name(folder_name)

        args_ = parse_args()
        args_.ssh_input = info['ssh_input'] or 'ssh_wind_mask'
        args_.ageo_input = info['ageo_input'] or 'ssh_wind_mask'
        args_.ssh_model_path = SSH_MODEL_PATHS.get(info['ssh_input'], '')
        args_.norm = info['norm']
        args_.env = 'linux'
        args_.area = 'scs'

        args = get_my_config(args_)
        return args

    # ---------------------------------------------------------
    # 2. 加载模型
    # ---------------------------------------------------------
    def load_models(self, args, model_para_path):
        """加载 SSH 模型和 ageo 模型"""
        device = args.device

        # SSH 模型 (冻结)
        ssh_model = SimVP_Model(**args.ssh_model_config).to(device)
        if args.ssh_model_path and os.path.exists(args.ssh_model_path):
            ssh_model.load_state_dict(
                torch.load(args.ssh_model_path, map_location=device, weights_only=True)
            )
            print(f"SSH model loaded from {args.ssh_model_path}")
        else:
            print(f"Warning: SSH model not found at '{args.ssh_model_path}'")
        ssh_model.eval()
        for p in ssh_model.parameters():
            p.requires_grad = False

        # ageo 模型
        ageo_model = SimVP_Model(**args.model_config).to(device)
        if os.path.exists(model_para_path):
            ageo_model.load_state_dict(
                torch.load(model_para_path, map_location=device, weights_only=True)
            )
            print(f"Ageo model loaded from {model_para_path}")
        else:
            print(f"Warning: Ageo model not found at '{model_para_path}'")
        ageo_model.eval()

        return ssh_model, ageo_model

    # ---------------------------------------------------------
    # 3. 预测总流
    # ---------------------------------------------------------
    def predict_total(self, ssh_model, ageo_model, ssh_input, datax,
                      args, lon, lat):
        """
        预测总流 = 非地转流 + f_weight * 地转流
        返回: pred_total (B, T_out, 2, H, W)
        """
        device = args.device

        with torch.no_grad():
            # SSH 模型预测
            pred_ssh = ssh_model(ssh_input.to(device))  # (B, T_out, 1, H, W)

            # 反归一化 SSH 到物理量
            pred_ssh_phys = pred_ssh * args.ssh_std + args.ssh_mean

            # 计算地转流
            u_geo, v_geo, f_weight = compute_geostrophic_current_gpu(
                pred_ssh_phys, lon, lat,
                if_solid_f=getattr(args, 'if_solid_f', True)
            )
            u_geo = u_geo[:, :, 0:1]
            v_geo = v_geo[:, :, 0:1]

            # 地转流归一化到与 target 相同的尺度
            u_geo_norm = (u_geo - args.u_c_mu) / args.u_c_std
            v_geo_norm = (v_geo - args.v_c_mu) / args.v_c_std

            # ageo 模型预测非地转流
            pred_ageo = ageo_model(datax.to(device))  # (B, T_out, 2, H, W)

            # 总流 = 非地转流 + f_weight * 地转流
            f_weight = f_weight.to(device)
            pred_total = pred_ageo.clone()
            pred_total[:, :, 0:1] = pred_ageo[:, :, 0:1] + f_weight * u_geo_norm
            pred_total[:, :, 1:2] = pred_ageo[:, :, 1:2] + f_weight * v_geo_norm

        return pred_total

    # ---------------------------------------------------------
    # 4. 单模型评估 (lead + date)
    # ---------------------------------------------------------
    def evaluate_one_model(self, ssh_model, ageo_model, args, test_dataset,
                           mse_func, corr_func, model_name,
                           lon, lat):

        device = args.device
        mask_land = torch.from_numpy(args.mask_land)
        mask_ocean = ~mask_land

        from torch.utils.data import DataLoader
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.results[model_name] = {}

        # 收集所有预测和目标
        all_preds = []
        all_targets = []

        print("  Collecting predictions...")
        for datax, ssh_input, datay in dataloader:
            pred_total = self.predict_total(
                ssh_model, ageo_model, ssh_input, datax, args, lon, lat
            )
            all_preds.append(pred_total.cpu())
            all_targets.append(datay.cpu())

        # 拼接: (N, T_out, 2, H, W)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        T_out = preds.shape[1]

        # ---------- RMSE by lead ----------
        rmse_lead = []
        corr_lead = []
        for t in range(T_out):
            pred_t = preds[:, t]  # (N, 2, H, W)
            target_t = targets[:, t]

            # MSE (2通道平均)
            mask_exp = mask_ocean[None, None].expand_as(pred_t)
            mse = ((pred_t - target_t) ** 2 * mask_exp).sum() / mask_exp.sum()
            rmse_lead.append(torch.sqrt(mse).item())

            # 相关系数 (2通道平均)
            corr_vals = []
            for ch in range(2):
                p = pred_t[:, ch][mask_ocean]
                g = target_t[:, ch][mask_ocean]
                if p.numel() > 1 and p.std() > 0 and g.std() > 0:
                    corr_vals.append(torch.corrcoef(torch.stack([p, g]))[0, 1].item())
                else:
                    corr_vals.append(0.0)
            corr_lead.append(np.mean(corr_vals))

        self.results[model_name]['rmse_lead'] = np.array(rmse_lead)
        self.results[model_name]['corr_lead'] = np.array(corr_lead)

        # ---------- RMSE by date ----------
        N = preds.shape[0]
        rmse_date = []
        corr_date = []
        for i in range(N):
            pred_i = preds[i]  # (T_out, 2, H, W)
            target_i = targets[i]

            mask_exp = mask_ocean[None, None].expand_as(pred_i)
            mse = ((pred_i - target_i) ** 2 * mask_exp).sum() / mask_exp.sum()
            rmse_date.append(torch.sqrt(mse).item())

            corr_vals = []
            for ch in range(2):
                p = pred_i[:, ch][mask_ocean]
                g = target_i[:, ch][mask_ocean]
                if p.numel() > 1 and p.std() > 0 and g.std() > 0:
                    corr_vals.append(torch.corrcoef(torch.stack([p, g]))[0, 1].item())
                else:
                    corr_vals.append(0.0)
            corr_date.append(np.mean(corr_vals))

        self.results[model_name]['rmse_date'] = np.array(rmse_date)
        self.results[model_name]['corr_date'] = np.array(corr_date)

        print(f"  {model_name}: RMSE(lead)={np.mean(rmse_lead):.4f}, "
              f"Corr(lead)={np.mean(corr_lead):.4f}")

    # ---------------------------------------------------------
    # 5. 空间评估
    # ---------------------------------------------------------
    def evaluate_one_model_spatial(self, ssh_model, ageo_model, args, test_dataset,
                                   model_name, lon, lat):

        device = args.device
        mask_land = torch.from_numpy(args.mask_land)
        mask_ocean = ~mask_land

        from torch.utils.data import DataLoader
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.results[model_name] = {}

        all_preds = []
        all_targets = []

        print("  Collecting spatial predictions...")
        for datax, ssh_input, datay in dataloader:
            pred_total = self.predict_total(
                ssh_model, ageo_model, ssh_input, datax, args, lon, lat
            )
            all_preds.append(pred_total.cpu())
            all_targets.append(datay.cpu())

        preds = torch.cat(all_preds, dim=0)   # (N, T_out, 2, H, W)
        targets = torch.cat(all_targets, dim=0)

        # MAE 空间分布 (对 N 和 T_out 平均)
        mae_spatial = torch.abs(preds - targets).mean(dim=(0, 1))  # (2, H, W)
        # 2通道平均
        mae_spatial = mae_spatial.mean(dim=0)  # (H, W)

        self.results[model_name]['mae_spatial'] = mae_spatial.numpy()
        self.results[model_name]['targets_spatial'] = targets.mean(dim=(0, 1)).numpy()
        self.results[model_name]['preds_spatial'] = preds.mean(dim=(0, 1)).numpy()

        print(f"  {model_name}: MAE(ocean)={mae_spatial[mask_ocean].mean().item():.4f}")

    # ---------------------------------------------------------
    # 6. Persistence 基准
    # ---------------------------------------------------------
    def evaluate_persistence(self, args, test_dataset, model_name='Persistence'):

        mask_land = torch.from_numpy(args.mask_land)
        mask_ocean = ~mask_land

        from torch.utils.data import DataLoader
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        self.results[model_name] = {}

        all_preds = []
        all_targets = []

        print("  Collecting persistence predictions...")
        for datax, ssh_input, datay in dataloader:
            # Persistence: 用最后一个输入帧作为预测
            last_input = datax[:, -1:, :args.output_channels]  # (1, 1, 2, H, W) or fewer
            T_out = datay.shape[1]
            # 如果输入通道含 uo/vo, 直接用; 否则用零
            if last_input.shape[2] >= 2:
                pred_persist = last_input.expand(-1, T_out, 2, -1, -1)
            else:
                pred_persist = torch.zeros_like(datay)
            all_preds.append(pred_persist.cpu())
            all_targets.append(datay.cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        T_out = preds.shape[1]

        # RMSE by lead
        rmse_lead = []
        corr_lead = []
        for t in range(T_out):
            pred_t = preds[:, t]
            target_t = targets[:, t]
            mask_exp = mask_ocean[None, None].expand_as(pred_t)
            mse = ((pred_t - target_t) ** 2 * mask_exp).sum() / mask_exp.sum()
            rmse_lead.append(torch.sqrt(mse).item())

            corr_vals = []
            for ch in range(2):
                p = pred_t[:, ch][mask_ocean]
                g = target_t[:, ch][mask_ocean]
                if p.numel() > 1 and p.std() > 0 and g.std() > 0:
                    corr_vals.append(torch.corrcoef(torch.stack([p, g]))[0, 1].item())
                else:
                    corr_vals.append(0.0)
            corr_lead.append(np.mean(corr_vals))

        self.results[model_name]['rmse_lead'] = np.array(rmse_lead)
        self.results[model_name]['corr_lead'] = np.array(corr_lead)

        print(f"  {model_name}: RMSE(lead)={np.mean(rmse_lead):.4f}, "
              f"Corr(lead)={np.mean(corr_lead):.4f}")

    # ---------------------------------------------------------
    # 7. 主运行函数
    # ---------------------------------------------------------
    def run(self, spatial=False, season='full'):

        model_info_list = extract_model_info(self.parent_dir)
        # 只处理 current_ 开头的文件夹
        model_info_list = [
            (p, n, f) for p, n, f in model_info_list
            if 'current' in n.lower()
        ]
        print(f"找到 {len(model_info_list)} 个模型文件，开始处理...")

        last_args = None
        last_lon = None
        last_lat = None
        last_dataset = None

        for model_para_path, simple_name, folder_path in model_info_list:
            print(f"\n{'*' * 50}")
            print(f"处理模型: {simple_name}")
            print(f"模型路径: {model_para_path}")

            folder_name = Path(folder_path).name
            args = self.build_args(folder_name, season)
            set_all_seeds(args.SEED)

            mask_land = torch.from_numpy(args.mask_land)

            lon, lat = None, None
            test_dataset = CurrentDataset(args, mode='test', norm=args.norm)
            lon, lat = test_dataset.lon, test_dataset.lat
            if lon.ndim == 1 and lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            lon = torch.from_numpy(lon.astype(np.float32))
            lat = torch.from_numpy(lat.astype(np.float32))

            mask_ocean = ~mask_land
            mse_func = MSELossIgnoreNaNCurrent(args, mask_ocean)
            corr_func = MaskPearsonCorr(mask_ocean)

            ssh_model, ageo_model = self.load_models(args, model_para_path)

            model_name = simple_name
            if spatial:
                self.evaluate_one_model_spatial(
                    ssh_model, ageo_model, args, test_dataset,
                    model_name, lon, lat
                )
            else:
                self.evaluate_one_model(
                    ssh_model, ageo_model, args, test_dataset,
                    mse_func, corr_func, model_name,
                    lon, lat
                )

            last_args = args
            last_lon = lon
            last_lat = lat
            last_dataset = test_dataset

            # 释放显存
            del ssh_model, ageo_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Persistence 基准
        if not spatial and last_args is not None:
            self.evaluate_persistence(last_args, last_dataset)

        self.save_results(spatial=spatial)

    # ---------------------------------------------------------
    # 8. 保存
    # ---------------------------------------------------------
    def save_results(self, spatial=False):
        if spatial:
            save_dir = os.path.join(self.save_dir, "spatial")
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.save_dir
            os.makedirs(save_dir, exist_ok=True)

        for model_name, metrics in self.results.items():
            print(f"Saving {model_name}...")
            save_path = os.path.join(save_dir, f"{model_name}.npz")
            np.savez(save_path, **metrics)

        print("Evaluation finished.")


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    base_dir = Path(r"/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42")
    save_dir = base_dir / "evaluation"

    evaluator = CurrentModelEvaluator(
        parent_dir=str(base_dir),
        save_dir=str(save_dir),
    )
    evaluator.run(spatial=False, season='full')
