"""
端到端表层流预测核心评估代码
单一模型架构: SSH+wind+mask → u,v 总流 (无地转/非地转分解)
评估指标: RMSE (by lead/date), Correlation, 空间MAE, Persistence基准
"""
import os
import re
import torch
import numpy as np
from pathlib import Path

from e2e_current_prediction.configs import parse_args, get_my_config, VALID_INPUTS
from e2e_current_prediction.dataset import E2ECurrentDataset
from current_prediction.mytools_sc import (
    MSELossIgnoreNaNCurrent,
    set_all_seeds,
)
from models import SimVP_Model


def extract_model_info(parent_dir):
    """从父目录中提取所有 model_paras.pkl 文件的路径及相关信息"""
    parent_path = Path(parent_dir)
    pattern = re.compile(r'^(.*?)(?:_\d{8}_\d{4})?$')
    results = []

    for folder in parent_path.iterdir():
        if folder.is_dir():
            pkl_file = folder / "model_paras.pkl"
            if pkl_file.exists() and folder.name.lower().startswith('e2e_'):
                name = folder.name
                match = pattern.match(name)
                simple_name = match.group(1) if match else name
                results.append((str(pkl_file), simple_name, str(folder)))
    return results


def parse_folder_name(folder_name):
    """
    从文件夹名解析配置
    格式: e2e_current_{e2e_input}_norm_{timestamp}
    返回: dict with e2e_input, norm
    """
    info = {'e2e_input': None, 'norm': False}

    name = re.sub(r'_\d{8}_\d{4}$', '', folder_name)
    if name.startswith('e2e_current_'):
        name = name[len('e2e_current_'):]

    if name.endswith('_norm'):
        info['norm'] = True
        name = name[:-len('_norm')]

    info['e2e_input'] = name

    return info


class E2EModelEvaluator:
    """端到端表层流预测模型评估器"""

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
        args_.e2e_input = info['e2e_input'] if info['e2e_input'] in VALID_INPUTS else 'ssh_wind_mask'
        args_.norm = info['norm']
        args_.env = 'linux'
        args_.area = 'scs'

        args = get_my_config(args_)
        return args

    # ---------------------------------------------------------
    # 2. 加载模型
    # ---------------------------------------------------------
    def load_model(self, args, model_para_path):
        """加载端到端模型"""
        device = args.device

        model = SimVP_Model(**args.model_config).to(device)
        if os.path.exists(model_para_path):
            model.load_state_dict(
                torch.load(model_para_path, map_location=device, weights_only=True)
            )
            print(f"E2E model loaded from {model_para_path}")
        else:
            print(f"Warning: E2E model not found at '{model_para_path}'")
        model.eval()

        return model

    # ---------------------------------------------------------
    # 3. 预测总流
    # ---------------------------------------------------------
    def predict_total(self, model, datax, args):
        """
        预测总流 (端到端直接输出)
        返回: pred_total (B, T_out, 2, H, W)
        """
        device = args.device

        with torch.no_grad():
            pred_total = model(datax.to(device))  # (B, T_out, 2, H, W)

        return pred_total

    # ---------------------------------------------------------
    # 4. 单模型评估 (lead + date)
    # ---------------------------------------------------------
    def evaluate_one_model(self, model, args, test_dataset,
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
        for datax, datay in dataloader:
            pred_total = self.predict_total(
                model, datax, args
            )
            all_preds.append(pred_total.cpu())
            all_targets.append(datay.cpu())

        # 拼接: (N, T_out, 2, H, W)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        T_out = preds.shape[1]

        # 反归一化到物理空间 (m/s)
        preds[:, :, 0] = preds[:, :, 0] * args.u_c_std + args.u_c_mu
        preds[:, :, 1] = preds[:, :, 1] * args.v_c_std + args.v_c_mu
        targets[:, :, 0] = targets[:, :, 0] * args.u_c_std + args.u_c_mu
        targets[:, :, 1] = targets[:, :, 1] * args.v_c_std + args.v_c_mu

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
                p = pred_t[:, ch][:, mask_ocean].flatten()
                g = target_t[:, ch][:, mask_ocean].flatten()
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
                p = pred_i[:, ch][:, mask_ocean].flatten()
                g = target_i[:, ch][:, mask_ocean].flatten()
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
    def evaluate_one_model_spatial(self, model, args, test_dataset,
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
        for datax, datay in dataloader:
            pred_total = self.predict_total(
                model, datax, args
            )
            all_preds.append(pred_total.cpu())
            all_targets.append(datay.cpu())

        preds = torch.cat(all_preds, dim=0)   # (N, T_out, 2, H, W)
        targets = torch.cat(all_targets, dim=0)

        # 反归一化到物理空间 (m/s)
        preds[:, :, 0] = preds[:, :, 0] * args.u_c_std + args.u_c_mu
        preds[:, :, 1] = preds[:, :, 1] * args.v_c_std + args.v_c_mu
        targets[:, :, 0] = targets[:, :, 0] * args.u_c_std + args.u_c_mu
        targets[:, :, 1] = targets[:, :, 1] * args.v_c_std + args.v_c_mu

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

        self.results[model_name] = {}

        input_length = args.input_length
        output_length = args.output_length

        all_preds = []
        all_targets = []

        print("  Collecting persistence predictions...")
        # 按索引遍历, 直接访问 test_dataset.target 获取最后一个观测时刻的流速
        for idx in range(len(test_dataset)):
            _, datay = test_dataset[idx]  # datay: (T_out, 2, H, W)
            # Persistence: 用最后一个观测时刻的流速作为预测
            # target[idx+input_length-1] 是预测窗口前一步的流速
            last_current = test_dataset.target[idx + input_length - 1]  # (2, H, W)
            pred_persist = last_current.unsqueeze(0).expand(
                output_length, -1, -1, -1
            )  # (T_out, 2, H, W)
            all_preds.append(pred_persist.unsqueeze(0))   # (1, T_out, 2, H, W)
            all_targets.append(datay.unsqueeze(0))

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        T_out = preds.shape[1]

        # 反归一化到物理空间 (m/s)
        preds[:, :, 0] = preds[:, :, 0] * args.u_c_std + args.u_c_mu
        preds[:, :, 1] = preds[:, :, 1] * args.v_c_std + args.v_c_mu
        targets[:, :, 0] = targets[:, :, 0] * args.u_c_std + args.u_c_mu
        targets[:, :, 1] = targets[:, :, 1] * args.v_c_std + args.v_c_mu

        # ---------- RMSE by lead ----------
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
                p = pred_t[:, ch][:, mask_ocean].flatten()
                g = target_t[:, ch][:, mask_ocean].flatten()
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
            pred_i = preds[i]
            target_i = targets[i]
            mask_exp = mask_ocean[None, None].expand_as(pred_i)
            mse = ((pred_i - target_i) ** 2 * mask_exp).sum() / mask_exp.sum()
            rmse_date.append(torch.sqrt(mse).item())

            corr_vals = []
            for ch in range(2):
                p = pred_i[:, ch][:, mask_ocean].flatten()
                g = target_i[:, ch][:, mask_ocean].flatten()
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
    # 7. 主运行函数
    # ---------------------------------------------------------
    def run(self, spatial=False, season='full'):

        model_info_list = extract_model_info(self.parent_dir)
        # 只处理 e2e_ 开头的文件夹
        model_info_list = [
            (p, n, f) for p, n, f in model_info_list
            if 'e2e' in n.lower()
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
            test_dataset = E2ECurrentDataset(args, mode='test', norm=args.norm)
            lon, lat = test_dataset.lon, test_dataset.lat
            if lon.ndim == 1 and lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            lon = lon.astype(np.float32)
            lat = lat.astype(np.float32)

            mask_ocean = ~mask_land
            mse_func = MSELossIgnoreNaNCurrent(args, mask_ocean)
            corr_func = None

            model = self.load_model(args, model_para_path)

            # 使用完整文件夹名 (含时间戳) 作为模型名, 避免同配置不同运行的结果互相覆盖
            model_name = Path(folder_path).name
            if spatial:
                self.evaluate_one_model_spatial(
                    model, args, test_dataset,
                    model_name, lon, lat
                )
            else:
                self.evaluate_one_model(
                    model, args, test_dataset,
                    mse_func, corr_func, model_name,
                    lon, lat
                )

            last_args = args
            last_lon = lon
            last_lat = lat
            last_dataset = test_dataset

            # 释放显存
            del model
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

    # 1. 标准评估: RMSE + Corr (lead/date) + Persistence 基准
    print("=" * 60)
    print("E2E Standard Evaluation (RMSE + Corr + Persistence)")
    print("=" * 60)
    evaluator = E2EModelEvaluator(
        parent_dir=str(base_dir),
        save_dir=str(save_dir),
    )
    evaluator.run(spatial=False, season='full')

    # 2. 空间评估: MAE 空间分布
    print("\n" + "=" * 60)
    print("E2E Spatial Evaluation (MAE spatial distribution)")
    print("=" * 60)
    evaluator_spatial = E2EModelEvaluator(
        parent_dir=str(base_dir),
        save_dir=str(save_dir),
    )
    evaluator_spatial.run(spatial=True, season='full')
