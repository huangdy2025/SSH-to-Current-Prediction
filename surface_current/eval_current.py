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
import math
from mytools import MSELossIgnoreNaN, MaskPearsonCorr


def compute_rmse(pred, target, mask=None):
    """计算RMSE，忽略NaN/陆地"""
    if mask is not None:
        valid = ~mask
        pred = pred[valid]
        target = target[valid]
    else:
        valid = ~(torch.isnan(target) | torch.isinf(target))
        pred = pred[valid]
        target = target[valid]

    if pred.numel() == 0:
        return torch.tensor(0.0)

    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_pcc(pred, target, mask=None):
    """计算Pearson相关系数"""
    if mask is not None:
        valid = ~mask
        pred = pred[valid]
        target = target[valid]
    else:
        valid = ~(torch.isnan(target) | torch.isinf(target))
        pred = pred[valid]
        target = target[valid]

    if pred.numel() < 2:
        return torch.tensor(0.0)

    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-10)

    return numerator / denominator


def compute_speed(u, v):
    """计算流速大小"""
    return torch.sqrt(u ** 2 + v ** 2)


def evaluate_surface_current(pred_u, pred_v, target_u, target_v, mask=None, lat=None):
    """
    评估表层流预测

    参数:
        pred_u, pred_v: 预测的流速分量
        target_u, target_v: 真值流速分量
        mask: land mask (True=land)
        lat: 纬度网格 (用于加权)

    返回:
        metrics: dict
    """
    metrics = {}

    # 总流速
    pred_speed = compute_speed(pred_u, pred_v)
    target_speed = compute_speed(target_u, target_v)

    # RMSE (分量)
    metrics['RMSE_u'] = compute_rmse(pred_u, target_u, mask).item()
    metrics['RMSE_v'] = compute_rmse(pred_v, target_v, mask).item()

    # RMSE (总流速)
    metrics['RMSE_speed'] = compute_rmse(pred_speed, target_speed, mask).item()

    # PCC (分量)
    metrics['PCC_u'] = compute_pcc(pred_u, target_u, mask).item()
    metrics['PCC_v'] = compute_pcc(pred_v, target_v, mask).item()

    # PCC (总流速)
    metrics['PCC_speed'] = compute_pcc(pred_speed, target_speed, mask).item()

    # MAE (分量)
    if mask is not None:
        valid = ~mask
        metrics['MAE_u'] = torch.mean(torch.abs(pred_u[valid] - target_u[valid])).item()
        metrics['MAE_v'] = torch.mean(torch.abs(pred_v[valid] - target_v[valid])).item()
    else:
        metrics['MAE_u'] = torch.mean(torch.abs(pred_u - target_u)).item()
        metrics['MAE_v'] = torch.mean(torch.abs(pred_v - target_v)).item()

    # 纬度加权RMSE (如果提供纬度)
    if lat is not None:
        lat_tensor = torch.from_numpy(lat).float()
        weight = torch.cos(torch.deg2rad(lat_tensor))
        weight = weight / weight.mean()

        pred_speed_w = pred_speed * weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        target_speed_w = target_speed * weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        metrics['RMSE_speed_weighted'] = compute_rmse(pred_speed_w, target_speed_w, mask).item()

    return metrics


def compare_with_geo_only(u_geo, v_geo, u_total_pred, v_total_pred, u_target, v_target, mask=None):
    """
    对比: 纯地转流 vs 地转+Ekman修正

    返回:
        comparison: dict
    """
    metrics_geo = evaluate_surface_current(u_geo, v_geo, u_target, v_target, mask)
    metrics_total = evaluate_surface_current(u_total_pred, v_total_pred, u_target, v_target, mask)

    comparison = {}
    for key in metrics_geo:
        comparison[f'geo_{key}'] = metrics_geo[key]
        comparison[f'total_{key}'] = metrics_total[key]
        improvement = (metrics_geo[key] - metrics_total[key]) / metrics_geo[key] * 100
        if 'RMSE' in key or 'MAE' in key:
            comparison[f'{key}_improvement_%'] = improvement
        else:
            comparison[f'{key}_improvement_%'] = -improvement

    return comparison


def print_metrics(metrics, title="Metrics"):
    """打印评估指标"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if 'RMSE' in key:
            print(f"  {key}: {value:.6f} m/s")
        elif 'PCC' in key:
            print(f"  {key}: {value:.4f}")
        elif 'MAE' in key:
            print(f"  {key}: {value:.6f} m/s")
        elif 'improvement' in key:
            print(f"  {key}: {value:+.2f}%")
        else:
            print(f"  {key}: {value:.6f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_npz', type=str, required=True, help='Prediction .npz file')
    parser.add_argument('--target_nc', type=str, required=True, help='Ground truth NC file')
    parser.add_argument('--mask_npy', type=str, default=None, help='Land mask .npy file')
    parser.add_argument('--lat_npy', type=str, default=None, help='Latitude .npy file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()

    # 加载预测
    pred = np.load(args.pred_npz)
    u_total = torch.from_numpy(pred['u_total']).float()
    v_total = torch.from_numpy(pred['v_total']).float()
    u_geo = torch.from_numpy(pred['u_geo']).float()
    v_geo = torch.from_numpy(pred['v_geo']).float()

    # 加载真值
    import xarray as xr
    truth_data = xr.open_dataset(args.target_nc)
    u_target = torch.from_numpy(truth_data.uo.values.astype(np.float32)).unsqueeze(1)
    v_target = torch.from_numpy(truth_data.vo.values.astype(np.float32)).unsqueeze(1)

    # 加载mask
    mask = None
    if args.mask_npy is not None:
        mask = torch.from_numpy(np.load(args.mask_npy)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(u_target)

    # 加载纬度
    lat = None
    if args.lat_npy is not None:
        lat = np.load(args.lat_npy)

    # 评估
    metrics_total = evaluate_surface_current(u_total, v_total, u_target, v_target, mask, lat)
    print_metrics(metrics_total, "Total Current (Geo + Ekman) Metrics")

    # 对比
    comparison = compare_with_geo_only(u_geo, v_geo, u_total, v_total, u_target, v_target, mask)
    print_metrics(comparison, "Comparison: Geo Only vs Geo + Ekman")

    # 保存结果
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        np.savez(
            os.path.join(args.output_dir, 'evaluation_results.npz'),
            metrics_total=metrics_total,
            comparison=comparison
        )
        print(f"Results saved to {args.output_dir}/evaluation_results.npz")
