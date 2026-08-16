"""
海岸线区域评估代码
评估表层流预测在海岸线 20km 范围内的 RMSE

架构: 总流 = 地转流(SSH模型预测SSH → 解析计算) + 非地转流(ageo模型学习)
SSH 模型冻结，ageo 模型加载训练好的权重
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt

from current_prediction.configs_sc import parse_args, get_my_config
from current_prediction.dataset_sc import CurrentDataset
from current_prediction.mytools_sc import (
    MSELossIgnoreNaNCurrent,
    compute_geostrophic_current_gpu,
    set_all_seeds,
)
from current_prediction.trainers_sc import CurrentModel
from models import SimVP_Model


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ============= 配置参数 =============
    args_ = parse_args()
    args_.ssh_input = 'ssh_wind_mask'
    args_.ageo_input = 'ssh_wind_mask'
    args_.ssh_model_path = '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_wind_mask_20260813_0114/model_paras.pkl'
    args_.norm = True
    args_ = get_my_config(args_)

    set_all_seeds(args_.SEED)
    device = args_.device

    # ageo 模型权重路径 (训练好的非地转流模型)
    ageo_model_path = '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/current_ssh_wind_mask_norm_20260815_1919/model_paras.pkl'

    # ============= 数据集 =============
    dataset = CurrentDataset(args_, mode='test', norm=args_.norm)
    dataloader = DataLoader(dataset, batch_size=args_.batch_size_eval, shuffle=False)

    # ============= 构建海岸线 mask (20km buffer) =============
    # land_mask: (H, W) True=陆地(invalid)
    land_mask = args_.mask_land
    lat_1d = dataset.lat          # (H,)

    buffer_km = 20

    # Step 1: 计算海像素到陆地的距离（像素单位）
    dist_to_land = distance_transform_edt(~land_mask)

    H, W = land_mask.shape

    # Step 2: 经纬度分辨率换算（每行单独 km/像素）
    # 1°纬度约 111.0 km，经度方向距离随纬度乘 cos(lat)
    pixel_km_per_row = 111.0 * 0.083 * np.cos(np.deg2rad(lat_1d))   # (H,)

    # 为每一行计算 buffer 需要的像素数
    buffer_pixels_per_row = buffer_km / pixel_km_per_row            # (H,)

    # 广播成 (H, W)
    buffer_pixels = buffer_pixels_per_row[:, None]                  # (H, 1) → (H, W)

    # Step 3: 判断是否在海岸 buffer 内（仅保留海侧像素）
    coastal_mask = (dist_to_land <= buffer_pixels) & (~land_mask)
    print('coastal_mask shape:', coastal_mask.shape)
    print('coastal_mask sum:', np.sum(coastal_mask))

    # ============= 构建 SSH 模型 (frozen) =============
    ssh_model = SimVP_Model(**args_.ssh_model_config).to(device)
    if args_.ssh_model_path and os.path.exists(args_.ssh_model_path):
        ssh_state = torch.load(args_.ssh_model_path, map_location=device)
        ssh_model.load_state_dict(ssh_state)
        print(f"SSH model loaded from {args_.ssh_model_path}")
    else:
        print(f"Warning: SSH model not found at '{args_.ssh_model_path}', using untrained SSH model")
    ssh_model.eval()
    for param in ssh_model.parameters():
        param.requires_grad = False
    print(f"SSH model params: {sum(p.numel() for p in ssh_model.parameters()):,}")

    # ============= 构建 ageo 模型 =============
    model = SimVP_Model(**args_.model_config).to(device)
    if os.path.exists(ageo_model_path):
        model.load_state_dict(torch.load(ageo_model_path, map_location=device))
        print(f"Ageo model loaded from {ageo_model_path}")
    else:
        print(f"Warning: Ageo model not found at '{ageo_model_path}', using untrained model")
    model.eval()
    print(f"Ageo model params: {sum(p.numel() for p in model.parameters()):,}")

    # ============= lon/lat 网格 (2D) =============
    lon, lat = dataset.lon, dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # ============= loss (使用海岸线 mask) =============
    loss_func = MSELossIgnoreNaNCurrent(args_, torch.from_numpy(coastal_mask))

    # ============= trainer =============
    trainer = CurrentModel(
        model, ssh_model, args_,
        loss_func_test=loss_func,
        loss_func_train=loss_func,
        mode='test',
        lon=lon, lat=lat, stds=args_.ssh_std,
        mask_land=torch.from_numpy(args_.mask_land),
        ssh_mean=args_.ssh_mean, ssh_std=args_.ssh_std,
        u_c_mu=args_.u_c_mu, u_c_std=args_.u_c_std,
        v_c_mu=args_.v_c_mu, v_c_std=args_.v_c_std,
    )

    rmse = trainer.test_model(dataloader)
    print(f'Coastal RMSE (20km buffer): {rmse}')
