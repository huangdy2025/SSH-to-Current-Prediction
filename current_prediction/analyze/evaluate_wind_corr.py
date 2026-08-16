"""
分析表层流预测误差与风速大小的相关性。

基于 ssh_prediction/analyze/evaluate_wind_corr.py 适配 current_prediction 架构:
- CurrentDataset 返回 (datax, ssh_input, datay) 三元组
- 预测流程: SSH模型 → 地转流 → ageo模型 → 非地转流 → 总流
- 风速从数据集中按预测时间窗口提取
"""
import os
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

try:
    from plotter.spatial_plotter import PlotterMultiPanel
    HAS_PLOTTER = True
except ImportError:
    try:
        from ssh_prediction.plotter.spatial_plotter import PlotterMultiPanel
        HAS_PLOTTER = True
    except ImportError:
        HAS_PLOTTER = False

from current_prediction.mytools_sc import compute_geostrophic_current_gpu, set_all_seeds
from ssh_prediction.mytools import MaskPearsonCorr


# ==============================
# 向量化相关系数
# ==============================
def calc_temporal_corr_p_vectorized(x, y, axis=0):
    N = x.shape[axis]
    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    x_dev = x - x_mean
    y_dev = y - y_mean

    cov = np.sum(x_dev * y_dev, axis=axis)
    var_x = np.sum(x_dev ** 2, axis=axis)
    var_y = np.sum(y_dev ** 2, axis=axis)

    denominator = np.sqrt(var_x * var_y)
    corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator != 0)

    df = N - 2
    r_clipped = np.clip(corr, -1.0 + 1e-8, 1.0 - 1e-8)
    t_stat = r_clipped * np.sqrt(df / (1.0 - r_clipped ** 2))
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)

    return corr, p_val


# ==============================
# 主分析类
# ==============================
class WindStressAnalyzer:
    def __init__(self, args, dataset, pearson_tool,
                 ssh_model, ageo_model_A, ageo_model_B=None,
                 lon=None, lat=None,
                 ssh_mean=0.0, ssh_std=1.0,
                 u_c_mu=0.0, u_c_std=1.0,
                 v_c_mu=0.0, v_c_std=1.0):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        self.pearson_tool = pearson_tool

        self.ssh_model = ssh_model.to(self.device)
        self.ssh_model.eval()

        self.ageo_model_A = ageo_model_A.to(self.device)
        self.ageo_model_A.eval()

        self.ageo_model_B = ageo_model_B
        if self.ageo_model_B is not None:
            self.ageo_model_B = self.ageo_model_B.to(self.device)
            self.ageo_model_B.eval()

        # 地转流计算参数 (2D lon/lat)
        self.lon = lon
        self.lat = lat
        self.if_solid_f = getattr(args, 'if_solid_f', True)

        # 归一化参数
        self.ssh_mean = ssh_mean
        self.ssh_std = ssh_std
        self.u_c_mu = u_c_mu
        self.u_c_std = u_c_std
        self.v_c_mu = v_c_mu
        self.v_c_std = v_c_std

        # 预提取预测时间窗口的风场数据 (视图, 不拷贝)
        self._prepare_wind_views()

    def _prepare_wind_views(self):
        """利用 sliding_window_view 创建预测时间窗口的风场视图"""
        from numpy.lib.stride_tricks import sliding_window_view
        il = self.dataset.input_length
        ol = self.dataset.output_length
        n_samples = len(self.dataset)
        # (T-ol+1, H, W, ol) -> 切片到预测时段 -> (n_samples, H, W, ol)
        u10_win = sliding_window_view(self.dataset.u10, ol, axis=0)
        v10_win = sliding_window_view(self.dataset.v10, ol, axis=0)
        self._u10_pred_view = u10_win[il:il + n_samples]
        self._v10_pred_view = v10_win[il:il + n_samples]

    def _get_wind_batch(self, start, end):
        """提取一个 batch 的风场数据, 返回 (B, ol, H, W) numpy 数组"""
        u10 = np.ascontiguousarray(
            self._u10_pred_view[start:end].transpose(0, 3, 1, 2)
        )
        v10 = np.ascontiguousarray(
            self._v10_pred_view[start:end].transpose(0, 3, 1, 2)
        )
        return u10, v10

    def _predict_total(self, ssh_input, datax, ageo_model):
        """完整预测流程: SSH → 地转流 → ageo → 总流"""
        with torch.no_grad():
            # 1. SSH 模型预测未来 SSH
            pred_ssh = self.ssh_model(ssh_input)  # (B, T_out, 1, H, W)
            pred_ssh_phys = pred_ssh * self.ssh_std + self.ssh_mean

            # 2. 从预测的 SSH 计算地转流
            u_geo, v_geo, f_weight = compute_geostrophic_current_gpu(
                pred_ssh_phys, self.lon, self.lat,
                if_solid_f=self.if_solid_f
            )
            u_geo = u_geo[:, :, 0:1]  # (B, T_out, 1, H, W)
            v_geo = v_geo[:, :, 0:1]

            # 3. 非地转模型预测非地转流
            pred_ageo = ageo_model(datax)  # (B, T_out, 2, H, W)

            # 4. 总流 = 地转流 + 非地转流 (归一化到 target 尺度)
            u_geo_norm = (u_geo - self.u_c_mu) / self.u_c_std
            v_geo_norm = (v_geo - self.v_c_mu) / self.v_c_std
            f_weight = f_weight.to(self.device)

            pred_total = pred_ageo.clone()
            pred_total[:, :, 0:1] = pred_ageo[:, :, 0:1] + f_weight * u_geo_norm
            pred_total[:, :, 1:2] = pred_ageo[:, :, 1:2] + f_weight * v_geo_norm

        return pred_total

    # ==============================
    # 时间相关性 + 多面板绘图
    # ==============================
    def run_full_analysis_time(self, save_dir, mask_land):
        os.makedirs(save_dir, exist_ok=True)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=4)

        collected = {'err': [], 'wind': [], 'imp': []}

        print("Stage 1: Collecting data (Wind Speed only)...")
        sample_offset = 0

        with torch.no_grad():
            for i, (datax, ssh_input, datay) in enumerate(dataloader):
                B = datax.shape[0]
                datax = datax.to(self.device)
                ssh_input = ssh_input.to(self.device)
                datay = datay.to(self.device)

                # 预测总流 (模型 A)
                pred_A = self._predict_total(ssh_input, datax, self.ageo_model_A)

                # 误差: 2通道 (uo, vo) 平均
                err = torch.abs(pred_A - datay)  # (B, T_out, 2, H, W)
                err = err.mean(dim=2)  # (B, T_out, H, W)

                # 改进量 (如果存在模型 B)
                imp = None
                if self.ageo_model_B is not None:
                    pred_B = self._predict_total(ssh_input, datax, self.ageo_model_B)
                    err_B = torch.abs(pred_B - datay).mean(dim=2)
                    imp = err - err_B

                # 风速大小 (预测时间窗口)
                u10_batch, v10_batch = self._get_wind_batch(sample_offset, sample_offset + B)
                wind_mag = np.sqrt(u10_batch ** 2 + v10_batch ** 2)  # (B, T_out, H, W)

                collected['err'].append(err.cpu().numpy())
                collected['wind'].append(wind_mag)
                if imp is not None:
                    collected['imp'].append(imp.cpu().numpy())

                sample_offset += B

        # 拼接
        err = np.concatenate(collected['err'], axis=0)  # (N, T_out, H, W)
        wind = np.concatenate(collected['wind'], axis=0)
        imp = np.concatenate(collected['imp'], axis=0) if collected['imp'] else None

        print("Stage 2: Calculating Correlation & Significance...")
        corr_wind_err, p_wind_err = calc_temporal_corr_p_vectorized(wind, err)

        corr_wind_imp, p_wind_imp = None, None
        if imp is not None:
            corr_wind_imp, p_wind_imp = calc_temporal_corr_p_vectorized(wind, imp)

        # ==============================
        # 绘图
        # ==============================
        print("Stage 3: Multi-panel plotting with significance hatches...")
        self._plot_results(
            corr_wind_err, p_wind_err,
            corr_wind_imp, p_wind_imp,
            mask_land, save_dir,
            filename="wind_corr_significance.png",
            title="Correlation with Wind Speed (Hatches indicate p < 0.05)",
        )

    def _plot_results(self, corr_err, p_err, corr_imp, p_imp,
                      mask_land, save_dir, filename, title):
        lon_min, lon_max = np.min(self.dataset.lon), np.max(self.dataset.lon)
        lat_min, lat_max = np.min(self.dataset.lat), np.max(self.dataset.lat)
        extent = [lon_min, lon_max, lat_min, lat_max]

        mask_valid = ~(mask_land.cpu().numpy() if torch.is_tensor(mask_land) else mask_land)

        target_leads = [0, 3, 6, 9]
        valid_leads = [t for t in target_leads if t < corr_err.shape[0]]

        def prep_with_significance(corr, p):
            corr_maps = []
            sig_masks = []
            for t in valid_leads:
                c_map = np.where(mask_valid, corr[t], np.nan)
                s_mask = (p[t] < 0.05) & mask_valid
                corr_maps.append(c_map)
                sig_masks.append(s_mask)
            return corr_maps, sig_masks

        plot_data = {}
        sig_data = {}

        c_maps, s_masks = prep_with_significance(corr_err, p_err)
        plot_data["Err-Wind"] = c_maps
        sig_data["Err-Wind"] = s_masks

        if corr_imp is not None:
            c_maps_imp, s_masks_imp = prep_with_significance(corr_imp, p_imp)
            plot_data["Imp-Wind"] = c_maps_imp
            sig_data["Imp-Wind"] = s_masks_imp

        vmax = 0.62
        save_path = os.path.join(save_dir, filename)

        if HAS_PLOTTER:
            plotter = PlotterMultiPanel(
                self.dataset.lon, self.dataset.lat, extent=extent
            )
            plotter.plot_panel(
                data=plot_data,
                significance_data=sig_data,
                cbar_range={'data': (-vmax, vmax)},
                title=title,
                save_path=save_path,
                cmap='RdBu_r',
                hatch='////',
            )
        else:
            self._plot_fallback(
                plot_data, sig_data, extent, vmax, save_path, title, valid_leads
            )

        print(f"Saved: {save_path}")

    def _plot_fallback(self, plot_data, sig_data, extent, vmax,
                       save_path, title, valid_leads):
        import matplotlib.pyplot as plt
        n_rows = len(plot_data)
        n_cols = len(valid_leads)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=150
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, (name, maps) in enumerate(plot_data.items()):
            for j, m in enumerate(maps):
                ax = axes[i, j]
                im = ax.imshow(m, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
                if sig_data and name in sig_data:
                    sig = sig_data[name][j]
                    ax.contourf(sig, levels=[0.5, 1.5], hatches=['////'], colors='none')
                ax.set_title(f"{name} Lead {valid_leads[j] + 1}")
                plt.colorbar(im, ax=ax)

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# ==============================
# main
# ==============================
if __name__ == "__main__":
    from current_prediction.configs_sc import parse_args, get_my_config
    from current_prediction.dataset_sc import CurrentDataset
    from models import SimVP_Model

    args_ = parse_args()
    args = get_my_config(args_)

    set_all_seeds(args.SEED)

    mask_land = torch.from_numpy(args.mask_land)  # True=陆地

    # ============= 构建冻结的 SSH 模型 =============
    ssh_model = SimVP_Model(**args.ssh_model_config)
    if args.ssh_model_path and os.path.exists(args.ssh_model_path):
        ssh_model.load_state_dict(
            torch.load(args.ssh_model_path, map_location=args.device, weights_only=True)
        )
        print(f"SSH model loaded from {args.ssh_model_path}")
    else:
        print(f"Warning: SSH model not found at '{args.ssh_model_path}'")

    # ============= 构建非地转流模型 =============
    modelA = SimVP_Model(**args.model_config)
    modelB = SimVP_Model(**args.model_config)

    # TODO: 替换为实际模型权重路径
    model_para_path_A = os.path.join(args.model_savepath, "model_paras.pkl")
    model_para_path_B = os.path.join(args.model_savepath, "model_paras.pkl")

    modelA.load_state_dict(
        torch.load(model_para_path_A, map_location=args.device, weights_only=True)
    )
    modelB.load_state_dict(
        torch.load(model_para_path_B, map_location=args.device, weights_only=True)
    )

    # ============= 数据集 =============
    test_dataset = CurrentDataset(args, mode='test', norm=args.norm)

    lon, lat = test_dataset.lon, test_dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # ============= 分析 =============
    pearson_tool = MaskPearsonCorr(~mask_land)

    analyzer = WindStressAnalyzer(
        args, test_dataset, pearson_tool,
        ssh_model, modelA, modelB,
        lon=lon, lat=lat,
        ssh_mean=args.ssh_mean, ssh_std=args.ssh_std,
        u_c_mu=args.u_c_mu, u_c_std=args.u_c_std,
        v_c_mu=args.v_c_mu, v_c_std=args.v_c_std,
    )

    analyzer.run_full_analysis_time(
        save_dir=os.path.join(args.model_savepath, "wind_corr"),
        mask_land=mask_land,
    )
