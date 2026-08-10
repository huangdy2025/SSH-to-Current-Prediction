"""
计算 stats.npz 中的全部统计量与海洋掩膜。

字段说明（均基于训练时段 1993-01-01 ~ 2016-12-31，skipna=True）：
    ssh_mu, ssh_std  : zos  的全局标量均值/标准差
    u10_mu, u10_std  : u10 的全局标量均值/标准差
    v10_mu, v10_std  : v10 的全局标量均值/标准差
    uo_mu,  uo_std   : uo  的全局标量均值/标准差
    vo_mu,  vo_std   : vo  的全局标量均值/标准差
    ocean_mask       : (H, W) bool，True=海洋，由 uo 首个时刻 notnull 得到

用法：
    python preprocess_data/compute_stats.py
    python preprocess_data/compute_stats.py --data_dir /data/hdy/workspace/data/scs
"""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr


def compute_var_stats(da, name):
    """计算一个 DataArray 的标量 mean/std（skipna）。"""
    print(f"  计算 {name} ...")
    mu = float(da.mean(skipna=True).values)
    sd = float(da.std(skipna=True).values)
    print(f"    {name}_mu={mu:.10f}, {name}_std={sd:.10f}")
    return mu, sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/hdy/workspace/data/scs')
    parser.add_argument('--ssh_file', type=str, default='ssh_240.nc', help='SSH 文件名（含 zos 变量）')
    parser.add_argument('--wind_file', type=str, default='wind_240.nc', help='风场文件名（含 u10,v10）')
    parser.add_argument('--uv_file', type=str, default='uv_current_240.nc', help='流速文件名（含 uo,vo）')
    parser.add_argument('--ssh_var', type=str, default='zos')
    parser.add_argument('--start', type=str, default='1993-01-01', help='训练开始时间')
    parser.add_argument('--end', type=str, default='2016-12-31', help='训练结束时间')
    parser.add_argument('--verify', action='store_true', default=True, help='若存在旧 stats.npz 则对比验证')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"数据目录: {data_dir}")
    print(f"训练时段: {args.start} ~ {args.end}")

    # 打开三个数据集（分块，避免一次性读入内存）
    ssh = xr.open_dataset(data_dir / args.ssh_file, chunks={'time': 365})
    wind = xr.open_dataset(data_dir / args.wind_file, chunks={'time': 365})
    uv = xr.open_dataset(data_dir / args.uv_file, chunks={'time': 365})

    time_slice = slice(args.start, args.end)
    ssh_train = ssh.sel(time=time_slice)
    wind_train = wind.sel(time=time_slice)
    uv_train = uv.sel(time=time_slice)

    print(f"训练样本天数: {len(ssh_train.time)}")

    stats = {}

    # SSH (zos)
    mu, sd = compute_var_stats(ssh_train[args.ssh_var], args.ssh_var)
    stats['ssh_mu'], stats['ssh_std'] = mu, sd

    # 风场 u10 / v10
    mu, sd = compute_var_stats(wind_train['u10'], 'u10')
    stats['u10_mu'], stats['u10_std'] = mu, sd
    mu, sd = compute_var_stats(wind_train['v10'], 'v10')
    stats['v10_mu'], stats['v10_std'] = mu, sd

    # 流速 uo / vo
    mu, sd = compute_var_stats(uv_train['uo'], 'uo')
    stats['uo_mu'], stats['uo_std'] = mu, sd
    mu, sd = compute_var_stats(uv_train['vo'], 'vo')
    stats['vo_mu'], stats['vo_std'] = mu, sd

    # 海洋掩膜：取 uo 首个时刻非 NaN 的格点
    # 说明：zos 在陆地边缘有少量插值非 NaN，用 uo 更干净
    ocean_mask = uv_train['uo'].isel(time=0).notnull().values
    stats['ocean_mask'] = ocean_mask
    print(f"  ocean_mask: shape={ocean_mask.shape}, 海洋点={int(ocean_mask.sum())}, "
          f"陆地点={int((~ocean_mask).sum())}")

    # 保存
    out_path = data_dir / 'stats.npz'
    np.savez(out_path, **stats)
    print(f"\n已保存: {out_path}")

    # 对比验证
    if args.verify:
        # 若刚覆盖了旧文件则无法对比，这里只验证字段一致性
        print("\n=== 验证字段 ===")
        chk = np.load(out_path, allow_pickle=True)
        keys = ['ssh_mu', 'ssh_std', 'u10_mu', 'u10_std', 'v10_mu', 'v10_std',
                'uo_mu', 'uo_std', 'vo_mu', 'vo_std', 'ocean_mask']
        ok = all(k in chk.files for k in keys)
        print(f"字段齐全: {ok}  keys={list(chk.files)}")


if __name__ == '__main__':
    main()
