"""
流场预测配置模块
支持地转+非地转分解的流场预测框架配置

设计要点:
1. ModelConfig 数据类封装单个模型的配置 (SSH / 非地转流), 消除临时对象
2. 公共参数 (area/env/height/width/input_length 等) 在 args 上只定义一次, 两个 ModelConfig 共享引用
3. 通道解析统一化为 _parse_channels(), SSH 和 ageo 共用同一逻辑
4. validate_config() 在 get_my_config() 末尾校验关联参数一致性
5. _build_simvp_config() 统一模型配置构建, 避免重复调用
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

from ssh_prediction.configs import get_simvp_tau_config, get_simvp_gsta_config


# ============================================================
# 通道定义: 每个变量名对应的通道数
# ============================================================
CHANNEL_MAP: Dict[str, int] = {
    'ssh':    1,
    'mask':   1,
    'wind':   2,   # u10 + v10
    'lonlat': 2,   # lon + lat
}

# 合法的输入组合 (通道顺序须与预训练 SSH 模型一致: ssh, wind, mask)
VALID_SSH_INPUTS = ['ssh_mask', 'ssh_wind_mask']
VALID_AGEO_INPUTS = ['ssh_mask', 'ssh_wind_mask', 'ssh_wind_mask_lonlat']


# ============================================================
# ModelConfig: 封装单个模型 (SSH 或 ageo) 的配置
# ============================================================
@dataclass
class ModelConfig:
    """单个模型的配置容器, 继承公共参数, 补充模型专属参数"""
    name: str = ''              # 模型类型: tau, gsta
    input_spec: str = ''        # 输入组合: ssh_mask, ssh_wind_mask, ssh_wind_mask_lonlat
    input_channels: int = 0
    output_channels: int = 0
    model_config: dict = field(default_factory=dict)

    def build_simvp_config(self, args, model_type: str = 'tau'):
        """构建 SimVP 模型配置字典"""
        builder = get_simvp_tau_config if model_type == 'tau' else get_simvp_gsta_config
        # 临时对象, 只暴露 build 需要的属性
        proxy = _ArgsProxy(
            input_length=args.input_length,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            height=args.height,
            width=args.width,
            gated=args.gated,
        )
        self.model_config = builder(proxy)
        self.name = model_type


@dataclass
class _ArgsProxy:
    """轻量代理对象, 供 get_simvp_*_config() 读取所需属性"""
    input_length: int = 10
    input_channels: int = 0
    output_channels: int = 1
    height: int = 240
    width: int = 240
    gated: bool = False


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train a current prediction model')

    # ---- 公共参数 (两个模型共享) ----
    parser.add_argument('--area', type=str, default='scs')
    parser.add_argument('--env', type=str, default='linux')
    parser.add_argument('--norm', action='store_true', default=False)

    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--output_length', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--gated', action='store_true', default=False)

    # 时间范围
    parser.add_argument('--start_time_train', type=str, default='1993-01-01')
    parser.add_argument('--end_time_train', type=str, default='2016-12-31')
    parser.add_argument('--start_time_val', type=str, default='2017-01-01')
    parser.add_argument('--end_time_val', type=str, default='2018-12-31')
    parser.add_argument('--start_time_test', type=str, default='2019-01-01')
    parser.add_argument('--end_time_test', type=str, default='2020-12-31')

    # ---- SSH 模型参数 (frozen) ----
    parser.add_argument('--ssh_model_name', type=str, default='tau',
                        help='SSH prediction model type')
    parser.add_argument('--ssh_input', type=str, default='ssh_mask',
                        choices=VALID_SSH_INPUTS,
                        help='SSH model input channels')
    parser.add_argument('--ssh_model_path', type=str, default='',
                        help='Path to pretrained SSH model weights')

    # ---- 非地转流模型参数 (trainable) ----
    parser.add_argument('--model_name', type=str, default='tau',
                        help='Current prediction model type (tau, gsta)')
    parser.add_argument('--ageo_input', type=str, default='ssh_wind_mask',
                        choices=VALID_AGEO_INPUTS,
                        help='Ageostrophic model input channels')

    # ---- 训练参数 ----
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_eval', type=int, default=4)
    parser.add_argument('--is_acc', action='store_false', help='Enable gradient accumulation')
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--early_stopping', action='store_false', default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient_clip', action='store_true', default=False)
    parser.add_argument('--gradient_clip_value', type=float, default=1.0)
    parser.add_argument('--gradient_clip_type', type=str, default='norm')

    # ---- Loss ----
    parser.add_argument('--loss_ignore_nan', action='store_true', default=False)
    parser.add_argument('--is_pinn', action='store_true', default=False)
    parser.add_argument('--pinn_lambda', type=float, default=0.)

    # ---- 地转流计算 ----
    parser.add_argument('--if_solid_f', action='store_true', default=True,
                        help='Use constant f (mean lat). Default True: solid f '
                             'to suppress geostrophic contribution near equator')

    # ---- 路径 ----
    parser.add_argument('--model_savepath', type=str,
                        default="/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/")

    # ---- 随机种子 ----
    parser.add_argument('--SEED', type=int, default=42)

    return parser.parse_args()


# ============================================================
# 通道解析 (统一化, SSH 和 ageo 共用)
# ============================================================
def _parse_channels(input_spec: str) -> int:
    """从输入组合字符串解析总通道数"""
    parts = input_spec.split('_')
    total = 0
    for part in parts:
        if part in CHANNEL_MAP:
            total += CHANNEL_MAP[part]
        else:
            raise ValueError(f"Unknown input component '{part}' in '{input_spec}'. "
                             f"Valid components: {list(CHANNEL_MAP.keys())}")
    return total


# ============================================================
# 路径配置
# ============================================================
_DATA_SHAPES = {
    'scs': (240, 240),
    'indian': (480, 480),
    'global': (720, 1440),
}

_DATA_PATHS = {
    'linux': {
        'scs': {
            'base': '/data/hdy/workspace/data/scs/',
            'ssh_file': 'ssh_240.nc',
            'wind_file': 'wind_240.nc',
            'uv_file': 'uv_current_240.nc',
            'stats_file': 'stats.npz',
            'model_savepath': '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/',
        },
    },
}


def _setup_paths(args):
    """设置数据路径, 统一管理"""
    env_paths = _DATA_PATHS.get(args.env)
    if env_paths is None:
        raise ValueError(f"Unsupported env: {args.env}. Supported: {list(_DATA_PATHS.keys())}")

    area_paths = env_paths.get(args.area)
    if area_paths is None:
        raise ValueError(f"Unsupported area: {args.area} for env {args.env}. "
                         f"Supported: {list(env_paths.keys())}")

    args.base = Path(area_paths['base'])
    args.ssh_path = str(args.base / area_paths['ssh_file'])
    args.wind_path = str(args.base / area_paths['wind_file'])
    args.uv_path = str(args.base / area_paths['uv_file'])
    args.stats_path = args.base / area_paths['stats_file']
    args.model_savepath = area_paths['model_savepath']

    args.height, args.width = _DATA_SHAPES[args.area]


# ============================================================
# 统计量加载
# ============================================================
def _load_stats(args):
    """从 stats.npz 加载所有统计量, 统一管理"""
    stats = np.load(args.stats_path, allow_pickle=True)
    args.ssh_mean = float(stats['ssh_mu'])
    args.ssh_std = float(stats['ssh_std'])
    args.u_c_mu = float(stats['uo_mu'])
    args.u_c_std = float(stats['uo_std'])
    args.v_c_mu = float(stats['vo_mu'])
    args.v_c_std = float(stats['vo_std'])
    args.u10_mu = float(stats['u10_mu'])
    args.u10_std = float(stats['u10_std'])
    args.v10_mu = float(stats['v10_mu'])
    args.v10_std = float(stats['v10_std'])
    args.mask_land = ~stats['ocean_mask']  # True=陆地(invalid)


# ============================================================
# 配置验证
# ============================================================
def validate_config(args) -> List[str]:
    """
    校验关联参数的有效性和兼容性
    返回警告列表 (空列表 = 全部通过)
    """
    warnings = []

    # 1. 模型类型一致性检查 (SSH 和 ageo 都用 SimVP)
    if args.ssh_model_name not in ('tau', 'gsta'):
        warnings.append(f"SSH model type '{args.ssh_model_name}' may not be supported "
                        f"(only 'tau' and 'gsta' tested)")

    if args.model_name not in ('tau', 'gsta'):
        warnings.append(f"Ageo model type '{args.model_name}' may not be supported "
                        f"(only 'tau' and 'gsta' tested)")

    # 2. SSH 模型路径检查
    if args.is_pinn and not args.ssh_model_path:
        warnings.append("PINN loss is enabled but no pretrained SSH model path provided. "
                        "Geostrophic PINN loss will use an untrained SSH model.")

    # 3. 输入组合合法性
    if args.ssh_input not in VALID_SSH_INPUTS:
        raise ValueError(f"Invalid ssh_input '{args.ssh_input}'. Valid: {VALID_SSH_INPUTS}")
    if args.ageo_input not in VALID_AGEO_INPUTS:
        raise ValueError(f"Invalid ageo_input '{args.ageo_input}'. Valid: {VALID_AGEO_INPUTS}")

    # 4. 通道数 > 0
    if args.ssh_config.input_channels == 0:
        raise ValueError("SSH model input channels = 0. Check ssh_input config.")
    if args.ageo_config.input_channels == 0:
        raise ValueError("Ageo model input channels = 0. Check ageo_input config.")

    # 5. output_channels 合理性
    if args.ageo_config.output_channels != 2:
        raise ValueError(f"Ageo output_channels must be 2 (u_ageo, v_ageo), got {args.ageo_config.output_channels}")
    if args.ssh_config.output_channels != 1:
        raise ValueError(f"SSH output_channels must be 1 (ssh), got {args.ssh_config.output_channels}")

    # 6. 时间范围合法性
    if args.start_time_train >= args.end_time_train:
        raise ValueError(f"Train time range invalid: {args.start_time_train} >= {args.end_time_train}")
    if args.start_time_val >= args.end_time_val:
        raise ValueError(f"Val time range invalid: {args.start_time_val} >= {args.end_time_val}")
    if args.start_time_test >= args.end_time_test:
        raise ValueError(f"Test time range invalid: {args.start_time_test} >= {args.end_time_test}")

    # 7. 数据路径存在性
    if not Path(args.ssh_path).exists():
        warnings.append(f"SSH data file not found: {args.ssh_path}")
    if not Path(args.wind_path).exists():
        warnings.append(f"Wind data file not found: {args.wind_path}")
    if not Path(args.uv_path).exists():
        warnings.append(f"UV current data file not found: {args.uv_path}")

    # 8. PINN lambda 一致性
    if args.is_pinn and args.pinn_lambda <= 0:
        warnings.append(f"is_pinn=True but pinn_lambda={args.pinn_lambda} (should be > 0)")

    return warnings


# ============================================================
# 主配置构建函数
# ============================================================
def get_my_config(args_):
    """
    构建流场预测的完整配置
    - 公共参数在 args 上只定义一次
    - ssh_config / ageo_config 分别引用公共参数
    - 最终 validate_config() 校验一致性
    """
    args = args_

    # 1. 路径和数据形状
    _setup_paths(args)

    # 2. 统计量
    _load_stats(args)

    # 3. 构建 SSH 模型配置
    args.ssh_config = ModelConfig(
        input_spec=args.ssh_input,
        input_channels=_parse_channels(args.ssh_input),
        output_channels=1,  # SSH only
    )
    args.ssh_config.build_simvp_config(args, model_type=args.ssh_model_name.lower())
    # 快捷别名 (供 trainers_sc.py 直接使用)
    args.ssh_input_channels = args.ssh_config.input_channels
    args.ssh_output_channels = args.ssh_config.output_channels
    args.ssh_model_config = args.ssh_config.model_config
    args.ssh_model_name = args.ssh_model_name.lower()

    # 4. 构建非地转流模型配置
    args.ageo_config = ModelConfig(
        input_spec=args.ageo_input,
        input_channels=_parse_channels(args.ageo_input),
        output_channels=2,  # u_ageo, v_ageo
    )
    args.ageo_config.build_simvp_config(args, model_type=args.model_name.lower())
    # 快捷别名 (供 dataset/trainers 直接使用)
    args.input_channels = args.ageo_config.input_channels
    args.output_channels = args.ageo_config.output_channels
    args.model_config = args.ageo_config.model_config
    args.model_name = args.model_name.lower()

    # 5. 其他派生属性
    args.file_name = f'current_{args.ageo_input}'
    if args.is_pinn:
        args.file_name += f'_pinn{args.pinn_lambda}'
    if args.norm:
        args.file_name += '_norm'
    args.patched = False
    args.one_seq = False

    # 6. 验证
    warnings = validate_config(args)
    if warnings:
        print("=" * 50)
        print("Config validation warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
        print("=" * 50)

    return args
