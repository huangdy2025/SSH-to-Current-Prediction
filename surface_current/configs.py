import argparse
from pathlib import Path
import re
def parse_args():
    parser = argparse.ArgumentParser(description='Train a surface current model')

    # Data
    parser.add_argument('--base', type=str, default= Path('/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/'))
    parser.add_argument('--output_dir', type=str, default= Path('/share/home/group_wangdongxiao/huangjinjun/HDY/SSH-Prediction-in-the-SCS/surface_current/output/'))

    parser.add_argument('--env', type=str, default='linux',)

    parser.add_argument('--norm', action='store_true', default=False)

    parser.add_argument('--need_ssh', action='store_true', default='/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/ssh_240.nc',
                        help='Whether to include ssh data')
    parser.add_argument('--need_mask', action='store_true', default=False,
                        help='Whether to use mask')

    parser.add_argument('--need_uv', action='store_true', default=False,
                        help='Whether to include uo, vo data')

    parser.add_argument('--need_wind', action='store_true', default=False,
                        help='Whether to include wind data')
    parser.add_argument('--wind_path', type=str,
                        default='/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/wind_240.nc',
                        help='Path to wind data (required if need_wind is True)')

    # Ekman model specific
    parser.add_argument('--need_ekman', action='store_true', default=False,
                        help='Enable Ekman residual model training mode')
    parser.add_argument('--ssh_model_path', type=str,
                        default='/share/home/group_wangdongxiao/huangjinjun/HDY/SSH-Prediction-in-the-SCS/output/SimVP_Model_seed42/var_ssh_20250101_0000/model_paras.pkl',
                        help='Path to pretrained SSH model weights (frozen during Ekman training)')
    parser.add_argument('--uv_current_path', type=str,
                        default='/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/uv_current_240.nc',
                        help='Path to total surface current data (uo, vo)')
    parser.add_argument('--min_max_stat_path', type=str,
                        default='/share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/min_max_stat.npz',
                        help='Path to min-max normalization stats for wind and current')

    # Time range
    parser.add_argument('--start_time_train', type=str, default= '1993-01-02',
                        help='Start time for training data')
    parser.add_argument('--end_time_train', type=str, default= '2019-12-31', 
                        help='End time for training data')
    parser.add_argument('--start_time_val', type=str, default= '2020-01-01', 
                        help='Start time for validation data')
    parser.add_argument('--end_time_val', type=str, default= '2020-06-30', 
                        help='End time for validation data')
    parser.add_argument('--start_time_test', type=str, default= '2020-07-01',
                        help='Start time for test data')
    parser.add_argument('--end_time_test', type=str, default= '2020-12-31',
                        help='End time for test data')

    # Data shape
    parser.add_argument('--input_length', type=int, default= 10,
                        help='Length of input sequence')
    parser.add_argument('--output_length', type=int, default= 10,
                        help='Length of output sequence')
    parser.add_argument('--height', type=int, default=240,
                        help='Image height')
    parser.add_argument('--width', type=int, default=240,
                        help='Image width')
    parser.add_argument('--input_channels', type=int, default=0,
                        help='Number of input channels')
    parser.add_argument('--output_channels', type=int, default=0,
                        help='Number of output channels')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size')

    parser.add_argument('--gated', action='store_true', default=False,
                        help='apply gated conv or not in the models predrnn and simvp')

    # Train
    parser.add_argument('--model_name', type=str, default='tau', )
    parser.add_argument('--mask_predformer', action='store_true', default=False)

    parser.add_argument('--shuffle', action='store_true', default=False)

    parser.add_argument('--device', type=str, default='cuda:0',)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--is_acc', action='store_false', help='Enable gradient accumulation')
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--early_stopping', action='store_false', default=True,)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--gradient_clip', action='store_true',default=False,)
    parser.add_argument('--gradient_clip_value', type=float, default=1.0)
    parser.add_argument('--gradient_clip_type', type=str, default='norm',
                        help='Gradient clip type:norm or value')

    parser.add_argument('--is_pinn', action='store_true')
    parser.add_argument('--pinn_lambda', type=float, default=0.)


    parser.add_argument('--loss_ignore_nan', action='store_true', default=False,
                        help='Whether to ignore nan values in loss')

    parser.add_argument('--model_savepath', type=str, default="/share/home/group_wangdongxiao/huangjinjun/HDY/SSH-Prediction-in-the-SCS/surface_current/output/")

    # 随机种子
    parser.add_argument('--SEED', type=int, default=42)

    return parser.parse_args()

def get_model_configs(args, model_configs_new=None):

    if args.model_name == 'gsta':
        model_config = get_simvp_gsta_config(args)
    elif args.model_name == 'tau':
        model_config = get_simvp_tau_config(args)
    elif args.model_name == 'ekman':
        model_config = get_ekman_config(args)
    elif args.model_name == 'predrnn':
        model_config = get_rnn_config(args)
    elif args.model_name == 'predformer':
        model_config = get_predformer_config(args)
    elif args.model_name == 'rest':
        model_config = get_rest_config(args)
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    if model_configs_new is not None:
        model_config.update(model_configs_new)
    return model_config

def get_simvp_gsta_config(args):

    simvp_gsta_config = {
        "in_shape": [args.input_length, args.input_channels, args.height, args.width],
        "C_out": args.output_channels,
        "hid_S": 16,
        "hid_T": 128,
        "N_S": 2,
        "N_T": 4,
        "model_type": 'gSTA',
        "drop": 0.3,
        "drop_path": 0.3,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3,
        "mlp_ratio": 8.0,
        "gated": args.gated
    }
    return simvp_gsta_config

def get_simvp_tau_config(args):

    simvp_tau_config = {
        "in_shape": [args.input_length, args.input_channels, args.height, args.width],
        "C_out": args.output_channels,
        "hid_S": 16,
        "hid_T": 256,
        "N_S": 4,
        "N_T": 6,
        "model_type": 'tau',
        "drop": 0.2,
        "drop_path": 0.2,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3,
        "mlp_ratio": 2.0,
        "gated": args.gated
    }
    return simvp_tau_config

def get_ekman_config(args):
    """Ekman SimVP: 输入3通道(SSH+u10+v10)，输出2通道(ue, ve)"""
    simvp_tau_config = {
        "in_shape": [args.input_length, 3, args.height, args.width],
        "C_out": 2,
        "hid_S": 16,
        "hid_T": 128,
        "N_S": 4,
        "N_T": 6,
        "model_type": 'tau',
        "drop": 0.2,
        "drop_path": 0.2,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3,
        "mlp_ratio": 2.0,
        "gated": args.gated
    }
    return simvp_tau_config

def get_rnn_config(args):
    rnn_config = {
        'num_layers':3,
        'hidden_dim': 64,
        'in_shape': (args.input_length, args.input_channels, args.height, args.width),
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'img_width': args.width,
        'img_height': args.height,
        'input_length': args.input_length,
        'patch_size': args.patch_size,
        'filter_size': 3,
        'stride': 1,
        'layer_norm': True,
        'reverse_schedule': True,
        'total_length': args.input_length+args.output_length,
        'decouple_beta': 0.3,
        'need_mask': args.need_mask,
        'device': args.device,
        'r_sampling_step_1': 10,
        'r_sampling_step_2': 30,
        'r_exp_alpha': int((30 - 10) / 2),
        'gated': args.gated,
    }
    return rnn_config

def get_predformer_config(args):

    model_config = {
        'height': args.height,
        'width': args.width,
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'input_length': args.input_length,
        'output_length': args.output_length,
        'patch_size': args.patch_size,
        'dim': 128,
        'heads': 4,
        'dropout': 0.3,
        'attn_dropout': 0.3,
        'drop_path': 0.3,
        'scale_dim': 4,
        'depth': 2,
        'Ndepth': 4
    }
    return model_config

def get_rest_config(args):
    model_config = {
        'in_channels': args.input_channels,
        'out_channels': args.output_channels,
        'in_length': args.input_length,
        'out_length': args.output_length,
        'hid_s': 4,
        'n_s': 2,
        'n_t': 12,
        'dropout': 0.3,
        'scale_t': 2
    }
    return model_config

def get_my_config(args_, model_config=None):
    args = args_
    if isinstance(args.base, str):
        args.base = Path(args.base)
    args.datapath =sorted(
    (
        p for p in args.base.iterdir()
        if p.suffix == ".nc" and re.search(r"\d+", p.stem)
    ),
    key=lambda p: int(re.search(r"\d+", p.stem).group())
    )

    args.path_means = args.base / "mean.npy"
    args.path_stds = args.base / "std.npy"
    args.path_land_mask = args.base / "mask.npy"
    args.path_adt_clim = args.base / "adt_clim.npy"

    args.file_name = 'var'
    args.input_channels = 0
    args.output_channels = 0

    if args.need_ssh:
        args.input_channels += 1
        args.output_channels += 1
        args.file_name += '_ssh'

    if args.need_wind:
        args.input_channels += 2
        args.file_name += '_wind'

    if args.need_mask:
        args.input_channels += 1
        args.file_name += '_mask'

    if args.need_ekman:
        args.input_channels = 3
        args.output_channels = 2
        args.file_name = 'var_ekman'

    args.patched=False
    args.one_seq = False

    args.model_name = args.model_name.lower()
    model_config = get_model_configs(args, model_config)

    if args.model_name == 'predrnn':
        args.patched = True
        args.one_seq = True

    args.model_config = model_config
    return args
