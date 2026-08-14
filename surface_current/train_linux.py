import os

# surface_current 训练脚本
# 用法: cd surface_current && python3 train_linux.py

model_names = ['tau']

need_mask = [True]

norms = [True]

gated = [False]

pinn_lambdas = [0, 0.1]
pinn_lambdas = [pinn_lambdas[0]]

loss_ignore_nan = [True]

shuffle = [True]

num_runs = 0
for model_name in model_names:
    print(f"{'*'*40} Model {model_name} {'*'*40}")
    for mask in need_mask:
        for g in gated:
            for p in pinn_lambdas:
                for l in loss_ignore_nan:
                    for s in shuffle:
                        for n in norms:

                            cmd_parts = [
                                "python3 trainers.py",
                                "--batch_size_train 16",
                                f"--model_name {model_name}",
                                "--patch_size 8",
                                "--gradient_clip",
                                "--gradient_clip_value 1",
                                "--base /share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/",
                                "--input_length 10",
                                "--output_length 10",
                                "--height 240",
                                "--width 240",
                                "--patience 30",
                                "--scheduler_factor 0.5",
                                "--min_lr 1e-6",
                                "--end_time_test 2020-12-31",
                                "--need_ssh",
                                "--need_wind",
                                "--need_ekman",
                                "--wind_path /share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/wind_240.nc",
                                "--uv_current_path /share/home/group_wangdongxiao/huangjinjun/HDY/20250926/data/processed1/uv_current_240.nc",
                                "--ssh_model_path /share/home/group_wangdongxiao/huangjinjun/HDY/SSH-Prediction-in-the-SCS/output/SimVP_Model_seed42/var_ssh/model_paras.pkl",
                            ]

                            num_runs += 1
                            print(f"{'*'*40} Run {num_runs} {'*'*40}\n")

                            if mask:
                                cmd_parts.append("--need_mask")
                            if g:
                                cmd_parts.append("--gated")
                            if p != 0:
                                cmd_parts.append("--is_pinn")
                                cmd_parts.append(f"--pinn_lambda {p}")
                            if l:
                                cmd_parts.append("--loss_ignore_nan")
                            if s:
                                cmd_parts.append("--shuffle")
                            if n:
                                cmd_parts.append("--norm")

                            cmd = " ".join(cmd_parts)
                            print(f"Running: {cmd}")
                            os.system(cmd)
                            print(f"{'='*80}\n")
