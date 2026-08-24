import os

ssh_inputs = ['ssh_mask', 'ssh_wind_mask']
#ssh_inputs = [ssh_inputs[1]]  # 选择要跑的实验, [0]=ssh_mask, [1]=ssh_wind_mask

# SSH 预训练模型路径 (与 ssh_input 一一对应)
ssh_model_paths = {
    'ssh_mask': '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_mask_20260811_1116/model_paras.pkl',
    'ssh_wind_mask': '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_wind_mask_20260813_0114/model_paras.pkl',
}

# ageo 输入 (非地转流需要风场驱动)
ageo_inputs = ['ssh_wind_mask', 'ssh_wind_mask_lonlat']

# ============= 公共配置 =============
ssh_model_name = 'tau'
model_name = 'tau'

norms = [False, True]
norms = [norms[1]]

loss_ignore_nan = [False, True]
loss_ignore_nan = [loss_ignore_nan[1]]

shuffle = [False, True]
shuffle = [shuffle[1]]

patches = [8, 4, 2]
patches = [patches[1]]

start_time = ['2021-01-01', '1993-01-01']
start_time = [start_time[1]]

num_runs = 0
for st in start_time:
    for norm in norms:
        for ssh_input in ssh_inputs:
            for ageo_input in ageo_inputs:
                for patch in patches:
                    for l in loss_ignore_nan:
                        for s in shuffle:

                            ssh_model_path = ssh_model_paths[ssh_input]

                            cmd_parts = [
                                "python3 -m current_prediction.trainers_sc",
                                "--batch_size_train 4",
                                f"--ssh_model_name {ssh_model_name}",
                                f"--model_name {model_name}",
                                f"--ssh_input {ssh_input}",
                                f"--ageo_input {ageo_input}",
                                f"--ssh_model_path {ssh_model_path}",
                                f"--patch_size {patch}",
                                "--gradient_clip",
                                "--gradient_clip_value 1",
                                "--env linux",
                                "--area scs",
                                "--input_length 10",
                                "--output_length 10",
                            ]

                            num_runs += 1
                            cmd_parts.append(f"--start_time_train {st}")
                            print(f"{'*' * 40} Run {num_runs}: ssh_input={ssh_input}, ageo_input={ageo_input} {'*' * 40}\n")

                            if l:
                                cmd_parts.append("--loss_ignore_nan")
                            if s:
                                cmd_parts.append("--shuffle")
                            if norm:
                                cmd_parts.append("--norm")

                            cmd = " ".join(cmd_parts)
                            print(f"Command: {cmd}\n")
                            os.system(cmd)
                            print(f"over\n")
