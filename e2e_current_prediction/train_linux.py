import os

e2e_inputs = ['ssh_wind_mask']

# ============= 公共配置 =============
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
        for e2e_input in e2e_inputs:
            for patch in patches:
                for l in loss_ignore_nan:
                    for s in shuffle:

                        cmd_parts = [
                            "python3 -m e2e_current_prediction.trainers",
                            "--batch_size_train 4",
                            f"--model_name {model_name}",
                            f"--e2e_input {e2e_input}",
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
                        print(f"{'*' * 40} Run {num_runs}: e2e_input={e2e_input} {'*' * 40}\n")

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
