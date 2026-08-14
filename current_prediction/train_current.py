import os
import sys

# 确保项目根目录在 path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============= 实验配置 =============
# SSH 模型配置
ssh_model_name = 'tau'
ssh_input = 'ssh_mask'        # 'ssh_mask' 或 'ssh_mask_wind'
ssh_model_path = ''            # 预训练 SSH 模型权重路径，例如 'output/scs/SimVP_Model_seed42/var_ssh_mask_xxx/model_paras.pkl'

# 非地转流模型配置
model_name = 'tau'
ageo_input = 'ssh_mask_wind'   # 'ssh_mask_wind' 或 'ssh_mask_wind_lonlat'

# 训练配置
norm = True
is_pinn = False
pinn_lambda = 0.0
loss_ignore_nan = True
shuffle = True
patch_size = 4
batch_size = 4
input_length = 10
output_length = 10
start_time_train = '1993-01-01'
env = 'linux'
area = 'scs'

# ============= 构建命令并运行 =============
cmd_parts = [
    f"{sys.executable} -m current_prediction.trainers_sc",
    f"--batch_size_train {batch_size}",
    f"--model_name {model_name}",
    f"--ssh_model_name {ssh_model_name}",
    f"--ssh_input {ssh_input}",
    f"--ageo_input {ageo_input}",
    f"--patch_size {patch_size}",
    "--gradient_clip",
    "--gradient_clip_value 1",
    f"--env {env}",
    f"--area {area}",
    f"--input_length {input_length}",
    f"--output_length {output_length}",
    f"--start_time_train {start_time_train}",
]

if ssh_model_path:
    cmd_parts.append(f"--ssh_model_path {ssh_model_path}")

if norm:
    cmd_parts.append("--norm")

if is_pinn and pinn_lambda > 0:
    cmd_parts.append("--is_pinn")
    cmd_parts.append(f"--pinn_lambda {pinn_lambda}")

if loss_ignore_nan:
    cmd_parts.append("--loss_ignore_nan")

if shuffle:
    cmd_parts.append("--shuffle")

cmd = " ".join(cmd_parts)
print(f"{'*' * 40} Run {'*' * 40}\n")
print(f"Command: {cmd}\n")
os.system(cmd)
print(f"over\n")
