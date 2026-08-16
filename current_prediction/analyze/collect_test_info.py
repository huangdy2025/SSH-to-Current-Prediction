import os
import re
import csv
from pathlib import Path


# ==============================
# 正则解析规则（只需要在这里加变量）
# ==============================

PATTERNS = {
    "model_type": r"model_type:\s*(\S+)",
    "model": r"Model:\s*(\S+)",
    "inputs": r"Inputs:\s*([^\n]+)",
    "loss_function": r"Loss Function:\s*([^\n]+)",
    "test_loss": r"Test loss:\s*([\d.]+)",
    "best_val_loss": r"Best validation loss:\s*([\d.]+)",
    "patch_size": r"patch_size:\s*(\d+)",
    "batch_size": r"Batch Size:\s*(\d+)",
    "gradient_clipping": r"Gradient Clipping:\s*([^\n]+)",
    "optimizer": r"Optimizer:\s*(\S+)",
    "total_parameters": r"Total Parameters:\s*([\d,]+)",
    "norm": r"norm:\s*(\S+)",
    "pinn": r"pinn:\s*(\S+)",
    "ssh_input": r"ssh_input:\s*(\S+)",
}


# ==============================
# 文件夹名解析
# ==============================

def parse_folder_name(folder):
    """从文件夹名解析 ageo_input, pinn, norm 信息

    格式: current_{ageo_input}[_pinn{lambda}][_norm]_{timestamp}
    例如: current_ssh_wind_mask_norm_20260815_1234
         -> ageo_input=ssh_wind_mask, norm=True
    例如: current_ssh_wind_mask_pinn0.7_norm_20260815_1234
         -> ageo_input=ssh_wind_mask, pinn=0.7, norm=True
    """
    info = {"ageo_input": None, "pinn": None, "norm": None}

    name = folder

    # 去掉时间戳后缀: _20260815_1234
    name = re.sub(r"_\d{8}_\d{4}$", "", name)

    # 去掉 current_ 前缀
    if name.startswith("current_"):
        name = name[len("current_"):]

    # 检查 _norm 后缀
    if name.endswith("_norm"):
        info["norm"] = "True"
        name = name[:-len("_norm")]

    # 检查 _pinn{value}
    pinn_match = re.search(r"_pinn([\d.]+)$", name)
    if pinn_match:
        info["pinn"] = pinn_match.group(1)
        name = name[:pinn_match.start()]

    # 剩余部分是 ageo_input
    info["ageo_input"] = name if name else None

    return info


# ==============================
# 单文件解析
# ==============================

def parse_log_file(log_file):

    info = {k: None for k in PATTERNS}

    info.update({
        "file_name": Path(log_file).name,
        "folder": Path(log_file).parent.name,
        "final_model_type": None,
        "ageo_input": None,
        "epochs_trained": None,
        "total_time": None
    })

    try:

        # ---------- 文件夹名解析 ----------
        folder_info = parse_folder_name(info["folder"])
        info["ageo_input"] = folder_info["ageo_input"]

        # norm 和 pinn 优先用文件夹名, 回退到日志
        if folder_info["norm"] is not None:
            info["norm"] = folder_info["norm"]
        if folder_info["pinn"] is not None:
            info["pinn"] = folder_info["pinn"]

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # ---------- 正则批量解析 ----------
        for key, pattern in PATTERNS.items():

            match = re.search(pattern, content)

            if not match:
                continue

            value = match.group(1).strip()

            if key in ["test_loss", "best_val_loss"]:
                value = float(value)

            if key == "batch_size":
                value = int(value)

            if key == "patch_size":
                value = int(value)

            if key == "total_parameters":
                value = int(value.replace(",", ""))

            if key == "inputs":
                value = re.sub(r"\s+", "", value)

            # 如果文件夹名已解析出 norm/pinn, 不覆盖
            if key in ["norm", "pinn"] and info[key] is not None:
                continue

            info[key] = value

        # ---------- 确定模型类型 ----------
        info["final_model_type"] = info["model_type"] or info["model"]

        # ---------- 训练轮数 ----------
        early_stop = re.search(
            r"Early stopping triggered at epoch (\d+)", content)

        if early_stop:
            info["epochs_trained"] = int(early_stop.group(1))

        else:
            epochs = re.findall(
                r"Epoch (\d+)/\d+", content)

            if epochs:
                info["epochs_trained"] = int(epochs[-1])

        # ---------- 总时间 ----------
        time_match = re.search(
            r"Total time:\s*([\dhms ]+)", content)

        if time_match:
            info["total_time"] = time_match.group(1).strip()

    except Exception as e:
        print(f"解析失败 {log_file}: {e}")

    return info


# ==============================
# 收集所有日志
# ==============================

def collect_logs(folder, output_csv, std=1.):

    log_files = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.startswith("training") and f.endswith(".log"):
                log_files.append(os.path.join(root, f))

    print(f"找到 {len(log_files)} 个日志")

    results = [parse_log_file(f) for f in log_files]

    results.sort(key=lambda x: (
        (x["final_model_type"] or "").lower(),
        x["file_name"]
    ))

    # ==============================
    # CSV字段
    # ==============================

    fields = [
        "模型类型",
        "文件名",
        "所在文件夹",
        "ssh_input",
        "ageo_input",
        "patch_size",
        "损失函数",
        "批次大小",
        "梯度裁剪",
        "Norm",
        "PINN",
        "总参数量(M)",
        "优化器",
        "测试损失",
        "最佳验证损失",
        "训练轮数",
        "总训练时间",
        "备注"
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()

        current_model = None

        for info in results:

            new_model = (
                current_model is not None
                and current_model != info["final_model_type"]
            )

            current_model = info["final_model_type"]

            if new_model:
                writer.writerow({})

            params = (
                f"{info['total_parameters']/1e6:.2f}M"
                if info["total_parameters"]
                else "N/A"
            )

            row = {

                "模型类型": info["final_model_type"] or "N/A",
                "文件名": info["file_name"],
                "所在文件夹": info["folder"],
                "ssh_input": info["ssh_input"] or "N/A",
                "ageo_input": info["ageo_input"] or "N/A",
                "损失函数": info["loss_function"] or "N/A",
                "patch_size": info["patch_size"] or "N/A",
                "批次大小": info["batch_size"] or "N/A",
                "梯度裁剪": info["gradient_clipping"] or "N/A",
                "Norm": info["norm"] or "N/A",
                "PINN": info["pinn"] or "N/A",
                "总参数量(M)": params,
                "优化器": info["optimizer"] or "N/A",
                "测试损失": f"{info['test_loss'] * std * 100:.2f}" if info["test_loss"] else "N/A",
                "最佳验证损失": f"{info['best_val_loss'] * 100 * std:.2f}" if info["best_val_loss"] else "N/A",
                "训练轮数": info["epochs_trained"] or "N/A",
                "总训练时间": info["total_time"] or "N/A",
                "备注": "↑ 新模型类型 ↑" if new_model else ""

            }

            writer.writerow(row)

    print("CSV 已生成:", output_csv)

    return output_csv

if __name__ == "__main__":

    folder_path = r"/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42"
    csv_path = os.path.join(folder_path, "model_performance_results.csv")
    collect_logs(folder_path, csv_path, std=1.0)
