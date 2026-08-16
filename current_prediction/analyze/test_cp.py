"""
模型性能基准测试
测试 FLOPs、参数量、延迟、吞吐量、显存占用

分别测试 SSH 模型和 ageo 模型的性能
"""
import torch
import time
from thop import profile  # 需要安装: pip install thop


def benchmark_model(model, input_size, device="cuda"):
    model.to(device)
    model.eval()

    # 构造模拟输入
    dummy_input = torch.randn(input_size).to(device)

    # 1. 计算理论复杂度 (FLOPs & Params)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"理论计算量: {flops / 1e9:.4f} GFLOPs")
    print(f"总参数量: {params / 1e6:.4f} M")

    # 2. 预热 (Warm-up)
    print("正在预热...")
    with torch.no_grad():
        for _ in range(200):
            _ = model(dummy_input)

    # 3. 测试延迟 (Latency) & 吞吐量 (Throughput)
    print("开始性能测试...")
    repetitions = 1000
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()

            # 等待 GPU 完成所有任务
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 单位毫秒
            timings.append(curr_time)

    avg_latency = sum(timings) / repetitions
    batch_size = input_size[0]
    throughput = (batch_size * 1000) / avg_latency

    print(f"平均延迟: {avg_latency:.2f} ms")
    print(f"吞吐量: {throughput:.2f} samples/s")

    # 4. 显存占用 (Memory)
    memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    print(f"最大显存保留: {memory_reserved:.2f} MB")


if __name__ == "__main__":
    from current_prediction.configs_sc import parse_args, get_my_config
    from models import SimVP_Model

    args_ = parse_args()
    args_.ssh_input = 'ssh_wind_mask'
    args_.ageo_input = 'ssh_wind_mask'
    args_.ssh_model_path = '/data/hdy/workspace/SSH-to-Current-Prediction/output/scs/SimVP_Model_seed42/var_ssh_wind_mask_20260813_0114/model_paras.pkl'
    args_.norm = True
    args = get_my_config(args_)

    # ============= 测试 SSH 模型 =============
    print("=" * 60)
    print("SSH Model Benchmark")
    print("=" * 60)
    ssh_model = SimVP_Model(**args.ssh_model_config)
    ssh_input_size = (1, args.input_length, args.ssh_input_channels, args.height, args.width)
    print(f"Input size: {ssh_input_size}")
    benchmark_model(ssh_model, ssh_input_size)

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats(args.device)

    # ============= 测试 ageo 模型 =============
    print("\n" + "=" * 60)
    print("Ageo Model Benchmark")
    print("=" * 60)
    ageo_model = SimVP_Model(**args.model_config)
    ageo_input_size = (1, args.input_length, args.input_channels, args.height, args.width)
    print(f"Input size: {ageo_input_size}")
    benchmark_model(ageo_model, ageo_input_size)
