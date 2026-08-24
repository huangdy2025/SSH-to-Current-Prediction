# SSH-to-Current Prediction in the South China Sea

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 简介

本项目研究基于海表高度（SSH）和风场，利用神经网络预测南海表层流场。总流可分解为**地转流**（由 SSH 梯度解析得到）与**非地转流**（主要由风驱动）。仓库实现了三种预测框架：

| 框架 | 模块 | 说明 |
|:----------|:-------|:---------|
| 地转 + 非地转分解 | `current_prediction/` | 冻结 SSH 模型预测未来 SSH → 解析计算地转流；可训练模型学习非地转残差流 |
| 端到端 | `e2e_current_prediction/` | 单一模型直接学习 SSH+风场+掩膜 → 总流 (u, v)，不做显式分解 |
| Ekman 残差 | `surface_current/` | 总流 = 地转流（来自预测 SSH）+ 由 SSH 预测与风场学习的 Ekman 流 |

## 代码结构

```
SSH-to-Current-Prediction
│
├── models/                     # 网络模型实现（SimVPv2、PredRNNv2、PredFormer 等）
├── ssh_prediction/             # SSH 预测模块（用作冻结的地转流来源）
├── current_prediction/         # 地转 + 非地转分解框架
│   ├── configs_sc.py           #   配置（SSH + ageo 模型）
│   ├── dataset_sc.py           #   数据集
│   ├── trainers_sc.py          #   训练器（含地转流计算）
│   └── train_current.py        #   批量实验启动脚本
├── e2e_current_prediction/     # 端到端流场预测框架
│   ├── configs.py              #   配置
│   ├── dataset.py              #   数据集
│   ├── trainers.py             #   训练器
│   └── train_linux.py          #   批量实验启动脚本
├── surface_current/            # Ekman 残差框架
│   ├── configs.py              #   配置
│   ├── dataset.py              #   数据集
│   ├── trainers.py             #   训练器（地转 + Ekman）
│   ├── train_linux.py          #   批量实验启动脚本
│   ├── eval_current.py         #   评估脚本
│   └── inference.py            #   推理脚本
├── preprocess_data/            # 数据预处理（ERA5 风场、统计量、样本生成）
├── output/                     # 训练输出（日志、权重、评估结果）
└── figures/                    # 图
```

## 数据

- **SSH**：CMEMS Absolute Dynamic Topography (ADT)
- **风场**：ERA5 (u10, v10)
- **流场**：CMEMS 表层流 (uo, vo)
- **区域**：南海，240 × 240 网格（1/8° 分辨率）
- **任务**：10 天输入 → 10 天输出
- **切分**：训练 1993–2016 / 验证 2017–2018 / 测试 2019–2020（`current_prediction/` 与 `e2e_current_prediction/` 默认）

## 快速开始

**环境**

```
Python >= 3.9
PyTorch >= 2.0
CUDA >= 11.8（surface_current 使用昇腾 NPU，需 torch_npu）
```

**地转 + 非地转分解**

```bash
python3 -m current_prediction.trainers_sc \
    --ssh_input ssh_wind_mask \
    --ageo_input ssh_wind_mask \
    --ssh_model_path /path/to/pretrained_ssh_model.pkl \
    --env linux --area scs --input_length 10 --output_length 10

# 批量实验
python3 current_prediction/train_current.py
```

**端到端**

```bash
python3 -m e2e_current_prediction.trainers \
    --model_name tau --e2e_input ssh_wind_mask \
    --env linux --area scs --input_length 10 --output_length 10

# 批量实验
python3 e2e_current_prediction/train_linux.py
```

**Ekman 残差**

```bash
cd surface_current && python3 trainers.py \
    --need_ssh --need_wind --need_ekman --need_mask \
    --ssh_model_path /path/to/pretrained_ssh_model.pkl

# 批量实验
python3 surface_current/train_linux.py
```
