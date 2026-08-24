# SSH-to-Current Prediction in the South China Sea

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## Overview

This repository investigates **data-driven surface current prediction** in the South China Sea (SCS) based on sea surface height (SSH) and wind forcing. Sea surface currents can be decomposed into a **geostrophic component**, derived analytically from SSH gradients, and an **ageostrophic component**, primarily wind (Ekman) driven. Instead of predicting currents purely from atmospheric/dynamic reanalysis, we explore how neural networks can reconstruct the surface current field from historical SSH (and wind/mask) sequences.

Three modeling frameworks are implemented and compared, all built on the spatiotemporal predictive models under `models/` (SimVPv2, PredRNNv2, PredFormer):

| Framework | Module | Approach |
|:----------|:-------|:---------|
| Geostrophic + Ageostrophic decomposition | `current_prediction/` | A frozen SSH model predicts future SSH, from which the geostrophic current is computed analytically; a trainable model learns the residual ageostrophic current |
| End-to-end | `e2e_current_prediction/` | A single model learns SSH+wind+mask → total current (u, v) directly, without explicit decomposition |
| Ekman residual | `surface_current/` | Total current = geostrophic current (from predicted SSH) + Ekman current learned from SSH prediction and wind |

---

## Methodology

### 1. Geostrophic + Ageostrophic Decomposition (`current_prediction/`)

The total surface current $U$ is decomposed as:

$$
U = U_{geo} + U_{ageo}
$$

where the geostrophic component is computed from the SSH field $\eta$ via the geostrophic balance:

$$
u_g = -\frac{g}{f}\frac{\partial \eta}{\partial y}, \qquad
v_g = \frac{g}{f}\frac{\partial \eta}{\partial x}
$$

Pipeline:

1. A **frozen pretrained SSH model** predicts the future SSH sequence from historical SSH (with optional wind/mask inputs).
2. The geostrophic current is computed analytically from the predicted SSH. A latitude-dependent weighting is applied so that the geostrophic contribution is suppressed near the equator (`if_solid_f` option: a solid $f$ (mean latitude) vs. a spatially varying $f$), letting the network account for the remaining flow.
3. A **trainable ageostrophic model** takes SSH+wind+mask and predicts the ageostrophic residual $(u_{ageo}, v_{ageo})$.
4. The final total current is $U_{total} = U_{geo} + U_{ageo}$.

An optional **geostrophic PINN loss** (`--is_pinn`, `--pinn_lambda`) can regularize training by penalizing the inconsistency between the geostrophic current implied by the predicted SSH and the target field.

Entry point: `python3 -m current_prediction.trainers_sc`

### 2. End-to-End (`e2e_current_prediction/`)

A single SimVP model directly maps `SSH + wind(u10, v10) + mask → (u, v)` total current over the forecast horizon, with no geostrophic/ageostrophic decomposition and no frozen SSH model. It serves as a baseline to evaluate whether explicit dynamical decomposition helps.

Entry point: `python3 -m e2e_current_prediction.trainers`

### 3. Ekman Residual (`surface_current/`)

Following the Ekman decomposition:

$$
U_{total} = U_{geo} + U_{ekman}
$$

1. A **frozen SSH model** predicts future SSH, from which the geostrophic current is computed.
2. An **Ekman model** takes the predicted SSH and wind (u10, v10) and predicts the Ekman (residual, wind-driven) current.
3. The total current is the sum of the two components, and the loss is computed on the total current against the target (uo, vo).

Entry point: `cd surface_current && python3 trainers.py`

---

## Models

| Model | Category | Notes |
|:------|:---------|:------|
| SimVPv2 (tau / gSTA) | CNN-based | Default backbone in all three current-prediction frameworks |
| PredRNNv2 | RNN-based | Spatiotemporal memory LSTM, autoregressive |
| PredFormer | Transformer-based | Factorized temporal-spatial self-attention |

The SSH prediction component used to derive the geostrophic current can be any of the above pretrained models (see `--ssh_model_path`).

---

## Dataset

| Property | Value |
|:---------|:------|
| SSH source | CMEMS Absolute Dynamic Topography (ADT) |
| Wind source | ERA5 (u10, v10) |
| Current source | CMEMS surface currents (uo, vo) |
| Region | South China Sea, 240 × 240 grid (1/8° resolution) |
| Task | 10-day input → 10-day output |
| Input | SSH (+ wind + land-ocean mask): e.g. (10, 4, 240, 240) for SSH+wind+mask |
| Output | Surface currents (u, v): (10, 2, 240, 240) |
| Train / Val / Test | 1993–2016 / 2017–2018 / 2019–2020 (default in `current_prediction/` and `e2e_current_prediction/`) |

Preprocessing utilities (ERA5 wind download/transform, statistics computation) are provided under `preprocess_data/`.

---

## Training Configuration

| Item | Value |
|:-----|:------|
| Loss Function | MSE (optional NaN ignoring over land, `--loss_ignore_nan`) |
| Optimizer | AdamW (lr=1×10⁻⁴, weight_decay=1×10⁻⁴) |
| Batch Size | 4 (current/e2e), 16 (surface_current) |
| Max Epochs | 200 |
| Early Stopping | patience = 10 |
| Gradient Clipping | norm = 1.0 |
| LR Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Random Seed | 42 |
| Hardware | CUDA GPU (current/e2e); Ascend NPU (surface_current) |

---

## Repository Structure

```
SSH-to-Current-Prediction
│
├── models/                     # Neural network implementations (SimVPv2, PredRNNv2, PredFormer, ...)
├── ssh_prediction/             # SSH prediction module (used as frozen geostrophic source)
├── current_prediction/         # Decomposition framework: geostrophic + ageostrophic current
│   ├── configs_sc.py           #   Config (SSH + ageo model)
│   ├── dataset_sc.py           #   Dataset
│   ├── trainers_sc.py          #   Trainer with geostrophic computation
│   └── train_current.py        #   Batch experiment launcher
├── e2e_current_prediction/     # End-to-end current prediction framework
│   ├── configs.py              #   Config
│   ├── dataset.py              #   Dataset
│   ├── trainers.py             #   E2E trainer
│   └── train_linux.py          #   Batch experiment launcher
├── surface_current/            # Ekman residual framework
│   ├── configs.py              #   Config
│   ├── dataset.py              #   Dataset
│   ├── trainers.py             #   Trainer (geostrophic + Ekman)
│   ├── train_linux.py          #   Batch experiment launcher
│   ├── eval_current.py         #   Evaluation script
│   └── inference.py            #   Inference script
├── preprocess_data/            # Data preprocessing (ERA5 wind, statistics, samples)
├── output/                     # Training outputs (logs, checkpoints, evaluations)
└── figures/                    # Figures
```

### Workflow

```
CMEMS/ERA5 Data → preprocess_data/ → dataset.py → configs.py → trainers*.py → output/
```

---

## Quick Start

**Environment**

```
Python >= 3.9
PyTorch >= 2.0
CUDA >= 11.8 (or Ascend NPU with torch_npu for surface_current)
```

**Decomposition framework (geostrophic + ageostrophic)**

```bash
# Single run
python3 -m current_prediction.trainers_sc \
    --ssh_input ssh_wind_mask \
    --ageo_input ssh_wind_mask \
    --ssh_model_path /path/to/pretrained_ssh_model.pkl \
    --env linux --area scs --input_length 10 --output_length 10

# Batch experiments
python3 current_prediction/train_current.py
```

**End-to-end framework**

```bash
python3 -m e2e_current_prediction.trainers \
    --model_name tau --e2e_input ssh_wind_mask \
    --env linux --area scs --input_length 10 --output_length 10

# Batch experiments
python3 e2e_current_prediction/train_linux.py
```

**Ekman framework**

```bash
cd surface_current && python3 trainers.py \
    --need_ssh --need_wind --need_ekman --need_mask \
    --ssh_model_path /path/to/pretrained_ssh_model.pkl

# Batch experiments
python3 surface_current/train_linux.py
```

---

## Citation

The SSH prediction component of this repository is based on our earlier work. If you find this repository useful for your research, please consider citing:

```bibtex
@article{huang2025investigation,
  title={Investigation of Physics-Informed Methods for Improving Sea Surface Height Prediction Based on Neural Networks in the South China Sea},
  author={Huang, Linxiao and Shu, Yeqiang and Yao, Jinglong},
  journal={Remote Sensing},
  volume={17},
  number={23},
  pages={3838},
  year={2025},
  publisher={MDPI}
}
```

---

## References

- **SimVPv2**: Tan C, Gao Z, Li S, et al. "SimVPv2: Towards Simple Yet Powerful Spatiotemporal Predictive Learning", *arXiv*, 2024
- **PredRNNv2**: Wang Y, Wu H, Zhang J, et al. "PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning", *IEEE TPAMI*, 2023, 45(2): 2208–2225
- **PredFormer**: Tang Y, Qi L, Xie F, et al. "Video Prediction Transformers Without Recurrence or Convolution", *arXiv*, 2025
- **Huang L, Shu Y, Yao J.** Investigation of Physics-Informed Methods for Improving Sea Surface Height Prediction Based on Neural Networks in the South China Sea. *Remote Sensing*, 2025, 17(23): 3838.

---

## Acknowledgements

This work was conducted at the South China Sea Institute of Oceanology, Chinese Academy of Sciences. Data support from Copernicus Marine Environment Monitoring Service (CMEMS), and wind data from ERA5 (ECMWF).

---

## Contact

**Linxiao Huang** — Email: huanglinx@qq.com