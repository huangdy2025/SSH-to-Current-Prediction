# 表层流（uo, vo）预测实验实现方案

## Context

现有代码用于预测 SSH（海面高度），最优实验配置为 tau 模型 + wind + PINN=0.7（`var_ssh_wind_mask_20260813_0114`）。现在需要复用完全相同的模型/训练配置，将预测目标从 SSH 改为表层流（uo, vo），输入保持 ssh+wind+mask（4通道，不含 uv 历史），PINN 物理约束改为地转平衡约束（从输入 SSH 计算地转流，约束预测的 uo/vo）。

## 关键事实

- **tau 模型不做 patchify**：`configs.py:288` `args.patched=False`，仅 predrnn 设 `patched=True`。`patch_size=8` 只被日志记录，tau 实际不用。故 `in_shape=[10,4,240,240]`、`C_out=2`，不涉及 patchify 逻辑。
- **表层流数据**在 `/data/hdy/workspace/data/scs/uv_current_240.nc`（变量 `uo`,`vo`，非 `ugos/vgos`），与 ssh/wind 时间完全对齐。
- **stats.npz 已含** `uo_mu/uo_std/vo_mu/vo_std`，但 `configs.py` 当前未读取。
- **现有 `dataset.py` need_uv 分支有 bug**：引用 `self.data.ugos`（ssh_240.nc 无此变量）、`self.field` 可能为 None、用 ssh 统计量归一化 uv。

## 数据通道设计

uv 目标与输入 field **分离存储**：
- `self.field`（模型输入）= `[ssh, u10, v10, mask]` = 4 通道，与 SSH 模型完全一致，不含 uv
- `self.target_field`（仅 need_uv 时）= `[uo, vo]` = 2 通道，单独归一化

`__getitem__`：need_uv 时 `datax = self.field[...]`（4通道），`datay = self.target_field[...]`（2通道）。

`configs.py` 通道计数：
- `need_ssh`：`input_channels += 1`；**仅当 `not need_uv`** 时 `output_channels += 1`
- `need_uv`：`output_channels += 2`；不加 `input_channels`；`file_name += '_uv'`
- 结果：表层流 input=4, output=2；SSH 任务（need_uv=False）与原来完全一致

## 修改方案

### 1. `ssh_prediction/configs.py` — `get_my_config`

- 第 257 行后新增：`args.uv_current_path = args.base / "uv_current_240.nc"`
- 第 261-266 行 stats 读取后补读：`args.uo_mean/uo_std/vo_mean/vo_std`（从 `stats['uo_mu']` 等）
- 第 275-287 行通道计数改为：
  - `if args.need_ssh:` 块内 `output_channels += 1` 加守卫 `if not args.need_uv:`
  - 新增 `if args.need_uv: args.output_channels += 2; args.file_name += '_uv'`
- `get_simvp_tau_config` **不改**（已是 mlp_ratio=8.0 等最优配置，in_shape/C_out 自动驱动）

### 2. `ssh_prediction/dataset.py` — `MvDataset`

- **`__init__` need_uv 分支重写**（删 ugos/vgos bug）：从 `args.uv_current_path` 加载 `uo,vo`，不进 `self.field`，暂存为 `self.uo/self.vo`
- **归一化后、need_mask 前**，构造 `self.target_field`：`concatenate([uo,vo])`，按 `uo_mean/uo_std`、`vo_mean/vo_std` 归一化，`nan_to_num`，转 tensor
- **`__getitem__`** 加 need_uv 分支：`datay = self.target_field[idx+T_in:idx+T_in+T_out]`

### 3. `ssh_prediction/tools/trainers.py`

- **`_compute_loss`**（第 32-34 行）：PINN 分支按 need_uv 选择损失函数
  ```python
  if getattr(self.config, 'need_uv', False):
      pinn_loss = self._compute_pinn_loss_current(inputs, preds)
  else:
      pinn_loss = self._compute_pinn_loss_sigmoid_weight(preds, targets)
  ```
- **新增 `_compute_pinn_loss_current(self, inputs, preds)`**：地转平衡约束（见下）
- **`main`**（第 151 行）stds 构造：need_uv 时 `stds = (args.ssh_std, args.uo_std, args.vo_std)`，否则 `stds = args.ssh_std`

#### PINN 地转平衡约束逻辑

物理目标：`uo ≈ u_geo(SSH)`, `vo ≈ v_geo(SSH)`。从**输入末帧 SSH**算地转流，广播到 T_out 帧，约束预测 uo/vo。

尺度转换（含均值校正）：
- `compute_geostrophic_current` 输入归一化 SSH，返回 `u_geo = u_geo_phys / ssh_std`
- 转到 uo 归一化尺度：`u_geo_norm = u_geo * ssh_std / uo_std - uo_mean / uo_std` = `(u_geo_phys - uo_mean) / uo_std`
- 减 `uo_mean/uo_std` 是必要的：否则损失最小化会逼 `uo_phys → uo_mean` 而非 `uo_phys → u_geo_phys`，引入 ~12% uo_std 的系统偏差

```python
def _compute_pinn_loss_current(self, inputs, preds):
    ssh_in = inputs[:, -1:, :1, :, :]                        # (B,1,1,H,W) 末帧归一化SSH
    u_geo, v_geo, w = compute_geostrophic_current(ssh_in, self.lon, self.lat, if_solid_f=True)
    ssh_std, u_std, v_std = self.stds                        # (ssh_std, uo_std, vo_std)
    u_geo_norm = u_geo * (ssh_std / u_std) - (self.config.uo_mean / u_std)
    v_geo_norm = v_geo * (ssh_std / v_std) - (self.config.vo_mean / v_std)
    T_out = preds.shape[1]
    u_geo_norm = u_geo_norm.expand(-1, T_out, -1, -1, -1).clone()
    v_geo_norm = v_geo_norm.expand(-1, T_out, -1, -1, -1).clone()
    mask = self.mask_land.expand_as(u_geo_norm)
    u_geo_norm[mask] = torch.nan; v_geo_norm[mask] = torch.nan  # 陆地靠NaN排除
    w = w.to(self.config.device)
    loss_u = self.loss_func_pinn(preds[:,:,0:1] * torch.sqrt(w), u_geo_norm * torch.sqrt(w))
    loss_v = self.loss_func_pinn(preds[:,:,1:2] * torch.sqrt(w), v_geo_norm * torch.sqrt(w))
    return loss_u + loss_v
```

### 4. `ssh_prediction/train_linux.py`

- `vars = [vars[1]]`（只跑 uv）；`need_wind = [need_wind[1]]`（启用 wind）
- `need_mask=[True]`、`model_names=['tau']`、`pinn_lambdas=[0,0.7]`、`loss_ignore_nan=[True]`、`shuffle=[True]`、`patches=[8]` 保持
- `--need_uv` 命令行已存在（第 84-85 行）；`need_ssh` 默认 True 自动带 ssh 输入
- `file_name` = `var_ssh_uv_wind_mask`，与 SSH 的 `var_ssh_wind_mask` 不冲突

### 5. `ssh_prediction/mytools.py` — 不改动

复用 `compute_geostrophic_current`、`MSELossIgnoreNaN`（无 mask_valid 时靠 target NaN 排除陆点）。

## 向后兼容

- SSH 任务（need_uv=False）完全不受影响：output_channels 逻辑、__getitem__、PINN、stds 均走原路径
- predrnn（one_seq=True）不兼容本方案，但用户用 tau，不处理
- 三个 nc 文件时间坐标一致，`sel(time=time_span)` 自动对齐

## 验证

1. 修改后先跑 `python3 -m ssh_prediction.tools.trainers --model_name tau --need_ssh --need_uv --need_wind --need_mask --is_pinn --pinn_lambda 0.7 --norm --loss_ignore_nan --shuffle --gradient_clip --gradient_clip_value 1 --env linux --input_length 10 --output_length 10 --batch_size_train 4` 确认：
   - 日志 `in_shape: [10, 4, 240, 240]`、`C_out: 2`、`Input channels: 4`、`pinn: 0.7`
   - field shape 为 `(T, 4, 240, 240)`、target_field 为 `(T, 2, 240, 240)`
   - 第 1 个 epoch 无 shape 报错、梯度非 inf
2. 确认 SSH 任务仍正常（need_uv=False 时走原路径）
