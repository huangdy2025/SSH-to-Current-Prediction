# 分解式流预测方法研究路线

> 2026-09-14 重构：以**分解式方法为论文主线**（物理可解释 + 创新点），e2e 作为无物理基线对照，不再优化 e2e。
> 单次训练成本约 24–30 小时（RTX 4090）。

## 0. 定位与论文故事线

- **方法贡献**：物理分解架构（地转解析精确满足 + 非地转网络学习）+ 掩膜感知地转算子 + 联合可训练 + 空间变化 f 的物理保真
- **物理发现**：分解出的年龄涡分量与风应力的相关结构、季节（季风）循环——依赖修复后的干净分解
- **对照**：e2e（0.0716/0.1362 m/s）与 Persistence（0.0853/0.1911）作为基线；分解式目标 = 追平或超越 e2e，同时提供 e2e 给不出的物理诊断
- **方法论立场**：分解式的物理在**结构**里（地转恒等成立），不做损失层面的物理约束（PINN）——那是给无物理结构模型（如 e2e）的补救手段

当前基线（分解最优 current_20260821_0201）：RMSE@1d 0.0724 / @10d 0.1364，Corr@10d 0.7966；与 e2e 差距：lead1 +1.1%、lead10 +0.1%。

## 1. 已完成的修复（2026-09-14，待重训生效）

**a. 海岸地转流污染【已修复，实测影响重大】**
- 成因：Sobel 梯度让陆地格点的 SSH 值（冻结 SSH 模型在陆地上无损失约束的输出）直接参与紧邻陆地海洋格点的差分；区域边缘另有 replicate padding 复制值问题。
- 实测量级（观测 SSH、陆地填均值模拟）：离陆 1 格海洋带的地转流平均被污染 **0.41 m/s**、最大 3.68 m/s（全场地转流量级仅 ~0.12 m/s）；离陆 2 格平均 0.09；开阔海域 0.003。
- 修复：`current_prediction/mytools_sc.py::compute_gradients_masked` 掩膜感知差分（两侧皆海洋→中心差分、一侧→单侧差分、无邻居→梯度 0 交给网络；区域边缘自动单侧差分取代 replicate padding；索引表按静态掩膜预计算缓存）。`compute_geostrophic_current_gpu` 新增 `ocean_mask` 参数，训练（主损失+PINN）与评估两端均已接入。注意：单纯换中心差分解决不了内部海岸线污染，必须掩膜感知。
- 对全域指标的影响估计 <1%（海岸带仅占 7.5% 海洋格点、近岸目标流场本身弱、网络已学会大部分抵消——分解模型离陆 1 格比 e2e 差 20%，开阔海域只差 0.7%，修复后该超额误差应消失）；主要价值在物理正确性与后续诊断。
- `ssh_prediction` 的 SSH PINN 存在同样 Sobel 问题，将来重训 SSH 时应换同一算子。

**b. 地转归一化 μ 偏移【已修复，对当前指标无可测影响】**
- 原公式 `(u_geo−u_c_mu)/u_c_std` 重复扣总流均值（σ 可分配、μ 不可分配），ageo 网络被迫携带补偿偏置。已改为 `u_geo/u_c_std`（trainers_sc.py 与 analyze.py 两处）。
- 旧 checkpoint 不可用新公式重评（网络已含补偿偏置），须重训。启用 sigmoid w 后此修复从"正确性"变为"必要性"（否则补偿项是 w(x,y)·μ/σ 空间图案）。

## 2. 关键代码事实（决定后续设计）

1. **现有 PINN 是惰性的且在分解框架中没有位置**：`trainers_sc.py:82-83` 中 `pred_ssh` 在 `torch.no_grad()` 下计算，`_compute_pinn_loss` 的输出对可训练参数是常数（梯度为零），且方向也是错的（拉向零）。**处置：不修复、不启用**——地转平衡在分解框架中由结构精确满足，损失层面的地转约束是无的放矢；SSH 侧的地转正则属于 SSH 预训练任务且已做（0114 检查点带 pinn λ=0.7）；年龄涡无封闭定律可约束。分解式的物理在结构里，不在损失里。
2. SSH 模型全程冻结（`requires_grad=False`）：SSH 预测误差经 g/f 微分放大后进入总流，无法被下游修正——这是 lead1 落后 e2e 的主要嫌疑。
3. 常数 f 与 SSH 预训练的关系需要重写（2026-09-14 新发现）：`ssh_prediction/mytools.py::compute_geostrophic_current` 原以**位置传参**调用 `compute_f_and_sigmoid_weight(lat, if_solid_f)`，而签名为 `(lat, k=2, phi0=5, if_solid_f=False)`——`if_solid_f` 实际传进了 `k`，真正的 `if_solid_f` 恒取默认 False。**即历史上所有日志记录 "if_solid_f: True" 的 SSH 训练（含 0114），PINN 实际运行的是空间变化 f + sigmoid 权重（k=1），并非常数 f**；日志与实际计算一直是两回事。该 bug 已修复（改关键字传参）。`current_prediction` 侧（`mytools_sc.py`）一直是正确的关键字传参——因此旧 current 框架（常数 f）与 SSH 预训练（实际空间 f）本就存在失配。物理备忘：常数 f 在 2°N 低估南部地转流 ~6 倍，由 ageo 网络吸收（南部"非地转"标记错误）。
4. `configs_sc.py` 的 `--if_solid_f` 已改为 BooleanOptionalAction（2026-09-14），`--no-if_solid_f` 可启用空间变化 f；两框架的实验命名均增加 `_spatialf` 后缀（ssh 侧同时恢复了丢失的 `_pinnλ/_norm` 后缀逻辑——8 月运行后该逻辑被删，重训目录将无法区分配置）。
5. SSH PINN 的 `u_std/v_std` 原为硬编码魔数 0.23（`stds` 传标量触发该分支），已改为从 stats.npz 读取真实 `uo_std/vo_std`（元组传入）；SSH 侧地转计算同样接入了掩膜感知梯度算子（`compute_geostrophic_current` 新增 `ocean_mask` 参数，替换挂着 #todo 的 Sobel）。

## 3. 分阶段实验路线

### 阶段一：修正后的公平基线（1 组）
SSH 检查点沿用 var_ssh_wind_mask_20260813_0114（常数 f 一致），叠加：掩膜梯度 + 去 μ + 更充分训练。
```bash
python3 -m current_prediction.trainers_sc --env linux --area scs \
  --ssh_input ssh_wind_mask \
  --ssh_model_path output/scs/SimVP_Model_seed42/var_ssh_wind_mask_20260813_0114/model_paras.pkl \
  --ageo_input ssh_wind_mask --norm --shuffle --loss_ignore_nan \
  --num_epochs 300 --patience 20
```
验证点：离陆 1 格相对 e2e 的 20% 超额误差消失；全域指标持平或小幅改善。

### 阶段二：联合微调【优化贡献：消除冻结误差传播】
解冻 SSH 模型、端到端微调整个分解管线（这不引入物理约束——物理仍在结构里；解决的是优化问题）：
- SSH 参数分组用小 lr（~1e-5），ageo 正常 lr（1e-4）；`pred_ssh` 改为带梯度计算（去掉 no_grad 与 requires_grad=False）
- 预期：直接攻击 lead1 差距（SSH 误差不再不可修正，微调使 SSH 预测向"有利于流场"的方向适配）；显存上升，必要时 batch 2 + acc_steps 2
- 消融：{冻结基线（阶段一）, 联合微调}；若微调导致 SSH 退化可加 SSH 侧蒸馏项稳住

### 阶段三：空间变化 f + 低纬 sigmoid（物理保真）【进行中，2026-09-14 启动】
SSH 两个对照变体已于 2026-09-14 串行启动（tau + ssh_wind_mask + pinn 0.7 + norm + loss_ignore_nan + shuffle + 梯度裁剪 1.0 + 等效 batch 4 = 实际 2 × acc 2，掩膜梯度算子 + 修复后的 f 传参 + 真实 std 已生效）：
- Run A（真·常数 f）：`var_ssh_wind_mask_pinn0.7_norm_20260914_1759`
- Run B（空间 f + sigmoid）：`--no-if_solid_f`，目录带 `_spatialf` 后缀，Run A 结束后自动开始
- 注意：由于上述传参 bug，历史上并不存在真正的常数 f SSH 模型——这两组是第一次"名副其实"的对照；对照结论应与 0114（日志常数 f、实际空间 f k=1）三方比较
- 完成后：current 框架用 `--no-if_solid_f` + 对应 SSH 检查点重训（`SSH_MODEL_PATHS` 需补 `_spatialf` 路径），验证南部（<7°N）误差下降；此时 sigmoid w 生效，去 μ 修复成为必要

### 阶段四：年龄涡物理（创新点候选，按阶段二三结果选做）
- 风应力特征：τ = ρₐ·Cd·|U|·U 及其旋度作为 ageo 输入通道（由现有 ERA5 风解析计算，无新数据）
- 或三分量分解：总流 = 地转 + Ekman 解析 + 网络残差（重新实现 surface_current 的思路于 current_prediction 内，不启用旧 NPU 代码）

### 阶段五：物理诊断与论文证据
修复后 ageo 分量才干净，运行（需补齐脚本中的 TODO 路径）：`analyze/evaluate_wind_corr.py`（误差/ageo vs 风速相关）、季节循环分析、空间 MAE 图（海岸带 vs 开阔海域）、与 e2e 的逐 lead 对比。必要时多 seed（43/44）集成收尾。

## 4. 排期与成本

| 阶段 | 训练组数 | GPU 时长 | 产出 |
|---|---|---|---|
| 一 | 1 | ~30h | 公平基线 + 海岸修复验证 |
| 二 | 1–2（消融） | ~30h×n | 联合微调（消除冻结误差传播） |
| 三 | SSH 1 + current 1–2 | ~20h + 25h×n | 物理保真版（空间 f + sigmoid） |
| 四 | 1–2 | ~25h×n | 创新点扩展（Ekman/风应力先验） |
| 五 | 0（分析） | ~小时级 | 论文图表 |

## 5. 不做

- 优化 e2e（方向已定，e2e 仅作基线）
- `surface_current/` 旧 NPU 代码（思路可在阶段四重新实现）
- 在旧 checkpoint 上用新公式重评（已含补偿偏置，现有 npz 保留即可）
