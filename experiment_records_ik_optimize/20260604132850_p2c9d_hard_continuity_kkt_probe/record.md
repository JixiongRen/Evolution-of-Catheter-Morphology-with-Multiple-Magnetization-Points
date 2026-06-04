# P2c9d hard-continuity KKT/IFT 实验记录

## 实验目的

P2c9、P2c9b、P2c9c 表明 soft penalty、纯 6D 初值打靶、base-slack 轻量 formulation 都无法完整解决 shooting 灵敏度问题。本轮实验验证 hard-continuity multiple shooting + KKT/IFT 路线：将段间 wrench continuity 作为硬等式约束，并基于 KKT 系统计算 `dy*/dI`。

## 实验方案

本轮优先覆盖两个核心失败点：

- `n3_mixed_base_i4`：active coil 方向持续灵敏度错配。
- `n3_mixed_rand20`：hard current baseline 不稳定。

变量：

- `y`：原始 18D segment-wrench shooting 变量。
- `lambda`：段间 hard continuity constraint 的乘子。

KKT residual：

- `grad_y L = 0`
- `c(y, I) = 0`

其中 `c` 是段间 propagated wrench 与下一段 proximal wrench 的硬连续约束。

## 实验过程记录

运行命令见 `commands.sh`。输出包括：

- `summary.json`
- `summary.csv`
- `baseline_summary.csv`
- `heartbeat.jsonl`
- `failure_events.jsonl`
- `sensitivity_audit.json`

## 实验结果

本轮 hard-continuity KKT 原型未恢复 baseline，未进入正式灵敏度审计。

结果摘要：

- 总体分类：`untrusted`。
- usable 行数：`0`。
- `n3_mixed_base_i4`：`kkt_norm≈82.06`，`objective_norm≈0.335`，`constraint_norm≈0.877`，`boundary≈0.406`。
- `n3_mixed_rand20`：`kkt_norm≈29.08`，`objective_norm≈0.139`，`constraint_norm≈0.176`，`boundary≈0.226`。
- 两个 case 均为 `F1_baseline_unstable`。

## 数据分析

重点判断：

- hard continuity 是否能恢复 `n3_mixed_rand20` baseline。
- KKT stationarity 与 continuity constraint residual 是否足够低。
- active coil 方向的 IFT chain 是否接近 re-optimized FD。
- 若失败，区分 constraint conflict、conditioning blow-up、baseline infeasible 或 IFT mismatch。

本轮数据说明：

- 直接 hard continuity KKT 从 oracle-y0 初始化无法形成可审计 baseline。
- `constraint_norm` 和 `boundary_residual` 同时较高，说明不仅是 KKT stationarity 未收敛，还存在 hard continuity 与 terminal/boundary 目标之间的约束冲突或吸引域问题。
- 该结果与 P2c9b 一致：一旦强制 wrench 严格连续，当前目标在现有 shooting 模型中很难同时满足 terminal pose 和 final boundary。
- 因此问题可能不只是 branch selection，而是当前 shooting 传播模型对刚段外载、磁场力矩、接口跳变的物理表达不完整。
- 继续单纯调 KKT 初值或迭代次数，短期收益可能较低；下一步应先做 continuation 和模型耦合审计。
- 只读代码审计发现：`forward_kinematics_optimized/segments_nondim.py` 中 `rigid_state_along_dim` 的刚段位置和姿态传播为 `p_sigma = pp + sigma * (R_p @ rigid.v_star)`、`Q_sigma = Qp`。`f_ext_total_dim` 和 `tau_ext_total_dim` 只进入 `f_sigma` 与 `tau_sigma`，不进入刚段 pose update。因此当前模型下磁场外载不会直接改变刚段位姿，只会改变后续传递的 wrench。这是 active coil 方向灵敏度错配的高优物理链路嫌疑。

## 实验结论

P2c9d 直接 hard-continuity KKT/IFT 原型失败，不能作为当前生产灵敏度主线。

目前四条路线的状态：

- P2c9 soft continuity：部分可用，但 active coil 和 hard current 失败。
- P2c9b true continuous 6D：baseline 全失败，自由度不足。
- P2c9c base slack：部分可用，但不优于 soft continuity。
- P2c9d hard KKT：baseline 失败，存在硬约束冲突或吸引域问题。

当前最可靠结论是：需要先审查 shooting 物理耦合链路，并用 continuation 将 hard constraint 从软约束逐步推到硬约束，而不是直接求解硬 KKT。

## 下一步计划

下一步建议推进 P2c9e：

- 对 `n3_mixed_base_i4` 和 `n3_mixed_rand20` 做 soft-to-hard continuation。
- 记录从 soft continuity 到 hard constraint 过程中 terminal residual、boundary、constraint、stationarity 的演化。
- 审查当前 `_advance_rigid` / `rigid_state_along_dim` 中外载是否改变刚段姿态传播；若外载只影响 distal wrench 而不影响刚段 pose，则 shooting sensitivity 可能天然缺少关键显式路径。
- 如果 continuation 显示 hard constraint 不可达，应将 failure 分类升级为 `model_mismatch_or_constraint_conflict`，并转向重新构建 rigid-load coupling。
- 优先审查并修正刚段载荷到姿态传播的耦合方式，再回到 P2c9d/P2c9e 审计。
