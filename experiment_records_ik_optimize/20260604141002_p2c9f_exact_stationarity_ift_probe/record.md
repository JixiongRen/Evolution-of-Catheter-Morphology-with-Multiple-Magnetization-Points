# P2c9f exact stationarity IFT 审计记录

## 实验目的

P2c9 使用 Gauss-Newton 近似计算 `dy*/dI`，即只用 `J^T J` 近似最优性方程导数。但在 residual 非零时，完整 stationarity IFT 应包含二阶项。本轮实验保持 P2c9 的 soft-continuity formulation 不变，只将灵敏度公式从 GN 近似替换为完整 stationarity IFT，判断 active coil 错配是否来自导数公式近似。

## 实验方案

候选 formulation：

- `cont_1.0` soft continuity。
- clean terminal residual。
- branch anchor 与 P2c9 保持一致。

对比灵敏度：

- `gn_chain`：旧的 Gauss-Newton 近似。
- `exact_ift_chain`：对 `grad_y 0.5||r(y,I)||^2 = 0` 做完整 IFT。

覆盖：

- 四个 P2c4 case。
- 每个 case active coil + 一个随机方向。
- `eps={0.02,0.05,0.10}`。

## 实验过程记录

运行命令见 `commands.sh`。输出：

- `summary.json`
- `summary.csv`
- `direction_summary.csv`
- `solve_summary.csv`
- `heartbeat.jsonl`
- `failure_events.jsonl`
- `sensitivity_audit.json`

## 实验结果

总体分类修正为：`partially-usable`。原因是 3 个 case 的全部审计方向均通过，但 `n3_mixed_rand20` baseline 仍失败，不能把全局结果标成完全 usable。

核心结果：

- 审计行数：`18`。
- usable 行数：`18`。
- 可用方向数：`6/6`。
- failed baseline 数：`1`，即 `n3_mixed_rand20`。
- exact IFT median relative error：约 `0.0314`。
- GN median relative error：约 `0.1269`。

最关键突破：

- `n3_mixed_base_i4 active_coil_5` 从 GN 的约 `0.415` relative error 降到 exact IFT 的约 `0.021`。
- `n3_mixed_base_i4 random_0` 从 GN 的约 `0.158` 降到 exact IFT 的约 `0.029`。
- `n1_tip_z_rand15 random_0` 从 GN 的约 `0.302` 降到 exact IFT 的约 `0.047`。
- `n2_z_pair_rand12` 中 GN 与 exact IFT 都可用，exact IFT 约 `0.058-0.070`。

仍未解决：

- `n3_mixed_rand20` baseline 失败：`residual_norm≈0.0736`，`stationarity_norm≈742.87`，`boundary≈0.0564`。
- 这说明 exact IFT 已修复 sensitivity formula 问题，但不能自动修复 hard current baseline。

## 数据分析

重点判断：

- exact IFT 是否显著优于 GN。
- `n3_mixed_base_i4 active_coil_5` 是否从失败变为可用。
- hard case `n3_mixed_rand20` 是否仍因 baseline 不稳失败。
- Hessian condition 是否导致 exact IFT 不稳定。

分析结论：

- P2c9/P2c9c/P2c9d 中最核心的 active coil mismatch，主要不是刚段显式耦合缺失，也不是 continuity formulation 本身，而是 GN 近似遗漏了 residual 非零时的二阶项。
- 完整 stationarity IFT 对 `grad_y 0.5||r(y,I)||^2=0` 求导，包含 Hessian 与 `d grad/dI`，因此能正确描述 `dy*/dI`。
- fixed-y 显式响应为 0 仍然是事实，但这并不阻止完整 IFT 捕捉通过平衡解分支传递的电流响应。
- 目前可用梯度应标记为 `soft-continuity exact-stationarity IFT`，不是原始 GN chain。
- hard current case `n3_mixed_rand20` 仍属于 baseline recovery 问题，需要 continuation/multi-start 或目标/权重路径修复。

## 实验结论

P2c9f 是当前长期目标中的关键阶段性突破。

已经达成：

- 建立了更可信的 sensitivity audit harness。
- 在 3 个可稳定形成 baseline 的 case 上，exact IFT 全方向、全 eps 通过。
- 修复了此前长期困扰的 `n3_mixed_base_i4 active_coil_5` 灵敏度错配。
- 证明 GN chain 不应作为 production gradient 主线。

尚未达成：

- 四个 P2c4 case 尚未全部通过，因为 `n3_mixed_rand20` baseline 仍失败。
- 还未进行 production shadow replay。

因此，本轮结果应标记为 `partially-usable`，但已经可以作为后续外层 IK 的 shadow gradient 候选。

## 下一步计划

下一步建议：

- 将 sensitivity audit 主线从 GN chain 切换为 exact stationarity IFT。
- 对 `n3_mixed_rand20` 单独做 baseline recovery，优先使用 continuation/multi-start，而不是再改灵敏度公式。
- 在 `n3_mixed_base_i4` 上做 production shadow replay，记录 predicted/actual reduction，但不改默认 production IK。
- 暂缓 P2c10 刚段载荷耦合修复，除非后续证据显示 exact IFT 仍不足。
