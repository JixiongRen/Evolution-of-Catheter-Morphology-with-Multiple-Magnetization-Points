# P2c9 严格 shooting 灵敏度门禁实验记录

## 实验目的

本轮实验面向长期目标：获得稳定、可审计、物理可解释的 shooting-based IK 任务灵敏度 `dg/dI`。P2c8 已经证明 soft 段间力/矩连续性可以改善部分方向，但仍存在方向失败。因此本轮不急于接入 production IK，而是建立更严格的全 case 审计门禁。

## 实验方案

主候选采用 P2c8 当前表现最好的 `cont_1.0`，即在 clean terminal residual 上加入段间 wrench continuity。FK oracle 只用于生成目标工况和初始参考，不作为唯一真值。

审计内容：

- 四个 P2c4 case。
- 三档有限差分步长：`0.02, 0.05, 0.10 A`。
- 每个 case 包含 active coil 方向和两个随机方向。
- 对每个方向计算 implicit/GN chain 与 re-optimized finite difference。
- 对每个 case 做 `oracle_y0`、`zero_y0`、`neighbor_y0` 三种 seed 的 branch gate。
- 对 `I -> I+dI -> I` 做回环检查，识别 branch jump。
- 记录 stationarity、rank、condition number、nullspace size、boundary residual、eps plateau、median rel_err 和 cosine sign。

## 实验过程记录

运行命令见 `commands.sh`。输出文件包括：

- `summary.json`
- `summary.csv`
- `direction_summary.csv`
- `seed_summary.csv`
- `solve_summary.csv`
- `heartbeat.jsonl`
- `failure_events.jsonl`
- `sensitivity_audit.json`
- `stdout.log`
- `stderr.log`

## 实验结果

主审计结果：

- 总体分类：`partially-usable`。
- 总行数：`27`，其中 `24` 行 usable，`3` 行 untrusted。
- 方向总数：`9`，其中 `8` 个方向通过，`1` 个方向失败。
- 全局 median relative error：约 `0.0874`。
- `n2_z_pair_rand12` 和 `n1_tip_z_rand15` 的全部方向、全部 eps 均通过。
- `n3_mixed_base_i4` 的两个随机方向通过，但 active coil 方向三档 eps 均失败，relative error 约 `0.414-0.419`，cosine 约 `0.073`。
- `n3_mixed_rand20` 的 oracle-y0 基线失败，未进入灵敏度审计，分类为 `F1_baseline_unstable`。

基线修复扫描结果：

- 扫描对象：`n3_mixed_rand20`。
- 扫描参数：`continuity_mu={0.01,0.1,1.0,10.0}`，`max_iter={24,80,160}`。
- 结论：`not_recovered`。
- 最好的成功解为 `continuity_mu=0.1, max_iter=160`，`residual_norm≈0.00935`，但 `stationarity_norm≈78.54`，不能作为可线性化点。
- 最低 stationarity 解为 `continuity_mu=10.0, max_iter=160`，`stationarity_norm≈2.31e-4`，但 `residual_norm≈0.0272` 且未达 success 门限。
- 这说明软连续罚项存在明显权重拉扯：弱 continuity 可以压 residual，但 stationarity 差；强 continuity 可以压 stationarity，但任务 residual 变差。

## 数据分析

重点判据：

- 只有全部 eps 行通过、且 eps plateau 稳定时，一个方向才算通过。
- stationarity 是前置门禁；高 stationarity 不允许标记为 usable。
- condition number 过高时标记为 ill-conditioned，不允许直接进入 production gradient。
- re-optimized FD 仅作为审计，不作为 runtime 主梯度。

数据解释：

- `cont_1.0` 比 P2c7 的原始分支正则可靠很多，已经能在多数方向上得到与 re-optimized FD 一致的链式灵敏度。
- 但失败方向不是 eps 不稳定：`n3_mixed_base_i4/active_coil_5` 的 eps plateau 很稳定，只是 implicit/GN chain 与 re-optimized FD 系统性不一致。这更像 formulation 层面的缺失，而不是有限差分步长选择问题。
- `n3_mixed_rand20` 的基线无法同时满足 residual 与 stationarity，说明 hard current 工况下 soft penalty formulation 不足以提供稳定 KKT 点。
- seed gate 暴露出吸引域问题：`zero_y0` 有时几何接近但 stationarity 很高，`neighbor_y0` 多数会跳到坏分支或不可行分支，说明当前解分支仍依赖 oracle-y0。
- 当前结果不能进入 production IK，只能作为 diagnostic fallback 或局部 shadow replay 的候选。

## 实验结论

P2c9 严格门禁完成后，当前 shooting 灵敏度仍未达到长期目标。

可保留的阶段性成果：

- 建立了可复用 strict sensitivity audit harness。
- 完成了四个 P2c4 case 的 stationarity、rank、condition number、eps plateau、branch consistency、median rel_err、cosine、boundary residual 报告。
- 取得了部分可用的 `dg/dI`：`n2_z_pair_rand12`、`n1_tip_z_rand15` 和 `n3_mixed_base_i4` 的随机方向表现稳定。

不能升级的原因：

- `n3_mixed_base_i4` 的 active coil 方向稳定失败。
- `n3_mixed_rand20` 基线不可恢复。
- 当前 soft continuity penalty 仍是权重依赖的分支选择，不是真正的物理硬约束。

## 下一步计划

下一步必须进入 formulation 级修复：

- P2c9a：实现 multiple shooting hard continuity，把段间 wrench continuity 从软罚项改成等式约束，并基于 KKT/IFT 审计 `dg/dI`。
- P2c9b：并行实现 true continuous initial-value shooting，以 base wrench 6D 或 base wrench + 少量 slack/mode 作为变量，使用 continuation/multi-start 修复吸引域。
- 对 `n3_mixed_rand20` 优先做 target/current continuation，而不是继续调 `continuity_mu`。
- 在任何 production shadow 前，必须让 active coil 方向和 hard current 基线通过 stationarity 与 sensitivity gate。
