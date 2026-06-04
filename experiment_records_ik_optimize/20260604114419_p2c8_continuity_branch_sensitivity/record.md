# P2c8 段间连续性分支灵敏度审计记录

## 实验目的

本轮实验继续排查 P2c7 中 shooting 灵敏度不稳定的问题。核心目标不是证明某个 FK oracle 完全正确，而是检查 clean shooting 自身在电流扰动下是否形成稳定、可微、物理合理的解分支。

## 实验方案

在 P2c7 的 `reg_only` 分支正则基础上，新增段间力/矩连续性残差。残差定义为：第 `i` 个柔性段和刚段积分后的 distal wrench，与第 `i+1` 段 shooting block 给出的 proximal wrench 之间的差值，统一用无量纲内力/内矩单位表示。

本轮对比模式：

- `reg_only`：P2c7 原始分支正则。
- `cont_0.01`：弱段间连续性约束。
- `cont_0.1`：中等段间连续性约束。
- `cont_1.0`：强段间连续性约束。

## 实验过程记录

运行命令见 `commands.sh`。实验输出包括 `summary.json`、`summary.csv`、`solve_summary.csv`、`heartbeat.jsonl`、`failure_events.jsonl` 和 `sensitivity_audit.json`。

## 实验结果

本轮有效复跑覆盖两个 case、两个 continuity 权重和每个 case 两个电流扰动方向。总计 8 行灵敏度审计，其中 4 行达到 `usable`，4 行仍为 `untrusted` 或 `nonstationary-diagnostic`，总体分类修正为 `partially-usable`。

关键结果：

- `cont_0.1` 在 `n3_mixed_base_i4` 的随机方向可用，相对误差约 `0.097`；但 active coil 方向相对误差约 `0.348`，仍不可信。
- `cont_1.0` 是当前最好模式，median relative error 约 `0.048`。
- `cont_1.0` 在 `n3_mixed_base_i4` 的随机方向相对误差约 `0.016`，在 `n2_z_pair_rand12` 两个方向相对误差约 `0.056` 和 `0.041`。
- `cont_1.0` 在 `n3_mixed_base_i4` 的 active coil 方向仍失败，相对误差约 `0.415`。
- `cont_0.1` 在 `n2_z_pair_rand12` 出现 `stationarity_norm≈1.69`，应归类为非平稳分支诊断，不应算通过。
- `cont_1.0` 将两个 case 的 stationarity 压到约 `1e-8` 或更低，说明段间连续性约束确实显著改善了 shooting 内层最优性。

运行时间方面，单个 branch solve 的 steady time 约 `0.16-0.37 s`，本轮主要耗时来自 JAX 编译和多模式审计。

## 数据分析

重点观察：

- `stationarity_norm` 是否显著下降。
- `rel_err_total` 是否下降。
- implicit/GN chain 与 re-optimized finite difference 是否同向。
- 强连续性是否导致 terminal pose 或 boundary residual 退化。

分析结论：

- P2c7 中的原始 `reg_only` 失败不是单纯 FK oracle 偏差导致，而是 shooting 参数化本身存在分支和最优性问题。
- 段间 wrench continuity 是有效的病因定位工具：加入后 stationarity 明显下降，部分方向上的 implicit sensitivity 与 re-optimized finite difference 明显靠拢。
- 但是 soft continuity penalty 仍然是权重依赖的正则化分支，不应直接宣称为最终物理灵敏度。
- active coil 方向仍失败，说明当前 formulation 仍存在方向相关的线性化误差、尺度问题或分支不可观测问题。
- 当前结果支持继续推进“硬连续 multiple shooting + KKT/IFT”或“真连续初值打靶”路线，而不是继续在原始 18D 重置式 shooting 上堆更多几何项。

## 实验结论

P2c8 达成病因定位目标，但未达成长期灵敏度终局目标。

本轮可以确认：段间力/矩连续性是影响 shooting 灵敏度质量的关键因素之一。`cont_1.0` 已经把多个方向的灵敏度相对误差压到 `0.02-0.06` 量级，但仍存在 active coil 方向不通过，因此当前只能标记为 `partially-usable`，不能接入外层 IK 作为生产灵敏度。

## 下一步计划

下一步建议从两个方向并行推进，但主线应更偏物理：

- P2c9a：把 soft continuity 从诊断罚项升级为 multiple shooting 硬等式约束，并建立 KKT/IFT 灵敏度审计。
- P2c9b：实现真连续初值打靶，只优化基座内力/内矩或少量低阶 mode，用 continuation 和 multi-start 解决吸引域问题。
- 保留 `cont_1.0` 作为短期诊断 fallback，但不作为最终 production gradient。
- 后续审计必须把 stationarity 作为前置门禁，并继续报告 `branch_jump`、条件数、epsilon 一致性和 re-optimized finite difference。
