# P2c9e continuation 与刚段耦合审计记录

## 实验目的

P2c9d hard-continuity KKT 在核心失败 case 上 baseline 不稳定。本轮实验不再继续盲试新优化器，而是做两件事：

- soft-to-hard continuation：观察 continuity 权重逐步增强时 residual、boundary、continuity、stationarity 如何演化。
- 刚段载荷耦合审计：确认磁力/磁矩是否进入刚段 pose propagation，从而判断 active coil 灵敏度错配是否可能来自模型显式路径缺失。

## 实验方案

覆盖 case：

- `n3_mixed_base_i4`
- `n3_mixed_rand20`

continuation 权重：

- `0, 1e-4, 1e-2, 1e0, 1e2`

记录指标：

- terminal/boundary residual。
- continuity residual。
- stationarity。
- active coil fixed-y explicit task response。
- 刚段传播函数中外载对 pose 的依赖关系。

## 实验过程记录

运行命令见 `commands.sh`。输出包括：

- `summary.json`
- `continuation_summary.csv`
- `explicit_response_summary.csv`
- `rigid_coupling_audit.json`
- `heartbeat.jsonl`
- `failure_events.jsonl`

## 实验结果

总体分类：`model-coupling-suspect`。

continuation 结果：

- `n3_mixed_base_i4` 在 `continuity_mu=1.0` 时可达较好状态：`residual_norm≈0.00161`，`stationarity_norm≈3.13e-7`，`boundary≈5.66e-4`，`continuity_norm≈0.00146`。
- `n3_mixed_base_i4` 在 `continuity_mu=100` 时 residual 变差到 `0.0145`，说明继续推向更硬 continuity 会牺牲任务残差。
- `n3_mixed_rand20` 全部 continuation stage 均未通过 baseline；最佳 stationarity 出现在 `continuity_mu=1e-4`，但 `residual_norm≈0.0177`，`stationarity_norm≈50.6`。
- `n3_mixed_rand20` 随 continuity 增大时 continuity residual 下降，但 terminal/boundary/stationarity 不能同时达标。

fixed-y 显式响应：

- 两个 case、所有 continuity stage 下，active coil 的 fixed-y terminal task response norm 均为 `0.0`。
- 这说明在当前 shooting forward 中，电流变化对 terminal task 没有显式路径，全部灵敏度都被迫通过 `dy*/dI` 体现。

刚段耦合审计：

- `rigid_state_along_dim` 中 `p_sigma = pp + sigma * (R_p @ rigid.v_star)`。
- `Q_sigma = Qp`。
- `f_ext_total_dim` 和 `tau_ext_total_dim` 不进入刚段 pose update。
- 当前实现中刚段外载只更新 distal force/moment，不直接更新刚段位置/姿态。

## 数据分析

重点判断：

- hard continuity 是否可以通过 continuation 平滑接近。
- fixed-y 下 active coil 是否有显式 terminal task 响应。
- 刚段外载是否直接影响刚段姿态传播。

关键判断：

- P2c9/P2c9c/P2c9d 的 active coil mismatch 不是有限差分步长问题，也不只是分支选择问题；P2c9e 证明 fixed-y 显式任务响应恒为零。
- 对于磁控导管，电流对磁体刚段的力/矩如果不直接改变刚段 pose，IK sensitivity 会退化为“优化变量如何重新分配 wrench”的间接响应。这会放大 branch selection 与正则权重的影响。
- `n3_mixed_rand20` 的 hard case 无法通过 continuation 恢复，说明当前 formulation 可能缺少关键物理自由度或耦合项。
- 当前更合理的失败分类是 `model_coupling_suspect`，而不是继续归为单纯 `F5_sensitivity_mismatch`。

## 实验结论

P2c9e 没有拿到最终可用灵敏度，但建立了更强的纠偏证据：当前 shooting-based sensitivity 的核心瓶颈很可能是刚段载荷到 pose propagation 的显式耦合缺失或不足。

因此，继续在现有 shooting residual 上调整 soft/hard continuity、slack、KKT 初值，预期收益有限。下一步应优先修正或替代刚段传播模型，使磁力/磁矩能够通过 pose update 进入 terminal task，再重新审计 `dg/dI`。

## 下一步计划

下一步建议推进 P2c10：

- 建立 rigid-load pose coupling 修复原型。
- 对比旧刚段传播与新刚段传播下的 FK/shooting terminal pose。
- 重新运行 P2c9 strict sensitivity gate，优先看 active coil fixed-y explicit response 是否变为非零。
- 若修复后 shooting 内部链式审计改善，再考虑 production shadow replay。
