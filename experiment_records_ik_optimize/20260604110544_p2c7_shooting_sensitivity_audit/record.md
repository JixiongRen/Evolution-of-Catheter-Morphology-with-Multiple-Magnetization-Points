# P2c7 shooting 灵敏度内部链式审计

## 实验目的

本轮目标是建立可复用的 shooting-based IK 任务灵敏度审计工具。FK oracle 只作为弱对照，不作为唯一真值。本轮优先检查同一 shooting 模型内部的一致性：

$$
\Delta g_{\mathrm{reopt}} \approx g_I \Delta I + g_y \frac{dy^*}{dI}\Delta I
$$

## 实验方案

实验使用 P2c5 的 JAX segment-wrench shooting。任务向量为 terminal position 与 terminal SO(3)。内层使用固定 base terminal target，并加入轻量 branch-selection regularization：

$$
\min_y \frac{1}{2}\|r_{\mathrm{clean}}(y,I)\|^2 + \frac{\mu}{2}\|y-y_{\mathrm{ref}}\|^2
$$

这里的正则只用于定义可微分支，不能混入 clean terminal-only 结论。

## 记录文件

- `summary.json/csv`：方向审计结果。
- `sensitivity_audit.json`：机器可读审计总表。
- `heartbeat.jsonl`：低噪声进度。
- `failure_events.jsonl`：失败分类和纠偏建议。
- `solve_summary.csv`：base branch 求解质量。

## 运行命令

见 `commands.sh`。

## 当前状态

quick 审计已完成。当前结论不是 production usable，而是 diagnostic-only：harness 能运行并完成失败分类；clean implicit/GN chain 暂未解释 re-optimized FD；sampled Jacobian fallback 在已采样方向子空间可用作局部诊断。

## 结果

quick 模式运行两个 case：

- `n3_mixed_base_i4`
- `n2_z_pair_rand12`

每个 case 使用 3 个电流方向与 `eps={0.02,0.05,0.10} A`，共 18 条方向审计记录。

总体：

- classification：`diagnostic-only`
- usable count：`0`
- diagnostic-only count：`9`
- row count：`18`
- median rel_err_total：`0.497`
- total wall time：`390.30 s`

`n3_mixed_base_i4` 的所有方向均为 `F5_sensitivity_mismatch`：

- rel_err_total 约 `0.994-1.002`
- frozen partial norm 为 `0`
- implicit chain norm 约 `1e-6`
- re-optimized FD norm 约 `1.9e-4` 到 `2.0e-3`
- rank 为 `18`
- condition number 约 `3.27e6`
- stationarity_norm 约 `5.28`

`n2_z_pair_rand12` 的所有方向被标记为 `diagnostic-only`：

- re-optimized FD norm 为 `0`
- implicit total norm 为 `0`
- branch jump 为 `False`
- stationarity_norm 约 `1.01`

sampled Jacobian fallback：

- `n3_mixed_base_i4` 在已采样 3 个方向子空间内可精确回放 FD 样本，sampled_subspace_rel_err 约 `1e-16`。
- 该结果只说明 sampled fallback 可作为局部诊断或兜底，不是完整 8 维 production gradient。

## 数据分析

本轮给出三个重要诊断。

第一，terminal pose 对电流的 frozen partial 为零是合理的。当前 shooting forward 中电流通过刚段磁力/力矩影响内力/内矩状态，而刚段位姿传播本身不直接让 terminal pose 对电流显式变化。因此任务灵敏度主要来自 `dy*/dI`。

第二，当前 branch-conditioned clean objective 的 implicit/GN chain term 明显过小，无法解释 re-optimized FD。`n3_mixed_base_i4` 的 FD 响应在 `1e-4-1e-3` 量级，而 chain term 在 `1e-6` 量级。这说明当前 stationarity/GN 近似没有抓住 solver path，或者当前内层分支选择仍不够可微。

第三，sampled Jacobian fallback 可以作为工程兜底，但只能覆盖已采样方向子空间。它不解决物理解释问题，也不能替代可审计 implicit sensitivity。

## 实验结论

P2c7 初版完成了 audit harness 和失败纠偏记录，但未拿到 usable sensitivity。

当前状态应标记为：

```text
diagnostic-only: clean shooting internal chain audit runs, but implicit/GN chain is not yet usable.
```

主要失败类型：

- `F5_sensitivity_mismatch`
- 隐式链式项与 re-optimized FD 不一致
- stationarity_norm 偏高，说明当前正则化 clean branch 不是足够好的 KKT 点

不应进入 production 双层 IK，也不应把该 implicit chain 接入外层梯度。

## 下一步计划

下一轮纠偏建议：

1. 改进 branch-selection objective。尝试更强或更物理的非 oracle 正则，例如段间 wrench continuity diagnostic/penalty，避免只靠 `||y-y_ref||^2`。
2. 将 stationarity 作为主门禁重新优化，避免 residual 小但 KKT 不稳。
3. 增加 sampled Jacobian fallback runner，用更少方向做局部 trust-region step 诊断。
4. 若 clean terminal-only 仍不稳定，再切 multipoint-position hold-out，而不是直接上 full pose multipoint。
5. full 四工况审计应在上述纠偏后再跑，避免重复消耗。
