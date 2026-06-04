# P2c shooting IK 灵敏度可视化说明

## 图像列表

| 文件 | 内容 |
|---|---|
| `p2c_sensitivity_dashboard.png` | 总览图，汇总 exact IFT、shadow replay、branch gate 和 continuation 断裂点。 |
| `p2c9f_exact_vs_gn_rel_error.png` | P2c9f 中 exact stationarity IFT 与 GN chain 的相对误差对比。 |
| `p2c9g_shadow_replay.png` | P2c9g 中 exact IFT 对小电流扰动的 predicted/actual shadow replay 效果。 |
| `p2c9i_branch_roundtrip.png` | P2c9i 中多初值分支一致性和 `I -> I+dI -> I` 回环误差。 |
| `p2c9j_continuation_break.png` | P2c9j 中从 stable case 过渡到 hard case 时 residual、boundary 和 stationarity 的变化。 |

## 核心解读

### 1. exact IFT 明显优于 GN chain

`p2c9f_exact_vs_gn_rel_error.png` 显示 exact stationarity IFT 的 median relative error 明显低于 GN chain。尤其是之前 active-coil 方向的误差，exact IFT 已进入可用区间。

结论：后续不应再以 GN chain 作为主灵敏度路线。

### 2. stable case 的 shadow replay 通过

`p2c9g_shadow_replay.png` 显示 3 个 stable case 的 6 个方向全部通过：

- median relative error 约 `0.032`
- cosine 约 `0.9999`

结论：对 stable branch，exact IFT 不只是离线 FD 审计好，也能预测小电流扰动后的实际 task 变化。

### 3. stable case 的分支可跟踪

`p2c9i_branch_roundtrip.png` 显示 3 个 stable case 的多初值 branch gate 与 active-coil roundtrip gate 均通过。`n3_mixed_rand20` 不通过是因为 baseline 本身不稳定，不是因为 shadow replay 单独失败。

结论：stable set 的 `dg/dI` 可以标记为 shadow-candidate，但仍不应直接接 production IK。

### 4. hard case 的问题是线性化断裂

`p2c9j_continuation_break.png` 显示从 `n3_mixed_base_i4` 过渡到 `n3_mixed_rand20` 时：

- `alpha=0.0` 和 `alpha=0.2` 可线性化；
- `alpha=0.4` 开始 stationarity 明显失控；
- `alpha=1.0` 才出现 residual 意义上的最终失败。

结论：hard case 的失败不是突然的求解崩溃，而是先发生 stationarity gate 失效。后续应围绕 `alpha=0.2 -> 0.4` 做局部结构诊断。

## 下一步建议

1. 保留 exact stationarity IFT 作为 stable branch 的主灵敏度候选。
2. 不发布 `n3_mixed_rand20` 的 `dg/dI`，继续保持 `F1_baseline_unstable / linearization-break` 分类。
3. 下一轮优先做 `alpha=0.2,0.3,0.4` 的 residual block decomposition。
4. 若 stationarity 主要来自 continuity 与 terminal residual 冲突，再进入 hard-continuity multiple shooting / KKT formulation。

