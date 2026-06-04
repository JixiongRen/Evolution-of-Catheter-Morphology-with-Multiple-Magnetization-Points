# P2c9h n3_mixed_rand20 baseline recovery 实验记录

## 实验目的

P2c9g 已证明 exact stationarity IFT 在稳定 baseline 上可以通过 shadow replay，但 `n3_mixed_rand20` 仍触发 `F1_baseline_unstable`。本轮实验专门对该 hard case 做 baseline recovery，目标是在不发布坏灵敏度的前提下，尝试恢复一个可线性化的 shooting baseline。

## 实验方案

本轮只处理：

- case：`n3_mixed_rand20`
- neighbor case：`n3_mixed_base_i4`

扫描策略：

- 初值：
  - `oracle`
  - `zero`
  - `half_oracle`
  - `neighbor`
- continuity 权重：
  - `0.3`
  - `1.0`
  - `3.0`
  - `10.0`
- regularization 权重：
  - `0.0`
  - `1.0e-6`
  - `1.0e-4`
- scan max iter：`16`
- refine max iter：`48`
- top-k refine：`3`

评分函数：

```text
score = residual_norm + 0.02 * boundary_residual_norm + 1e-4 * min(stationarity_norm, 1e4)
```

该评分不是成功判据，只用于从扫描结果里挑选候选精修。成功恢复仍要求 baseline residual、boundary 和 stationarity 同时处于可线性化范围。

## 实验过程记录

运行命令见 `commands.sh`。实验显式设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

实验过程中没有修改 production IK 或 FK 代码，只新增本轮实验脚本和结果文件。

## 实验结果

总体分类：`not-recovered`。

运行统计：

| 指标 | 数值 |
|---|---:|
| row_count | 51 |
| total wall time | 1643.52 s |

最佳扫描点：

| 指标 | 数值 |
|---|---:|
| phase | scan |
| continuity_mu | 0.3 |
| reg_mu | 0.0 |
| init | half_oracle |
| residual_norm | 0.02893 |
| stationarity_norm | 62.55 |
| boundary_residual_norm | 0.04103 |
| success | false |

最佳精修点虽然将 residual 进一步降到约 `0.01795`，但 stationarity 上升到约 `270.84`，因此不应视为恢复成功。

## 数据分析

本轮结果说明 `n3_mixed_rand20` 的问题不只是迭代次数不足。

观察到的典型模式：

- 降低 continuity 权重可以降低 residual，但 stationarity 仍然很高。
- 精修可以继续压低 residual，但 stationarity 反而变差。
- `half_oracle` 比 `oracle / zero / neighbor` 更容易接近较低 residual，但仍无法进入可线性化 baseline。

这说明当前 formulation 可能在 hard current case 上存在以下问题之一：

1. residual landscape 有明显 branch/basin 问题；
2. soft-continuity 正则正在选择一个数学上可拟合但 stationarity 不可信的分支；
3. 当前 shooting state 对该 case 的物理自由度或刚段载荷耦合表达不足；
4. FK oracle 提供的 terminal target 可能处在当前 shooting formulation 的困难区域，但这不能简单解释为 FK 真值错误。

## 实验结论

`n3_mixed_rand20` 仍保持 `F1_baseline_unstable` 分类。

因此该 case 不能发布 `dg/dI`，也不能进入 shadow replay 或 production IK。P2c9g 的 stable-case usable 结论仍成立，但 hard case 需要进入下一层纠偏。

## 下一步计划

建议下一轮不再继续扩大普通权重扫描，而是切换到结构性诊断：

1. 做 branch tracking：保留 incumbent、challenger、cold_ref 三分支，不让 warm start 覆盖潜在分支。
2. 做 target continuation：从较容易的邻近 current/target terminal pose 连续移动到 `n3_mixed_rand20`。
3. 审查 rigid-load to pose coupling：确认刚段外载是否显式影响 terminal pose propagation。
4. 若仍不能恢复，考虑 hard-continuity multiple shooting + exact KKT/IFT，但必须先解决吸引域和分支初始化。

