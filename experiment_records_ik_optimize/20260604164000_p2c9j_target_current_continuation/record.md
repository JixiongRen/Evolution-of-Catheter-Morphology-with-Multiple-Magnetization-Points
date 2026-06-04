# P2c9j target/current continuation 实验记录

## 实验目的

P2c9h/P2c9i 表明 `n3_mixed_rand20` 不是灵敏度公式失败，而是 baseline/branch/formulation 问题。本轮实验进一步定位问题出现的位置：从稳定的 `n3_mixed_base_i4` 连续过渡到 hard case `n3_mixed_rand20`，观察 shooting baseline 何时失去可线性化性。

## 实验方案

两个端点使用相同三磁体配置：

- base：`n3_mixed_base_i4`
- target：`n3_mixed_rand20`

电流按 alpha 线性插值：

```text
I(alpha) = (1 - alpha) I_base + alpha I_target
```

alpha 取值：

```text
0, 0.2, 0.4, 0.6, 0.8, 1.0
```

每个 alpha 都重新运行 FK oracle 生成对应 terminal target，但 FK 只作为 target generator，不作为灵敏度主裁判。每个 alpha 的 shooting 初值包括：

- `oracle`
- `continuation`
- `half_oracle`

其中 `continuation` 使用上一个 alpha 的成功 shooting 状态。

## 实验过程记录

运行命令见 `commands.sh`。实验显式设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

本轮不修改 production IK，也不发布 `n3_mixed_rand20` 的梯度。

## 实验结果

总体分类：`linearization-break`。

关键结论：

- 按旧 residual/boundary success，路径到 `alpha=0.8` 仍可找到拟合解，`alpha=1.0` 失败。
- 按真正用于灵敏度的可线性化门禁，`alpha=0.4` 已经断裂，因为 stationarity 明显升高。

| alpha | any success | any linearizable | best init | residual | stationarity | boundary |
|---:|---|---|---|---:|---:|---:|
| 0.0 | true | true | oracle | 0.00161 | 3.60e-10 | 0.00057 |
| 0.2 | true | true | oracle | 0.00188 | 7.80e-06 | 0.00271 |
| 0.4 | true | false | oracle | 0.00241 | 28.32 | 0.00287 |
| 0.6 | true | false | half_oracle | 0.00700 | 216.91 | 0.00694 |
| 0.8 | true | false | oracle | 0.00435 | 12.77 | 0.00357 |
| 1.0 | false | false | half_oracle | 0.03599 | 25.64 | 0.05748 |

失败事件：

- `first_linearization_failure_alpha = 0.4`
- `first_failure_alpha = 1.0`

## 数据分析

本轮最重要的发现是：hard case 的问题不是到 `alpha=1.0` 才突然出现。早在 `alpha=0.4`，虽然 terminal residual 仍然很小，但 stationarity 已经从 `1e-5` 量级跃升到 `28`。这意味着该点已经不适合做 IFT 线性化。

因此后续不能只盯着 residual 或 terminal pose 拟合。对于灵敏度，stationarity 是前置门禁。若 stationarity 失控，即使末端姿态看起来接近，也不能把该点的 `dg/dI` 接入外层 IK。

`alpha=0.4` 到 `0.8` 的现象表明当前 soft-continuity formulation 可以继续拟合终端目标，但其选择的状态分支已经不是可稳定线性化的物理解分支。这和 P2c9h 的结果一致：普通权重扫描或增加迭代次数很难真正恢复 hard case。

## 实验结论

1. `n3_mixed_rand20` 的 hard failure 是逐渐积累的分支/可线性化断裂，不只是最终点求解失败。
2. 对当前 formulation，`alpha=0.4` 是最早可观察到的 stationarity gate 断裂点。
3. 下一步应围绕 `alpha=0.2 -> 0.4` 做局部结构诊断，而不是继续扩大 `alpha=1.0` 的普通扫描。
4. stable set 的 exact IFT 可用结论不受影响，但 hard path 需要 formulation 级纠偏。

## 下一步计划

建议下一轮聚焦 `alpha=0.2, 0.3, 0.4`：

1. 对该区间做更细 alpha 网格，定位 stationarity 从可用到失控的阈值。
2. 审查 residual block 贡献，找出 stationarity 主要来自 terminal、regularization 还是 continuity。
3. 对断裂点执行 rigid-load coupling audit，判断刚段外载显式耦合缺失是否导致全部响应被压到 `dy*/dI`。
4. 若 residual block 显示 continuity 与 terminal 互相冲突，则进入 hard-continuity multiple shooting / KKT formulation。

