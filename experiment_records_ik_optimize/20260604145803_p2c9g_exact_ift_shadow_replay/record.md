# P2c9g exact IFT shadow replay 实验记录

## 实验目的

本轮实验继续推进 shooting-based IK 任务灵敏度 `dg/dI` 主线。重点不是把 FK oracle 当作唯一真值，而是验证 P2c9f 得到的 exact stationarity IFT 灵敏度能否在 shooting 内部产生可用的预测效果。

具体目标：

- 将 P2c9f 的 exact stationarity IFT 从“与 re-optimized FD 对比”推进到 shadow replay。
- 检查小电流步长 `I -> I + dI` 下，灵敏度预测的 terminal task 变化是否与重新 shooting 后的实际变化一致。
- 继续对 hard case 做失败分类，不因单个 hard baseline 失败而否定稳定分支上的局部灵敏度。

## 实验方案

本轮仍使用 P2c8/P2c9f 中表现最好的 soft-continuity clean shooting 设定：

- `continuity_mu = 1.0`
- `reg_mu = 1.0e-4`
- `max_iter = 24`
- `step = 0.05 A`
- `random_count = 1`

对每个 case，流程为：

1. 使用 FK oracle 仅生成 terminal pose 目标和 oracle 初值，不把 FK FD 作为主判据。
2. 求解 baseline shooting 状态 `y*(I)`。
3. 对 exact stationarity IFT 计算完整 `6 x 8` 任务灵敏度矩阵。
4. 对 active coil 方向和一个随机方向执行 shadow replay：
   - 使用 `dg/dI @ dI` 预测 terminal task 变化。
   - 重新求解 `I + dI` 下的 shooting 状态。
   - 对比 predicted delta 与 actual delta。

本轮任务向量为 terminal position + terminal SO(3)，不加入 centerline oracle 或刚段中点约束。

## 实验过程记录

运行命令见 `commands.sh`。实际执行命令显式设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

本轮初次试跑中发现 full 8-coil sensitivity matrix 的实现逐列重复计算 Hessian，导致第一 case 耗时异常。随后修正为一次计算 `H、G_I、T_y、T_I`，一次性求解完整矩阵。该修正只改变实验脚本的工程实现，不改变 exact IFT 数学定义。

另一个统计问题是旧 `_safe_rel` 使用 `1e-9` 作为 cosine denominator floor。由于本轮 shadow delta 的范数约为 `1e-7` 到 `1e-6`，该 floor 会把实际接近 `1.0` 的方向余弦压到接近 `0`。本轮已改为单独统计原始方向余弦，summary/csv 已按保存的 predicted/actual 向量重新后处理。

## 实验结果

总体分类：`partially-usable`。

原因是 3 个稳定 baseline 的 shadow replay 全部通过，但 `n3_mixed_rand20` 仍未通过 baseline root gate。

关键指标：

| 指标 | 数值 |
|---|---:|
| row_count | 6 |
| usable_count | 6 |
| failed_baseline_count | 1 |
| median rel_err | 0.0321 |
| median cosine | 0.9999 |

稳定 case 的结果：

| case | direction | rel_err | cosine | status |
|---|---|---:|---:|---|
| n3_mixed_base_i4 | active_coil_5 | 0.0289 | 0.9997 | usable |
| n3_mixed_base_i4 | random_0 | 0.0353 | 0.9994 | usable |
| n2_z_pair_rand12 | active_coil_2 | 0.0696 | 0.9999 | usable |
| n2_z_pair_rand12 | random_0 | 0.0682 | 0.9998 | usable |
| n1_tip_z_rand15 | active_coil_4 | 0.0087 | 1.0000 | usable |
| n1_tip_z_rand15 | random_0 | 0.0204 | 0.9999 | usable |

失败 case：

- `n3_mixed_rand20`
- failure class：`F1_baseline_unstable`
- residual norm：`0.07358`
- stationarity norm：`742.87`
- boundary residual norm：`0.05636`

## 数据分析

本轮是 P2c9f 之后的重要推进。P2c9f 已证明 exact stationarity IFT 在 re-optimized FD 审计中明显优于 GN chain；P2c9g 进一步说明，在 baseline 可线性化的 case 上，exact IFT 不只是离线审计指标更好，也能预测小电流扰动后的实际 terminal task 变化。

这说明当前最可信的灵敏度主线应继续采用：

```text
soft-continuity clean shooting + exact stationarity IFT + strict baseline/branch gate
```

但本轮不能标记为全局 usable，因为 `n3_mixed_rand20` 仍然不是可线性化的 baseline。这个失败不是 shadow replay 或 exact IFT 的失败，而是 baseline root gate 未通过。按照当前纠偏规则，该 case 应先进入 baseline recovery，而不是继续发布 `dg/dI`。

## 实验结论

1. 对稳定 baseline，exact stationarity IFT 的 shadow replay 表现很好，6/6 方向通过，median rel_err 约 `3.2%`。
2. FK oracle 只参与目标生成和弱对照，本轮主判据是 shooting 内部 predicted/actual 一致性。
3. `n3_mixed_rand20` 仍是 hard baseline 问题，应继续归类为 `F1_baseline_unstable`。
4. 当前 exact IFT 可作为 shadow gradient candidate，但还不应默认接入 production 双层 IK。

## 下一步计划

下一轮优先级：

1. 对 `n3_mixed_rand20` 进入 baseline recovery：multi-start、neighbor warm start、continuation、soft-to-hard schedule。
2. 对 stable cases 增加 branch consistency gate：`I -> I+dI -> I` 回环和多初值分支对比。
3. 若 hard case baseline 长期不恢复，审查刚段外载到 terminal pose propagation 的显式耦合。
4. 在 stable set 上增加一轮 production shadow：使用 exact `dg/dI` 执行外层一小步目标下降，统计 predicted/actual reduction。

