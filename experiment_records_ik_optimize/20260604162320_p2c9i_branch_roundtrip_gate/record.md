# P2c9i branch roundtrip gate 实验记录

## 实验目的

本轮实验补齐 shooting IK 灵敏度主线中的 branch gate。P2c9g 已显示 exact stationarity IFT 在稳定 baseline 上可通过 shadow replay，但还需要确认这些灵敏度确实位于稳定分支上，而不是由 warm start 偶然选到某个局部解。

本轮重点验证：

- 多初值是否收敛到一致的 terminal task / shooting state 分支；
- `I -> I + dI -> I` 回环后是否回到原 baseline；
- hard case 是否仍应归类为 baseline 不稳定，而不是灵敏度公式失败。

## 实验方案

实验配置：

- cases：
  - `n3_mixed_base_i4`
  - `n2_z_pair_rand12`
  - `n1_tip_z_rand15`
  - `n3_mixed_rand20`
- shooting formulation：soft-continuity clean shooting
- `continuity_mu = 1.0`
- `reg_mu = 1.0e-4`
- `max_iter = 24`
- active-coil roundtrip step：`0.05 A`

初值集合：

- `oracle`
- `zero`
- `half_oracle`
- `neighbor`

其中 `neighbor` 来自 `n3_mixed_base_i4` 的可用 shooting 解。FK oracle 仍只用于生成 terminal pose 目标和 oracle 初值，不作为 branch gate 的主裁判。

## 实验过程记录

运行命令见 `commands.sh`。实验显式设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

本轮不加入 centerline oracle，不修改 production IK，也不将结果接入默认双层 IK。

## 实验结果

总体分类：`partially-usable`。

原因是 3 个 stable case 通过 branch gate 和 roundtrip gate，但 `n3_mixed_rand20` 仍无法得到可用 baseline。

| 指标 | 数值 |
|---|---:|
| case_count | 4 |
| branch_consistent_count | 3 |
| roundtrip_count | 3 |
| roundtrip_consistent_count | 3 |

case 级结果：

| case | branch classification | successful init count | branch task spread | branch state spread |
|---|---|---:|---:|---:|
| n3_mixed_base_i4 | branch-consistent | 2/4 | 0.0 | 0.0 |
| n2_z_pair_rand12 | branch-consistent | 1/4 | 0.0 | 0.0 |
| n1_tip_z_rand15 | branch-consistent | 3/4 | 8.69e-4 | 0.136 |
| n3_mixed_rand20 | baseline-unstable | 0/4 | 0.0 | 0.0 |

roundtrip 结果：

| case | state error | task error | result |
|---|---:|---:|---|
| n3_mixed_base_i4 | 4.87e-10 | 5.55e-13 | pass |
| n2_z_pair_rand12 | 5.72e-08 | 1.00e-11 | pass |
| n1_tip_z_rand15 | 8.90e-10 | 3.16e-14 | pass |

失败事件：

- `n3_mixed_rand20`
- failure class：`F1_baseline_unstable`
- evidence：`oracle_success=false`，`successful_init_count=0/4`

## 数据分析

P2c9i 支持 P2c9g 的结论：stable set 上的 exact IFT 灵敏度不是单纯 warm-start 偶然性。三个 stable case 的 active-coil 回环误差都非常小，说明在 `0.05 A` 小扰动范围内，当前 shooting 解分支是可跟踪的。

但 `n3_mixed_rand20` 依旧没有进入可线性化区域。结合 P2c9h 的恢复扫描结果，该 case 不应继续通过普通多初值或权重扫描硬凑灵敏度。它需要结构性纠偏。

## 实验结论

1. stable cases 已通过 branch consistency 和 roundtrip consistency。
2. exact IFT 在 stable cases 上可以继续作为 shadow gradient candidate。
3. `n3_mixed_rand20` 继续归类为 `F1_baseline_unstable`，不能发布 `dg/dI`。
4. 当前失败不是 FK oracle mismatch 的充分证据，也不是 exact IFT 公式失败；更可能是 hard baseline / branch / formulation 问题。

## 下一步计划

下一步应从普通扫描切换到结构性诊断：

1. target continuation：从 `n3_mixed_base_i4` 连续移动到 `n3_mixed_rand20`，观察分支何处断裂。
2. branch tracking：保留 incumbent、challenger、cold_ref 三条分支，不让 warm start 覆盖分支信息。
3. rigid-load coupling audit：审查刚段外载是否显式进入 terminal pose propagation。
4. 若结构性诊断仍失败，再考虑 hard-continuity multiple shooting + exact KKT/IFT。

