# P2c6 多磁体位姿目标 shooting 可行性验证

## 实验目的

本轮实验基于 P2c5 JAX clean shooting，验证“末端磁体完整位姿 + 上一段或多段磁体位置/轴向目标”是否能稳定、快速、物理可信地求解。

## 实验方案

本轮严格区分三类 residual：

- clean terminal-only baseline：末端位姿 + boundary。
- multipoint-position：末端位姿 + active magnet midpoint position + boundary。
- multipoint-axis：末端位姿 + active magnet midpoint position + previous active magnet axis + boundary。

禁止把 centerline target、smoothness 或刚段内部代理采样点作为监督残差。

## 实验过程记录

1. 扩展 `inverse_kinematics/ik_terminal_shooting_jax.py`，新增 fixed-shape multisite residual（固定形状多点残差）。
2. 新建 `run_multipoint_magnet_pose_probe.py`，运行 P0/P1/P2/P3。
3. 新建 `run_neighbor_warmstart_probe.py`，对 hard current case 做有限 neighbor warm start（邻近工况热启动）检查。
4. 所有命令均设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

## 实验结果

分类结论：

```text
diagnostic-only: position+axis feasible with zero-y0 basin failures
```

P0 语义 gate 全部通过。JAX shooting 与旧 Python shooting 在 terminal 和 rigid midpoint 上的差异均约 `1e-16 m` 量级，说明本轮内部 site convention（观测点约定）可用于 multipoint probe。

主实验：

| case | bundle | init | ok | steady_s | pos_mm | so3_deg | rigid_rms_mm | axis_deg | site_so3_deg | boundary |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n3_mixed_base_i4 | p1_position | oracle_y0 | True | 2.87 | 0.016 | 0.822 | 0.112 | 0.000 | 0.000 | 0.010 |
| n3_mixed_base_i4 | p1_position | zero_y0 | True | 4.50 | 0.235 | 0.892 | 0.932 | 0.000 | 0.000 | 0.047 |
| n3_mixed_rand20 | p1_position | oracle_y0 | True | 4.08 | 0.349 | 0.823 | 0.235 | 0.000 | 0.000 | 0.021 |
| n3_mixed_rand20 | p1_position | zero_y0 | False | 58.66 | 1.350 | 99.446 | 1.982 | 0.000 | 0.000 | 1.643 |
| n2_z_pair_rand12 | p1_position | oracle_y0 | True | 2.07 | 0.074 | 0.295 | 0.086 | 0.000 | 0.000 | 0.011 |
| n2_z_pair_rand12 | p1_position | zero_y0 | False | 41.08 | 0.012 | 1.871 | 0.061 | 0.000 | 0.000 | 0.040 |
| n1_tip_z_rand15 | p1_position | oracle_y0 | True | 1.74 | 0.059 | 0.273 | 0.062 | 0.000 | 0.000 | 0.013 |
| n1_tip_z_rand15 | p1_position | zero_y0 | True | 5.55 | 0.022 | 1.757 | 0.063 | 0.000 | 0.000 | 0.053 |
| n3_mixed_base_i4 | p2_position_axis | oracle_y0 | True | 2.09 | 0.016 | 0.822 | 0.112 | 0.821 | 0.000 | 0.010 |
| n3_mixed_base_i4 | p2_position_axis | zero_y0 | True | 4.47 | 0.235 | 0.892 | 0.932 | 2.302 | 0.000 | 0.047 |
| n3_mixed_rand20 | p2_position_axis | oracle_y0 | True | 3.95 | 0.349 | 0.823 | 0.235 | 3.195 | 0.000 | 0.021 |
| n3_mixed_rand20 | p2_position_axis | zero_y0 | False | 58.22 | 1.354 | 99.169 | 1.981 | 17.636 | 0.000 | 1.641 |
| n2_z_pair_rand12 | p2_position_axis | oracle_y0 | True | 2.01 | 0.074 | 0.295 | 0.086 | 0.200 | 0.000 | 0.011 |
| n2_z_pair_rand12 | p2_position_axis | zero_y0 | False | 40.39 | 0.090 | 1.907 | 0.079 | 3.294 | 0.000 | 0.045 |
| n1_tip_z_rand15 | p2_position_axis | oracle_y0 | True | 1.85 | 0.059 | 0.273 | 0.062 | 0.183 | 0.000 | 0.013 |
| n1_tip_z_rand15 | p2_position_axis | zero_y0 | True | 5.87 | 0.022 | 1.757 | 0.063 | 1.722 | 0.000 | 0.053 |

P3 optional full-pose stress：

| case | init | ok | steady_s | pos_mm | so3_deg | rigid_rms_mm | site_so3_deg | boundary |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| n3_mixed_base_i4 | oracle_y0 | True | 2.15 | 0.016 | 0.822 | 0.112 | 1.083 | 0.010 |
| n3_mixed_base_i4 | zero_y0 | True | 4.23 | 0.235 | 0.892 | 0.932 | 2.995 | 0.047 |
| n3_mixed_rand20 | oracle_y0 | True | 4.86 | 0.156 | 0.385 | 0.102 | 1.660 | 0.028 |
| n3_mixed_rand20 | zero_y0 | False | 58.57 | 1.354 | 99.178 | 1.981 | 17.718 | 1.641 |

Neighbor warm start：

| case | bundle | init | ok | steady_s | pos_mm | so3_deg | rigid_rms_mm | axis_deg | site_so3_deg | boundary |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n3_mixed_rand20 | p1_position | neighbor_base_zero_y0 | False | 49.71 | 0.006 | 0.150 | 0.005 | 0.000 | 0.000 | 0.206 |
| n3_mixed_rand20 | p2_position_axis | neighbor_base_zero_y0 | False | 49.23 | 0.022 | 2.034 | 0.038 | 7.364 | 0.000 | 0.168 |
| n3_mixed_rand20 | p3_previous_full_pose | neighbor_base_zero_y0 | False | 50.79 | 0.011 | 1.359 | 0.025 | 0.000 | 10.743 | 0.088 |

## 数据分析

P1/P2 均为 6/8 通过。通过组的 steady solve time 多数在 `1.7-5.9 s`，与 P2c5 同量级。说明 multipoint residual 本身可以在当前 JAX shooting 框架中快速求解。

失败集中在两个方向：

- `n3_mixed_rand20/zero_y0`：末端姿态、rigid 位置、边界均失败，属于 hard current 下的 initial basin（初值吸引域）失败。
- `n2_z_pair_rand12/zero_y0`：几何指标基本通过，但 steady solve time 超过 `30 s`，属于性能门限失败。

neighbor warm start 显示 `n3_mixed_rand20` 的位置约束并非不可达：P1 下位置误差降到 `0.006 mm`，rigid RMS 降到 `0.005 mm`，但 boundary 为 `0.206`，略高于 `0.2`，且耗时约 `50 s`。这说明失败主因不是目标几何不可达，而是 initial basin、boundary 权衡和 conditioning（条件数）问题。

P3 完整姿态压力测试只在 oracle_y0 和较温和 zero_y0 下稳定，不适合直接作为主线。

## 实验结论

本分支研究可行，但当前不能标记为 production-candidate（生产候选）。

更准确结论：

```text
diagnostic-only: multipoint position/axis shooting is feasible, but zero-y0 basin failures remain.
```

可保留的候选：

- `terminal full pose + active magnet midpoint position + boundary`
- `terminal full pose + active magnet midpoint position + previous active magnet axis + boundary`

不建议当前推进为主线：

- `terminal full pose + previous magnet full SO(3) + boundary`

失败来源判断：

- site convention：已通过 P0，不是主因。
- rank/conditioning：存在高条件数，是风险之一。
- boundary 退化：hard current neighbor warm start 中出现轻微边界超限。
- initial basin：主因，zero_y0 在 hard current 下失败。
- 模型自由度不足：暂未证明，oracle_y0 多数通过说明不是首要瓶颈。

## 下一步计划

建议回到主线前保留两个改进方向：

1. 做 better initialization（更好初值）：current continuation（电流延续）、neighbor warm start（邻近工况热启动）、或轻量 NN 初值估计。
2. 若继续 multipoint 分支，优先补段间 wrench continuity 诊断，不要直接增加更多完整位姿约束。

主线 IK 仍应继续推进；P2c6 结果作为可行性证据和诊断分支保留。
