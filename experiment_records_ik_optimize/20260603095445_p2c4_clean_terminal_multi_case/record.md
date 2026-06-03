# P2c4 Clean Terminal Pose Multi-Case 实验记录

## 实验目的

本轮扩展 P2c3 的 clean shooting 验证，测试不同磁化段数量、磁化模式和多线圈电流组合下，只约束末端位姿和物理边界时，打靶 IK 是否还能收敛到参考 FK 的末端位姿。

本轮 residual 只包含：

- terminal position residual
- terminal SO(3) residual
- physical boundary residual

明确不包含：

- centerline residual
- rigid midpoint residual

额外加入两种初值：

- `oracle_y0`：从 FK oracle 内力/内矩提取初值，用于对比历史实验。
- `zero_y0`：零内力/内矩初值，更接近无 oracle 初值的干净测试。

## 实验方案

固定设置：

- `M_list=(1,1,1)`
- `tol=1e-5`
- `max_iter=120`
- `candidate_max_nfev=16`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

测试工况：

- `n3_mixed_base_i4`：三段磁体，基准单主线圈电流。
- `n3_mixed_rand20`：三段磁体，多线圈随机电流，单线圈不超过 20A。
- `n2_z_pair_rand12`：两段 z 向反向磁体，第三段无磁化，多线圈随机电流。
- `n1_tip_z_rand15`：仅末端一段 z 向磁体，多线圈随机电流。

每个工况均运行 `oracle_y0` 和 `zero_y0`。

## 实验过程记录

运行命令见 `commands.sh`。

总墙钟时间：

- `4516.62s`
- 约 `75.3min`

输出文件：

- `summary.json`
- `summary.csv`
- `terminal_error_summary.png`
- 每个 case 下的 `oracle_vs_clean_shooting.png`
- 每个 case 下的 oracle / shooting centerline 与 shooting state

## 实验结果

全部 8 组均达到 Stage B。核心指标如下：

| case | init | pos err mm | SO3 err deg | centerline RMS mm | boundary | wall s |
|---|---|---:|---:|---:|---:|---:|
| n3_mixed_base_i4 | oracle_y0 | 0.058 | 0.0236 | 13.34 | 0.0900 | 330.10 |
| n3_mixed_base_i4 | zero_y0 | 0.0059 | 0.1144 | 13.41 | 0.2995 | 516.61 |
| n3_mixed_rand20 | oracle_y0 | 0.0033 | 0.0042 | 13.57 | 0.0732 | 551.58 |
| n3_mixed_rand20 | zero_y0 | 1.254 | 1.4084 | 14.54 | 0.0928 | 412.30 |
| n2_z_pair_rand12 | oracle_y0 | 0.0003 | 0.0003 | 13.30 | 0.0025 | 409.24 |
| n2_z_pair_rand12 | zero_y0 | 1.776 | 0.6358 | 14.31 | 0.2607 | 780.02 |
| n1_tip_z_rand15 | oracle_y0 | 0.00005 | 0.00007 | 13.34 | 0.0003 | 752.46 |
| n1_tip_z_rand15 | zero_y0 | 0.455 | 0.6505 | 13.78 | 0.2120 | 446.09 |

汇总图：

- `terminal_error_summary.png`

各工况姿态图：

- `n3_mixed_base_i4/oracle_vs_clean_shooting.png`
- `n3_mixed_rand20/oracle_vs_clean_shooting.png`
- `n2_z_pair_rand12/oracle_vs_clean_shooting.png`
- `n1_tip_z_rand15/oracle_vs_clean_shooting.png`

## 数据分析

本轮回答两个问题。

第一，去掉参考曲线和刚段中点后，末端位姿是否还能收敛？

答案是可以。4 个不同磁化/电流工况、2 种初值，共 8 组都达到 Stage B。即使使用 `zero_y0`，末端位置误差也保持在毫米级以内，姿态误差保持在约 `1.5deg` 以内。

第二，是否收敛到与参考 FK 完全一致？

只在末端位姿意义上一致，不在整条姿态曲线意义上一致。所有 case 的 centerline RMS drift 都在 `13-15mm` 左右。这说明末端位姿约束是欠定的：多条中间形态可以对应相近末端位姿。

`oracle_y0` 通常给出更小的末端误差和边界残差；`zero_y0` 也能达到 Stage B，但更慢或边界残差更大。这说明当前方法不完全依赖 oracle 初值，但优质初值仍显著改善收敛质量。

## 实验结论

clean terminal-pose shooting 在多个不同工况下可以稳定拟合参考 FK 的末端位姿。之前的末端位姿收敛不是由 centerline 或 rigid midpoint 数据泄露造成。

但如果“与参考一致”指整条导管曲线一致，则当前 clean shooting 不满足。它只保证末端位姿接近，不能保证中间形态一致。

当前卡点：

1. 速度慢，单组 shooting 约 `330-780s`。
2. 中间形态存在明显漂移。
3. sensitivity 仍未通过，不能直接接外层 IK 梯度。

## 下一步计划

1. 优先做 clean residual 下的 implicit shooting sensitivity。
2. 若需要控制中间形态，加入非 oracle 正则，例如内力/内矩范数、分段平滑、弯曲能量或安全通道约束。
3. 进一步减少 shooting residual evaluation 成本，目标把单组从数百秒压到分钟内。
