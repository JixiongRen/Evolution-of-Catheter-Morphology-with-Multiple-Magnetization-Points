# 固定网格多磁体多电流 FK 验证记录

## 实验目的

在不改变收敛阈值和离散化网格数的前提下，更换磁体段数量、磁体磁化配置和外部电流组合，测试当前新 FK 生产路径的收敛情况、姿态结果和墙钟时间。

## 固定设置

- 求解器：`dense_fused_cholesky`
- 收敛阈值：`tol=1e-5`
- 最大迭代：`max_iter=100000`
- 离散化：`M_list=5,5,5`
- 电流限制：每个线圈电流均在 `[-20, 20] A`
- 运行环境：`conda evolution_of_mcrs`
- XLA 设置：`XLA_PYTHON_CLIENT_PREALLOCATE=false`

## 实验方案

测试 12 组 case。磁体段数覆盖 1、2、3 段，每类包含两种磁化配置；电流采用两组确定性多线圈随机组合，避免单线圈通电。

## 实验过程记录

执行命令见 `commands.sh`。本轮使用 `evolution_of_mcrs` 环境，显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，运行日志保存为 `stdout.log`，资源统计保存为 `stderr_time.log`。

实际总墙钟时间为 `1:09:58`。JAX 后端为 GPU，运行过程中 CUDA 初始化日志已记录在 `stderr_time.log`。每组 case 均保存 `case_config.json`、`summary.json`、`centerline.npy` 和 `final_centerline.png`。

## 实验结果

共运行 12 组固定网格 FK case，其中 10 组严格满足 `normE < 1e-5`，2 组触发 `reject_patience` 后停止。

| 指标 | 数值 |
| --- | --- |
| case 数 | 12 |
| 严格收敛数 | 10 |
| 未严格收敛数 | 2 |
| 稳态求解时间最小值 | 16.80 s |
| 稳态求解时间中位数 | 25.98 s |
| 稳态求解时间最大值 | 75.79 s |
| 稳态求解时间均值 | 36.49 s |
| 每组 warm-up 时间中位数 | 315.51 s |
| 含 warm-up 总时间中位数 | 346.26 s |
| 迭代数中位数 | 1401.5 |
| 最小最终 `normE` | `9.65e-6` |
| 最大最终 `normE` | `1.90e-5` |

未严格收敛 case：

| case | 停止原因 | 最终 `normE` |
| --- | --- | --- |
| `n2_z_pair_rand_b` | `reject_patience` | `1.90e-5` |
| `n3_mixed_rand_b` | `reject_patience` | `1.25e-5` |

按磁体段数统计：

| 磁体段数 | 严格收敛 | 平均稳态时间 | 平均迭代数 |
| --- | --- | --- | --- |
| 1 | 4/4 | 47.51 s | 2544.5 |
| 2 | 3/4 | 29.60 s | 1621.25 |
| 3 | 3/4 | 32.36 s | 1707.0 |

汇总文件：

- `batch_summary.csv`
- `batch_summary.json`
- `batch_metadata.json`
- `batch_wall_time.png`
- `batch_residual.png`
- `batch_final_centerlines.png`

## 数据分析

在固定 `M_list=5,5,5` 和 `tol=1e-5` 的条件下，当前 fused Cholesky 生产路径在大多数随机多线圈电流与不同磁体配置下可以稳定收敛。稳态求解时间主要由迭代数决定，最快 case 为 `n2_xz_pair_rand_b`，耗时 16.80 s，878 次迭代；最慢 case 为 `n1_tip_z_rand_b`，耗时 75.79 s，4185 次迭代。

需要注意的是，每组独立构建 engine 后仍存在约 300 s 量级的 warm-up 编译/热身成本。该成本不是单次稳态求解的主要算法耗时，但会显著影响批量扫参实验的端到端时间。后续若要做大规模参数扫描，应优先减少跨 case 的重复 warm-up，而不是继续压缩单次稳态 LM 迭代成本。

两个未严格收敛 case 都停在 `reject_patience`，最终残差分别为阈值的约 1.90 倍和 1.25 倍。由于本轮要求不改变阈值、不改变网格数，因此未做放宽或 continuation 重跑。这说明当前求解器在部分磁体/电流组合上仍存在接受步停滞问题。

## 实验结论

本轮验证表明：在固定阈值和固定网格下，当前新 FK 对多磁体段数、多磁化配置、多线圈随机电流组合具备较好的收敛能力，但并非所有 case 都严格收敛。若只看稳态求解，当前生产路径多数 case 已在 1 分钟内完成；如果把每组独立 warm-up 编译计入端到端成本，则批量实验仍然很慢。

## 下一步计划

下一步建议分两条线推进：

1. 保持物理参数、阈值和网格不变，排查 `reject_patience` case 的失败原因，重点看阻尼更新、步长接受准则和残差下降停滞。
2. 针对批量实验优化跨 case 编译复用，避免每组独立 engine 触发约 300 s 的 warm-up 成本。
