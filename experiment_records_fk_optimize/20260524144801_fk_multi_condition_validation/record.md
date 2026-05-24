# 20260524144801_fk_multi_condition_validation 实验记录

## Experiment Objective

在无人值守窗口内完成两件事：

1. 绘制优化前后 FK（forward kinematics，正运动学）在相同工况下的最终收敛中心线对比。
2. 改变输入电流、磁化配置和离散化段数，验证优化后 FK 求解路径在多工况下的性能。

## Experiment Plan

采用上一轮 production-candidate（生产候选）求解设置：

```text
tol=1e-5
dense_solve_method=pos
lm_damping=0.1
current_factors=0,1.0
stage_max_iter=20
stage_tol=1e-3
final_max_iter=1450
final_tol=1e-5
```

对比曲线采用两套参考：

```text
pre_algorithm_baseline:
experiment_records_fk_optimize/20260524114754_fk_strict_solver_algorithm_opt/fk_driver_lm03_solve_strict_baseline/centerline.npy

post_optimized_candidate:
experiment_records_fk_optimize/20260524114754_fk_strict_solver_algorithm_opt/fk_pos_lm01_shortstage_final1450_production/centerline.npy

old_cli_callback:
docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/old_fk_tol1e-5/centerline_last_callback.npy
```

最终几何判定优先使用同一 driver/extraction convention（驱动器/提取约定）的 `pre_algorithm_baseline` 与 `post_optimized_candidate`。旧 CLI callback（回调）中心线仅作为历史物理参考，不与当前 driver 输出混作严格判定。

多工况矩阵：

```text
M_list: 3,2,2 and 5,5,5
currents: 15A single coil, 50A single coil, mixed current
magnetization: single active magnet, double active magnets, triple active magnets
total cases: 18
```

## Execution Log

- 2026-05-24 14:48：创建实验目录。
- 2026-05-24 14:50：扩展 `fk_under_60s_driver.py`，支持 `--coil_currents`、`--m_body_list`、`--M_list` 参数。
- 2026-05-24 14:55：创建 `plot_pre_post_fk.py` 和 `run_multi_condition_fk.py`。
- 2026-05-24 15:14：完成 18 组多工况验证。运行环境中 JAX backend（JAX 后端）为 CPU，`nvidia-smi` 报告 GPU access blocked（GPU 访问被系统阻止），因此本轮 wall time（墙钟时间）是 CPU 性能参考，不等价于 GPU 性能。
- 2026-05-24 15:20：尝试本地提交前发现 `.git` 被只读挂载，`git add` 无法创建 `.git/index.lock`。当前实验文件和代码修改已保留在工作区，等待 `.git` 恢复可写后提交到本地 `dev`。

## Results

### 优化前后对比

输出文件：

```text
pre_post_fk_centerline_overlay.png
pre_post_pointwise_error.png
pre_post_metrics.csv
pre_post_metrics.json
```

关键误差：

| Reference | Candidate | Max error | RMS error | End error | Max rel to L_ref |
| --- | --- | ---: | ---: | ---: | ---: |
| `pre_algorithm_baseline` | `post_optimized_candidate` | 0.000115405 m | 0.000101307 m | 0.000111677 m | 0.001049 |
| `old_cli_callback` | `post_optimized_candidate` | 0.017372514 m | 0.011037807 m | 0.017372514 m | 0.157932 |

同一 driver/extraction convention（驱动器/提取约定）下，优化后与优化前求解结果几何偏移约 `0.10% L_ref`。旧 CLI callback（回调）中心线偏移约 `15.79% L_ref`，继续作为历史参考，不作为本轮严格几何判定依据。

### 多工况验证

输出文件：

```text
multi_condition_summary.csv
multi_condition_summary.json
multi_condition_runtime.png
multi_condition_residual.png
multi_condition_centerlines.png
multi_condition_time.txt
```

总计 `18` 组，严格收敛 `4` 组：

| Group | Strict ok | Mean wall time | NormE range |
| --- | ---: | ---: | ---: |
| `M_list=3,2,2` | 3/9 | 22.561s | 9.678e-6 to 3.542e-4 |
| `M_list=5,5,5` | 1/9 | 82.981s | 9.855e-6 to 2.866e-4 |
| single active magnet | 1/6 | 49.354s | 9.985e-6 to 3.542e-4 |
| double active magnets | 1/6 | 50.213s | 9.678e-6 to 1.694e-4 |
| triple active magnets | 2/6 | 58.746s | 9.781e-6 to 1.064e-4 |
| mixed current | 0/6 | 53.791s | 1.239e-5 to 3.542e-4 |

严格收敛 case：

```text
M322_single_tip_z_i50_c6: 22.938s, normE=9.985e-6
M322_double_z_pair_i50_c6: 16.337s, normE=9.678e-6
M322_triple_mixed_i15_c6: 20.852s, normE=9.781e-6
M555_triple_mixed_i50_c6: 96.906s, normE=9.855e-6
```

## Data Analysis

1. 优化前后同工况最终中心线基本一致，当前 solver algorithm（求解算法）优化没有明显改变固定 case 的物理解。
2. 多工况验证显示 `dense_solve_method=pos`（正定求解器）并不是所有工况的充分收敛策略；很多 case 在 `1450` 次 final refine（最终精修）后停在 `1e-5` 到 `1e-4` 量级。
3. `M_list=3,2,2` 的 CPU 运行耗时显著低于 `M_list=5,5,5`，但也存在非收敛 case，说明降阶网格不能直接作为严格解替代。
4. mixed current（混合电流）全部未严格收敛，说明当前短 continuation（延续）策略对复杂载荷不够稳，需要更强的 continuation 路径或 adaptive damping（自适应阻尼）。
5. GPU 当前不可用：`nvidia-smi` 报告 GPU access blocked，JAX 自动退回 CPU。本轮性能绝对值不能用于 GPU 性能结论，但可用于跨工况相对难度判断。

## Conclusion

本轮完成目标：

1. 已绘制优化前后 FK 最终收敛中心线对比曲线，并保存逐点误差图。
2. 已完成 `18` 组多工况 FK 验证，覆盖单/双/三磁铁、三种电流输入和两种 `M_list` 离散化。

结论：上一轮固定 case 的 strict solver optimization（严格求解器优化）在同一中心线约定下保持几何一致；但该参数不是通用稳健求解策略。对多磁化/混合电流/高离散化组合，需要进一步引入 adaptive continuation（自适应延续）或更强的 globalization（全局化）策略。

## Next Step Plan

1. 在 `.git` 恢复可写后，将本轮实验文件和 driver 参数化修改提交到本地 `dev`，不 push。
2. 在 GPU 恢复可用后复跑多工况矩阵，更新 GPU wall time（墙钟时间）。
3. 针对未收敛 case 自动追加 final refine 或切换 damping schedule（阻尼调度），统计达到 `tol=1e-5` 所需迭代数。
4. 下一轮重点优化 mixed current（混合电流）和 single/double magnet（单/双磁铁）场景的收敛路径。
