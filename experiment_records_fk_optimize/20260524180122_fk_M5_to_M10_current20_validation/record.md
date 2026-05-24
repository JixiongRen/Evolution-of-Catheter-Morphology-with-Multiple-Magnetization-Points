# 20260524180122_fk_M5_to_M10_current20_validation 实验记录

## Experiment Objective

在 `M_list >= [5,5,5]` 且 `M_list <= [10,10,10]`、`-20 <= I <= 20A` 的条件下，覆盖磁力段数量为 1、2、3 的 FK（forward kinematics，正运动学）求解性能测试。

每组实验需要输出导管姿态变化的 morphology evolution（形态演进）图：同一图中绘制 continuation（延续）各阶段中心线，并标注起点与导管末端路径。

## Experiment Plan

### Solver Setting

```text
tol=1e-5
dense_solve_method=pos
lm_damping=0.1
current_factors=0,0.5,1.0
stage_max_iter=40
stage_tol=1e-3
final_max_iter=1600
final_tol=1e-5
```

### Case Matrix

```text
M_list:
  - 5,5,5
  - 7,7,7
  - 10,10,10

coil_currents:
  - rand_a: 11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4
  - rand_b: -19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2
  - rand_c: 4.5,14.9,-17.6,8.8,-2.1,-11.4,19.2,-5.7

magnetization:
  - single_tip: 0,0,0; 0,0,0; 0,0,0.005301
  - double_z: 0,0,-0.005301; 0,0,0.005301; 0,0,0
  - triple_mixed: 0,0,-0.005301; 0,0,0.005301; 0.005301,0,0
```

电流为固定 seed（固定随机种子思想）产生的多线圈随机向量，全部满足 `-20 <= I <= 20A`，且不包含单线圈通电 case。总计 `27` 组。

## Execution Log

- 2026-05-24 18:01：创建实验目录。
- 2026-05-24 18:03：创建批量运行脚本 `run_fk_M5_to_M10_validation.py`。
- 2026-05-24 18:09：确认本轮正式实验使用 GPU，`jax_backend=gpu`，`jax_devices=["cuda:0"]`。此前 CPU 结果不作为本轮正式性能依据。
- 2026-05-24 20:59：完成 27 组 GPU FK 验证；每组均输出 `stage_morphology_evolution.png` 和 `stage_centerlines.npz`。

## Results

### GPU Environment

```text
jax_backend: gpu
jax_devices: ["cuda:0"]
XLA_PYTHON_CLIENT_PREALLOCATE: false
case_count: 27
```

所有电流均为多线圈随机电流组合，且每个分量满足 `-20 <= I <= 20A`。

### Batch Summary

输出文件：

```text
batch_summary.csv
batch_summary.json
batch_runtime.png
batch_residual.png
batch_time.txt
```

每个 case 输出：

```text
case_config.json
summary.json
centerline.npy
solution.npz
stage_centerlines.npz
stage_morphology_evolution.png
```

总体结果：

```text
total cases: 27
strict converged: 9
failed to reach tol=1e-5 within final_max_iter=1600: 18
```

按离散化分组：

| M_list | Strict ok | Mean steady time | NormE range |
| --- | ---: | ---: | ---: |
| `5,5,5` | 4/9 | 70.694s | 9.974e-6 to 4.066e-4 |
| `7,7,7` | 3/9 | 97.652s | 9.830e-6 to 4.021e-4 |
| `10,10,10` | 2/9 | 179.009s | 9.990e-6 to 3.797e-4 |

按磁化段数量分组：

| Magnet config | Active magnetic segments | Strict ok | Mean steady time |
| --- | ---: | ---: | ---: |
| `single_tip` | 1 | 3/9 | 116.273s |
| `double_z` | 2 | 4/9 | 115.929s |
| `triple_mixed` | 3 | 2/9 | 115.154s |

按随机电流分组：

| Current config | Strict ok | Mean steady time |
| --- | ---: | ---: |
| `rand_a` | 2/9 | 120.157s |
| `rand_b` | 1/9 | 114.551s |
| `rand_c` | 6/9 | 112.647s |

严格收敛 case：

```text
M555_single_tip_rand_c: 61.993s, normE=9.994e-6, n_iter=1287
M555_double_z_rand_a: 69.027s, normE=9.974e-6, n_iter=1358
M555_double_z_rand_c: 61.245s, normE=9.976e-6, n_iter=1203
M555_triple_mixed_rand_c: 78.830s, normE=9.976e-6, n_iter=1583
M777_single_tip_rand_c: 86.138s, normE=1.000e-5, n_iter=1460
M777_double_z_rand_a: 94.168s, normE=9.864e-6, n_iter=1553
M777_double_z_rand_c: 87.372s, normE=9.830e-6, n_iter=1426
M101010_single_tip_rand_c: 173.483s, normE=9.990e-6, n_iter=1594
M101010_triple_mixed_rand_b: 175.081s, normE=9.997e-6, n_iter=1442
```

## Data Analysis

1. `M_list` 增大后耗时明显上升：`5,5,5` 平均约 `70.7s`，`10,10,10` 平均约 `179.0s`。这符合 Jacobian（雅可比）和 dense normal equation（稠密正规方程）维度增大的预期。
2. 当前 `final_max_iter=1600` 对随机电流和更高离散化仍不够稳健。很多 case 停在 `1e-4` 量级，需要 adaptive continuation（自适应延续）或更强 globalization（全局化策略）。
3. 三组随机电流中 `rand_c` 最容易收敛，`rand_b` 最难；这说明电流方向组合显著改变 convergence basin（收敛盆）。
4. 单/双/三磁化段的平均耗时接近，主要差异来自电流和 `M_list`，而不是磁化段数量本身。
5. 每组都已保存 stage morphology evolution（阶段形态演进）图，可直接观察 continuation（延续）阶段中心线与 tip path（末端路径）。

## Conclusion

本轮满足用户约束：`M_list` 未低于 `[5,5,5]`，未超过 `[10,10,10]`；电流为随机多线圈组合且全部在 `[-20,20]A`；磁力段数量覆盖 1、2、3；每组实验均绘制了导管姿态演进过程和末端位置演化路径。

求解性能结论：上一轮 production-candidate（生产候选）参数对固定基准 case 有效，但对随机多线圈电流、多离散化组合仍不够通用。若要求所有 case strict 收敛，需要下一轮做自动化 damping schedule（阻尼调度）、current continuation（电流延续）分段自适应和失败 case 二阶段精修。

## Next Step Plan

1. 针对未收敛的 18 个 case，追加 adaptive refinement（自适应精修）实验，统计达到 `tol=1e-5` 需要的额外迭代。
2. 对 M=10 case 优先尝试 sparse/block solve（稀疏/块结构线性求解），否则 dense normal equation（稠密正规方程）成本会继续限制性能。
3. 将每组 `stage_morphology_evolution.png` 按 `M_list` 和磁力段数量整理成对照图册，便于肉眼分析形态异常。
