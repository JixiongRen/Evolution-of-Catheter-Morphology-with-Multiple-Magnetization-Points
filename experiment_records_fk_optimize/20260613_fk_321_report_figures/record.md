# 3.2.1 FK 报告图生成记录

## 目的

为中期报告 3.2.1 生成更适合论文排版的正运动学姿态演化图、复杂度耗时图和收敛曲线。

## 口径

- 复杂度耗时图复用既有 M5/M7/M10 批量扫参记录，不重新运行。
- 姿态演化采用每 20 个 LM 迭代步重入求解并保存一次中心线检查点。
- `m555_random_current`：复现 M555 批次中的 5 个代表性随机电流/磁化配置工况。
- `gravity_only`：开启重力，关闭磁场，八个线圈电流为 0。

## 输出图表

- `fig_321_existing_stage_evolution_panels.png/pdf`
- `fig_321_runtime_complexity.png/pdf`
- `fig_321_m555_checkpoint_evolution_panels.png/pdf`（当 M555 检查点工况不少于 5 组时生成）
- `fig_321_convergence_curves.png/pdf`
- `cases/*/fig_evolution.png/pdf`

## 本轮运行摘要

- `n1_base_x_rand_b`：ok=False，iter=2000，final_normE=2.183e-04，wall=720.29s。
- `n1_tip_z_gravity_only`：ok=False，iter=5000，final_normE=4.764e-04，wall=1562.74s。
- `n1_tip_z_rand_a`：ok=False，iter=2000，final_normE=4.527e-04，wall=726.07s。
- `n2_xz_pair_rand_b`：ok=False，iter=2000，final_normE=1.583e-04，wall=738.57s。
- `n2_z_pair_rand_a`：ok=False，iter=2000，final_normE=3.709e-04，wall=716.39s。
- `n3_xyz_rand_a`：ok=False，iter=2000，final_normE=3.537e-04，wall=720.22s。

## 复杂度统计摘要

- 1 个磁化点，网格总区间 15：平均 69.21s，严格收敛 1/3。
- 1 个磁化点，网格总区间 21：平均 103.75s，严格收敛 1/3。
- 1 个磁化点，网格总区间 30：平均 175.86s，严格收敛 1/3。
- 2 个磁化点，网格总区间 15：平均 70.48s，严格收敛 2/3。
- 2 个磁化点，网格总区间 21：平均 92.77s，严格收敛 2/3。
- 2 个磁化点，网格总区间 30：平均 184.54s，严格收敛 0/3。
- 3 个磁化点，网格总区间 15：平均 72.40s，严格收敛 1/3。
- 3 个磁化点，网格总区间 21：平均 96.44s，严格收敛 0/3。
- 3 个磁化点，网格总区间 30：平均 176.62s，严格收敛 1/3。

## 元信息

- git_commit: `0d14620697ff180940b8fc404cd2632a46cd41c9`
- jax_backend: `gpu`
- jax_devices: `['cuda:0']`
- fonts: 中文优先使用 SimSun/Songti SC；本机未安装时使用随项目保存的 Source Han Serif CN。英文优先使用 Times New Roman；本机未安装时回退 Liberation Serif。
