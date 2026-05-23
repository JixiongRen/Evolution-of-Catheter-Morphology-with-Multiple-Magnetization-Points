# eff_opt_exp_20260523192724_fk_old_new_same_params_compare 实验记录

## 实验目的

在完全相同的物理参数和显式求解参数下，对比旧 FK 与优化后新 FK 的求解精度、耗时和输出形态，排查此前 FK 优化是否引入了不合理的精度退化。

## 实验方案

- 旧 FK：`forward_kinematics/fk.py`
- 新 FK：`forward_kinematics_optimized/fk.py`
- 两者使用相同输入配置：
  - `flex_lengths=0.03,0.03`
  - `rigid_lengths=0.003,0.003,0.003`
  - `L_protrude=0.11`
  - `M_list=5,5,5`
  - gravity enabled
  - magnetics enabled
  - `coil_currents=0,0,0,0,0,50,0,0`
  - `m_body_list=0,0,-0.005301; 0,0,0.005301; 0.005301,0,0`
  - `p0=0,0,0`
  - `Q0=0.70710678,0,0.70710678,0`
  - `axis_body=0,0,1`
- 两者显式使用相同求解设置：
  - `tol=1e-5`
  - `max_iter=100000`
  - `lm_damping=1e-1`
  - `jac_method=fwd`
- 使用同一个仓库标定文件和 actuation table。
- 记录 wall time、最终迭代数、最终 `||E||`、是否收敛和最后 callback 图。

## 实验过程记录

- 2026-05-23 19:27：创建实验目录和记录文档。
- 2026-05-23 19:28：启动旧 FK，同一物理配置，显式 `tol=1e-5`。
- 2026-05-23 19:36：旧 FK 完成。
- 2026-05-23 19:36：启动新 FK，同一物理配置，显式 `tol=1e-5`。
- 2026-05-23 19:42：新 FK 完成。
- 2026-05-23 19:45：提取最后 callback checkpoint 的 centerline，并生成旧/新形态对比图。

## 运行命令

旧 FK：

```bash
/usr/bin/time -f 'WALL_SECONDS=%e\nMAX_RSS_KB=%M' -o time.txt \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  PYTHONPATH=/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/forward_kinematics/fk.py \
  --flex_lengths=0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --L_protrude=0.11 \
  --M_list=5,5,5 \
  --enable_gravity \
  --enable_magnetics \
  --calib_file=/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/calib/mpem_calibration_file_sp=40_order=1.yaml \
  --actuation_table_pkl=/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl \
  --coil_currents=0,0,0,0,0,50,0,0 \
  '--m_body_list=0,0,-0.005301; 0,0,0.005301; 0.005301,0,0' \
  --p0=0,0,0 \
  --Q0=0.70710678,0,0.70710678,0 \
  --axis_body=0,0,1 \
  --tol=1e-5 \
  --max_iter=100000 \
  --lm_damping=1e-1 \
  --jac_method=fwd \
  --plot
```

新 FK 命令相同，仅入口替换为：

```bash
/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/forward_kinematics_optimized/fk.py
```

## 实验结果

| 指标 | 旧 FK | 新 FK |
| --- | ---: | ---: |
| 入口 | `forward_kinematics/fk.py` | `forward_kinematics_optimized/fk.py` |
| wall time | `527.92 s` | `353.07 s` |
| max RSS | `2.29 GB` | `2.32 GB` |
| 收敛状态 | `Converged: True` | `Converged: True` |
| 终止迭代 | `4577` | `2555` |
| final `||E||` | `9.994e-06` | `9.969e-06` |
| 最后 callback | `iter=4500, ||E||=1.063e-05` | `iter=2500, ||E||=1.210e-05` |

最后 callback centerline 对比：

| 指标 | 数值 |
| --- | ---: |
| centerline 点数 | `48` |
| 最大逐点位置差 | `33.96 mm` |
| 平均逐点位置差 | `9.18 mm` |
| RMS 逐点位置差 | `15.84 mm` |
| 末端位置差 | `33.96 mm` |
| 相对 `L_protrude=0.11 m` 的最大误差 | `30.87%` |

末端位置：

- 旧 FK：`[0.0253055412, -0.0643168345, 0.0462209269] m`
- 新 FK：`[-0.0020532590, -0.0580908626, 0.0653531769] m`

输出文件：

- 旧 FK 日志：`docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/old_fk_tol1e-5/fk_stdout.log`
- 新 FK 日志：`docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/new_fk_tol1e-5/fk_stdout.log`
- 对比图：`docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/old_new_centerline_compare.png`
- 对比数据：`docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/centerline_compare_summary.json`

## 数据分析

新 FK 在同一显式求解参数下确实更快：

- wall time 从 `527.92 s` 降到 `353.07 s`，约 `1.50x`。
- 迭代数从 `4577` 降到 `2555`，约 `1.79x`。
- 显存或内存占用没有明显改善，RSS 基本相当。

但该速度收益伴随明显形态漂移。旧/新末端差 `33.96 mm`，超过 `L_protrude` 的 `30%`，也超过此前使用的 10% FK 近似标准。因此这不是可接受的等价加速。

日志和代码对照显示一个关键差异：新 FK 修改了 nondim scale 策略。

- 旧 FK scale：`L_ref=0.11, F_ref=3.18086256, M_ref=0.34989488`
- 新 FK scale：`L_ref=0.11, F_ref=1.06029, M_ref=0.000396086`

尤其是 `M_ref` 从约 `3.50e-1` 变成约 `3.96e-4`，相差约 `883x`。这会改变无量纲残差中力矩相关项的有效权重和停止判据含义。即使两边都达到 `||E||≈1e-5`，其物理残差和形态并不等价。

因此，当前证据支持用户判断：此前 FK 优化中至少有一部分不合理，不能只依据无量纲 `||E||` 和速度认定优化有效。

## 实验结论

在本组相同参数对比中，新 FK 获得约 `1.5x` wall-time 提速，但求解形态与旧 FK 不一致，末端偏差达到 `33.96 mm`。该优化不应直接作为可信 FK 路径继续使用。

最可疑的问题是新 FK 的 nondim scale 修改，尤其是 `M_ref` 的变化过大，导致停止精度和残差权重不可与旧 FK 对齐。

## 下一步计划

- 先回退或隔离新 FK 中的 scale 修改，恢复旧 FK 的 `F_ref/M_ref` 定义，再重新做同参数对比。
- 在确认物理形态一致前，不应继续基于当前新 FK 结果评估 IK 精度。
- 后续 FK 加速应把物理残差、逐点 centerline 和末端位置作为硬门禁，而不是只看无量纲 `||E||`。
