# IK 多点位姿观测与残差层实验记录

## 实验目的

本轮实验对应 IK 优化总体方案中的 P0 和 P1。目标是在进入多点隐式伴随 IK 前，先建立可靠的多点位姿观测层和 residual 层，验证 tip 与 rigid midpoint 的位置、姿态、磁体轴向残差和 SO(3) 残差约定。

本轮不推进 P3 外层多点 IK 求解器，也不将任何 P2 诊断结果标记为 production candidate。

## 实验方案

1. 新增 `inverse_kinematics/ik_observations.py`，提供 `ObservationSpec`、位姿抽取、position residual、axis residual、SO(3) residual 和 residual stack。
2. 新增轻量测试 `inverse_kinematics/tests/test_observation_interface.py`，验证四元数、旋转矩阵、SO(3) 小角近似、axis residual 小角方向和 `query_sites()` 契约。
3. 运行 FK replay probe，覆盖三类 case：
   - zero current；
   - single-coil current；
   - mixed-current/mixed-magnet。
4. 记录 FK 后端路径、分支、JAX device、残差范数、replay 几何漂移和姿态漂移。

## 运行命令

```bash
bash experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/commands.sh
```

## 实验过程记录

- 初始分支：`ik_optimize`。
- 使用环境：`conda run -n evolution_of_mcrs`。
- 所有命令显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 本轮仅新增观测/残差层和测试，不修改现有 IK 外层优化器。

## 实验结果

### P0 后端验证

- 当前分支：`ik_optimize`。
- FK 后端路径：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/forward_kinematics_optimized/fk.py`。
- JAX device：`cuda:0`。
- Python 环境：`/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python`。
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

### P1 单元测试

`inverse_kinematics/tests/test_observation_interface.py` 通过，覆盖：

- SO(3) log 小角近似；
- axis residual 零误差和小角方向；
- rotation matrix 正交性与 `det≈1`；
- `query_sites()` 的 tip position 与旧 `extract_tip_p_dim()` 逐元素一致；
- rigid midpoint pose 数量为 3；
- tip/rigid self-target residual 接近 0。

### P1 FK replay probe

三组 case 的结果见 `summary.json` 和 `summary.csv`。

| case | strict ok | normE | self residual norm | tip replay delta | rigid replay max delta | rigid orientation replay max error |
| --- | --- | --- | --- | --- | --- | --- |
| zero_current | false | 9.41e-4 | 5.28e-16 | 0 | 0 | 0 |
| single_coil_current | false | 6.24e-4 | 5.93e-16 | 0 | 0 | 0 |
| mixed_current_mixed_magnet | false | 6.11e-4 | 5.40e-16 | 0 | 0 | 0 |

说明：本轮 P1 主要验证观测约定和 residual 几何契约，不以小网格 probe 的 strict FK convergence 作为 P1 门禁。三组 replay 的几何和姿态漂移为 0，说明同一 FK 解上的观测层是确定和可复现的。strict ok 为 false 的原因是小网格快速 probe 在 `tol=1e-5` 下停在 `6e-4` 到 `9e-4` 的 raw residual，后续进入生产 IK 前仍需使用正式 FK case 做更严格 replay。

### P2 diagnostic-only probe

P1 几何契约通过后，尝试了一个小电流方向的 dense task sensitivity 诊断。结果见 `p2_task_sensitivity_probe.json`。

- `condition_number = 2.95e12`。
- `status = ill_conditioned`。
- FD relative error = `0.782`。
- grade = `untrusted`。
- `production_candidate = false`。

该结果不能作为生产梯度使用。它的价值是证明当前多点 task adjoint 必须保留病态诊断、FD audit 和 fallback 策略，不能直接把 dense 多点伴随接入 P3 外层优化器。

## 数据分析

P1 的结果支持以下判断：

1. 新增 `ObservationSpec` 和 residual 工具可以稳定处理 tip 和 rigid midpoint 的 position、axis、SO(3)/pose residual。
2. `query_sites()` 与旧 tip extraction 在 tip position 上一致。
3. rigid midpoint pose 数量与三段 rigid/magnet 配置一致。
4. 四元数到旋转矩阵的约定通过正交性和行列式检查。
5. self-target residual 约为机器精度量级，说明 residual 装配本身没有明显符号或尺度错误。

P2 的结果支持以下判断：

1. dense multi-task adjoint 原型可以运行，但当前 case 的 GN/LM normal matrix 极度病态。
2. position + axis 混合 residual 的方向导数与中心差分误差过大，不可进入生产路径。
3. 下一轮 P2 必须优先处理 conditioning、ridge/SVD 截断、任务尺度和更可靠的 equilibrium convergence，而不是直接推进 P3。

## 实验结论

P0 完成。P1 几何契约完成并通过当前小型测试与 replay probe。P2 只完成 diagnostic-only 原型和一次 FD audit，结果为 untrusted，不满足生产化条件。

本轮不应进入 P3 外层多点 IK。当前最重要的后续工作是让 P2 的 task sensitivity 在更严格 FK 收敛和更合理尺度下通过 direction FD audit。

## 下一步计划

1. 对 P2 重新设计小 case：使用更稳定的 FK 收敛配置，并分开验证 position、axis、SO(3) residual。
2. 增加 ridge/SVD 截断或矩阵自由诊断，降低 `H = J^T J + lambda D` 病态影响。
3. 在 P2 direction FD 相对误差进入可用或 diagnostic-only 范围前，不接入 P3 外层多点 IK。
