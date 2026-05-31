# P2 任务残差敏感度审计记录

## 实验目的

本轮实验验证多观测残差对控制量的隐式敏感度是否可以作为双层 IK 外层梯度的可信来源。重点检查位置、方向轴、SO(3) 姿态以及位置加方向轴组合残差，在 LM 线性化、额外 ridge 正则和截断 SVD 稳定化下是否与有限差分结果一致。

## 实验方案

实验只修改诊断路径，不改变生产 IK/FK 求解路径。固定使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `evolution_of_mcrs` 环境。

审计对象为 `inverse_kinematics/ik_diff.py` 中的任务残差隐式敏感度计算。先修复旧 `compute_dp_dI_via_lm_adjoint` 返回值缺失问题，再为任务敏感度增加奇异值、有效秩、LM 阻尼、额外 ridge 和求解模式记录。

有限差分审计使用两组离散化：

- `M_list=1,1,1`，覆盖 `position`、`axis`、`so3`、`position_axis`。
- `M_list=3,3,3`，覆盖 `position`、`axis`。

每组使用两个单位电流扰动方向和两个步长 `0.10 A`、`0.25 A`。线性求解配置包括：

- `solve_base`：直接解 LM 法方程。
- `solve_ridge_1e-4`：额外加入 `1e-4` ridge。
- `svd_rcond_1e-8`：对 LM 法方程做截断 SVD。

## 实验过程记录

运行命令记录在 `commands.sh`。主要输出文件如下：

- `summary.json`：完整结构化结果。
- `summary.csv`：逐行审计结果。
- `quick_stdout.log`：审计脚本标准输出。
- `quick_stderr.log`：JAX/GPU 运行时警告。

回归检查：

- `python -m py_compile inverse_kinematics/ik_diff.py .../run_p2_task_sensitivity_audit.py` 通过。
- `inverse_kinematics.tests.test_observation_interface` 通过。
- `inverse_kinematics.tests.test_residual_interface` 通过。

## 实验结果

quick 审计生成 `72` 条对比记录：

- `usable`：0 条。
- `diagnostic-only`：1 条。
- `untrusted`：71 条。

基础 FK 求解结果：

- `M_list=1,1,1`：`base_ok=false`，`normE=8.6407e-4`，基础求解约 `71.15 s`，显式 `Jz` 约 `28.73 s`。
- `M_list=3,3,3`：`base_ok=false`，`normE=4.0147e-3`，基础求解约 `142.26 s`，显式 `Jz` 约 `60.95 s`。

典型敏感度误差：

- `M_list=1,1,1` 的 `position` 残差，直接法方程最优相对误差约 `0.717`。
- `M_list=1,1,1` 的 `so3` 残差，直接法方程最优相对误差约 `0.393`，仅达到诊断级。
- `M_list=3,3,3` 的 `position` 残差，直接法方程最优相对误差约 `0.891`。
- `M_list=3,3,3` 的 `axis` 残差，直接法方程最优相对误差约 `0.897`。

数值状态：

- 直接法方程路径普遍为 `ill_conditioned`，条件数达到 `1e11` 到 `1e12` 量级。
- 截断 SVD 路径被标记为 `truncated_svd`，有效秩低于状态维数。
- 额外 `1e-4` ridge 未能把相对误差稳定到可用范围。

## 数据分析

当前任务敏感度不能作为生产 IK 梯度使用。主要原因不是观测残差接口，而是两层耦合梯度的线性化基础不稳定：

1. LM 法方程高度病态，直接求解得到的方向对有限差分不一致。
2. 截断 SVD 可以显式暴露秩亏，但截断后预测方向仍与有限差分偏差较大。
3. 小 ridge 改善条件数有限，且会改变敏感度幅值，不能自然恢复有限差分一致性。
4. 基础 FK 虽然达到宽松残差范围，但 `base_ok=false`，说明在该诊断设置下内层平衡点仍不足以支撑严格隐式微分。

## 实验结论

P2 任务残差敏感度审计未通过。当前 `compute_task_sensitivities_via_lm_adjoint` 只能作为诊断工具，不能进入生产双层 IK 的外层梯度路径。

因此，本轮不进入 P3 的非生产 scaffold。后续如果继续推进 Richter 量级双层 IK，应优先解决以下问题：

- 用更可靠的内层 KKT 或平衡方程形式替代病态 LM 法方程敏感度。
- 将有限差分审计保留为每次梯度实现变更的门禁。
- 重新评估姿态残差的尺度归一化，避免位置与旋转观测在同一法方程中产生数值尺度冲突。

## 下一步计划

建议下一轮进入“P2b：稳定隐式梯度重构”：

1. 分离内层 equilibrium residual 与 LM optimizer residual，不再默认对 LM 法方程求敏感度。
2. 构造可审计的 KKT 线性系统或正则化投影系统，并显式记录秩、正则化和约束残差。
3. 先在 `M_list=1,1,1` 上要求 `position` 和 `axis` 有限差分相对误差进入 `0.2` 以内，再扩展到 `M_list=3,3,3`。
4. 保留当前生产 FK/IK 路径不变，所有新梯度实现通过显式开关进入实验路径。
