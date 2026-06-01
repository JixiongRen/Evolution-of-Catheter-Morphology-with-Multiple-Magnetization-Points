# P2c physical shooting solver 实验记录

## 实验目的

本轮实验实现一个非生产 physical shooting solver（物理打靶求解器），验证低维近端内力/内矩状态是否能绕开全状态 LM 法方程病态，并直接让电流、磁场、磁矩、弹性、重力、base pose 和边界条件进入 shooting 方程。

本轮目标不是完成 production IK，而是确认 Richter/Edelmann/Till 路线在当前代码物理模型下是否可行。

## 实验方案

实现位置：

```text
experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/run_p2c_physical_shooting_solver.py
```

该脚本是 experiment-local，不接入 production FK/IK。

第一版 shooting state：

```text
xi = [n0, m0]
```

其中 `n0` 是近端内力，`m0` 是近端内矩，共 6 维。优化变量使用无量纲形式：

```text
y = [n0 / F_ref, m0 / M_ref]
```

物理 forward：

- 从 FK oracle 相同的 base pose 出发。
- flexible segment 使用 `cosserat_rhs_dim` 做 RK4 积分。
- flexible distributed load 使用 FK oracle 的重力线载荷。
- rigid segment 使用 `compute_external_wrench_total_rigid` 计算重力和磁力/磁矩。
- rigid propagation 使用 `rigid_state_along_dim`。
- free-end boundary residual 使用 distal rigid 末端的无量纲内力/内矩。

磁场/磁矩来源：

- 使用 `forward_kinematics_optimized` 中的 magnetic model。
- 使用相同 `coil_currents` 和 `m_body_list`。
- 本轮只保留 rigid magnet lumped load（刚段磁体集中载荷），不实现 flexible distributed magnetic load。

## 实验配置

固定参数：

- `M_list=1,1,1`
- `L_protrude=0.11`
- base current: `(0,0,0,0,0,4,0,0) A`
- active current subspace: coil `5`
- `tol=1e-5`
- FK oracle `max_iter=120`
- shooting max evaluations: `4`
- RK4 steps per flex interval: `4`
- Jacobian step: `0.02 A`
- audit eps: `0.02 A`, `0.05 A`
- residual types: `position`, `axis`

说明：完整 8 线圈、多次 shooting re-solve 运行时间过长。本轮先使用单线圈子空间和 `fd_reuse_base_y` 形成可复现实验。该设置被标记为 approximation，不能作为 production-candidate。

## 实验过程记录

命令见 `commands.sh`。

中间尝试：

- 直接从零 `xi` 开始，多次 shooting re-solve 运行超过 15 分钟，中断。
- 缩小 active current subset 后仍过重。
- 加入 FK oracle proximal wrench 作为 informed guess 后，base geometry 明显改善。
- 最终使用 informed guess + fixed base shooting state 的轻量 FD audit，记录失败证据。

输出文件：

- `summary.json`
- `summary.csv`
- `base_stdout.log`
- `base_stderr.log`
- `quick_stdout.log`
- `quick_stderr.log`
- `M111/oracle_centerline.npy`
- `M111/physical_shooting_centerline.npy`
- `M111/shooting_state_xi.npy`
- `M111/physical_shooting_data.npz`
- `M111/jacobians.json`
- `M111/oracle_vs_physical_shooting_centerline.png`
- `M111/boundary_residual_history.png`
- `M111/singular_values.png`
- `M111/fd_audit_errors.png`

## 实验结果

总体 FD audit：

- rows: 8
- `usable`: 0
- `diagnostic-only`: 0
- `untrusted`: 8

FK oracle：

- `base_ok=false`
- `base_normE=8.6407e-4`
- base FK wall time: `74.49 s`

physical shooting：

- `shooting_success=false`
- `shooting_nfev=4`
- `shooting_cost=0.03767`
- shooting wall time: `63.51 s`
- boundary residual norm: `0.2745`

几何对比：

- tip error: `2.638e-2 m`
- RMS centerline error: `2.263e-2 m`
- max centerline error: `2.932e-2 m`
- rigid midpoint RMS error: `2.247e-2 m`
- axis proxy RMS error: `0.5483`

Jacobian SVD：

- position residual：rank 1，condition number 1.0，singular value `0.00570`
- axis residual：rank 0，singular value `0.0`

FD audit：

- position best relative error: `1.916`，`untrusted`
- axis best relative error: `1.0`，`untrusted`

## 数据分析

本轮 physical shooting 已经让电流、磁场、磁矩、弹性、重力和边界条件直接进入 forward equation，因此相比 P2b 的 Bezier 几何拟合更接近 Richter 路线。

但当前 physical shooting 仍未通过 base geometry gate。主要证据：

- RMS centerline error 达到 `2.26 cm`，相对约 11 cm 的导管长度超过 10%。
- free-end boundary residual norm 仍为 `0.2745`。
- axis proxy error 较大。
- shooting solver 未成功收敛。

因此按照决策规则，本轮不应评价其为可信 IK 梯度，更不能进入 production IK。

失败原因判断：

1. 6 维 `xi=[n0,m0]` 表达不足或求解器没有充分收敛。
2. 当前 RK4 IVP 与 FK collocation residual 存在数值离散差异。
3. rigid magnetic lumped load 与 FK oracle 一致调用，但 shooting 中缺少更完整的全局连接约束校正。
4. axis 残差完全失败，说明 frame/orientation 约定还未对齐。
5. 多次 re-solve 计算成本过高，必须先改进 shooting solve 性能或引入更好的 informed guess/continuation。

## 与 P2b 对比

P2b Bezier 几何代理：

- 几何贴合较好。
- 但不是物理模型。
- 敏感度不可信。

P2c physical shooting：

- 物理项进入方程。
- 但 base geometry 不通过。
- 敏感度仍不可信。

当前结论：P2c 的方向正确，但第一版 `xi=[n0,m0]` + RK4 IVP 还没有达到可用于 IK 外层梯度的条件。

## 实验结论

最低目标部分完成：已建立可运行的 experiment-local physical shooting solver，并明确定位当前与 FK oracle 的差距。

理想目标未完成：M=1 position-only FD audit 未达到 `usable`，axis 也未达到 `diagnostic-only`。

不进入 M=3，不进入 production IK。

## 下一步计划

建议下一轮为 `P2c.1-physical-shooting-convergence`：

1. 先只做 base geometry，不做 FD。
2. 使用 FK oracle 全状态提取更多 informed guess：
   - 每段近端/远端内力内矩。
   - 每段初始曲率或低阶 curvature modes。
3. 将 6 维 `xi` 扩展为分段低阶 physical modes，但保持低维。
4. 尝试把 shooting residual 从纯 free-end wrench 扩展为：
   - free-end wrench residual
   - rigid midpoint calibration residual
   - tip pose calibration residual
5. 先要求 M=1 centerline RMS error 低于 10%，再恢复 FD audit。
