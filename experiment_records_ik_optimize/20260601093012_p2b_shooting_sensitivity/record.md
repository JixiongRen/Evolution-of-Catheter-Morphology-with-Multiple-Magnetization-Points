# P2b shooting sensitivity 实验记录

## 实验目的

本轮实验验证低维 shooting sensitivity（打靶法敏感度）是否能绕开上一轮全状态 dense LM normal equation（稠密 LM 法方程）病态问题，为后续双层 IK 提供更可信的外层梯度。

上一轮 P2 的主要问题是直接使用完整 FK 状态变量 `z` 构造：

```text
H = Jz.T @ Jz + lam I
```

该矩阵条件数达到 `1e11-1e12`，导致 position、axis、so3 任务残差对控制量的敏感度与有限差分严重不一致。

本轮不接入 production IK，只建立 experiment-local 诊断路径。

## 实验方案

本轮采用非生产的低维 cubic Bezier shooting surrogate（立方 Bezier 打靶代理）：

- 使用当前最优 FK `forward_kinematics_optimized` 作为 oracle（参照真值）。
- 对 FK oracle 输出的中心线做低维拟合。
- shooting state 使用 4 个三维 Bezier 控制点，共 12 维；其中 base point 和 tip point 被端点约束，内部自由拟合维度为 6。
- 对每个电流扰动重新运行 FK oracle，再拟合 shooting state。
- 用 shooting state 重构中心线和方向轴，构造 position、axis、position+axis 三类任务残差。
- 用控制电流的中心差分构造 task Jacobian。
- 对 task Jacobian 做 SVD（奇异值分解）诊断，记录有效秩、条件数和截断信息。
- 用独立方向扰动审计 shooting Jacobian 对 FK oracle residual finite difference（有限差分）的预测误差。

该方案仍然依赖 FK oracle 生成扰动形态，因此不是最终高性能 IK 方法。它的目标是验证“低维 shooting 表达 + 显式 SVD 诊断”是否比全状态 LM 法方程更稳定。

## 实验配置

固定参数：

- `L_protrude=0.11`
- `flex_lengths_tail=(0.03, 0.03)`
- `rigid_lengths=(0.003, 0.003, 0.003)`
- `m_body_list=(0,0,-0.005301); (0,0,0.005301); (0.005301,0,0)`
- base current: `(0,0,0,0,0,4,0,0) A`
- `tol=1e-5`
- `max_iter=160`
- `linear_solver=dense_fused_cholesky`
- `scale_include_external=True`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

审计 case：

- `M_list=1,1,1`
- `M_list=3,3,3`

有限差分设置：

- Jacobian step: `0.02 A`
- Audit steps: `0.02 A`, `0.05 A`
- Direction count: 2

误差分级：

- `<0.2`: `usable`
- `0.2-0.5`: `diagnostic-only`
- `>0.5`: `untrusted`

## 实验过程记录

运行命令见 `commands.sh`。

主要输出：

- `summary.json`
- `summary.csv`
- `full_stdout.log`
- `full_stderr.log`
- `M111/oracle_centerline.npy`
- `M111/shooting_centerline.npy`
- `M111/shooting_data.npz`
- `M111/oracle_vs_shooting_centerline.png`
- `M111/singular_values.png`
- `M111/fd_audit_errors.png`
- `M333/oracle_centerline.npy`
- `M333/shooting_centerline.npy`
- `M333/shooting_data.npz`
- `M333/oracle_vs_shooting_centerline.png`
- `M333/singular_values.png`
- `M333/fd_audit_errors.png`

## 实验结果

总体结果：

- 总审计行数：24
- `usable`: 0
- `diagnostic-only`: 3
- `untrusted`: 21

### M=1,1,1

FK oracle：

- `base_ok=false`
- `normE=8.343996644750701e-4`
- base FK wall time: `76.05 s`
- solve count: 25

shooting 几何拟合：

- tip error: `0`
- RMS centerline error: `9.23e-5 m`
- max centerline error: `1.71e-4 m`

Jacobian SVD：

- position rank: 3，condition number: `12.35`
- axis rank: 2，condition number: `2.98`
- position+axis rank: 5，condition number: `2.176e4`

有限差分审计最佳结果：

- position: relative error `0.2835`，`diagnostic-only`
- axis: relative error `0.4343`，`diagnostic-only`
- position+axis: relative error `0.4335`，`diagnostic-only`

### M=3,3,3

FK oracle：

- `base_ok=false`
- `normE=9.349954850247459e-4`
- base FK wall time: `153.46 s`
- solve count: 25

shooting 几何拟合：

- tip error: `0`
- RMS centerline error: `3.58e-4 m`
- max centerline error: `6.23e-4 m`

Jacobian SVD：

- position rank: 3，condition number: `67.00`
- axis rank: 2，condition number: `12.61`
- position+axis rank: 5，condition number: `9.163e4`

有限差分审计最佳结果：

- position: relative error `0.7237`，`untrusted`
- axis: relative error `0.7914`，`untrusted`
- position+axis: relative error `0.7911`，`untrusted`

## 数据分析

低维 shooting 表达确实显著改善了 Jacobian 的条件数：

- 上一轮全状态 LM 法方程条件数约 `1e11-1e12`。
- 本轮 M=1 的 position 和 axis 条件数分别约 `12.35` 和 `2.98`。
- 本轮 M=3 的 position 和 axis 条件数分别约 `67.00` 和 `12.61`。

但条件数改善没有直接转化为 `usable` 梯度。原因如下：

1. shooting 几何拟合仍是形态代理，不是物理平衡方程本身。
2. axis residual 对中心线切向估计敏感，当前用 Bezier 切向近似刚段姿态，误差较大。
3. 小扰动过小时，FK oracle 的求解噪声会主导有限差分；本轮曾尝试 `0.005/0.01 A`，误差明显恶化。
4. `0.02/0.05 A` 下 M=1 可达到 `diagnostic-only`，说明局部方向有部分可信信号，但尚不足以作为生产 IK 梯度。
5. M=3 中等网格下 Bezier 代理表达能力不足，中心线 RMS 误差增加，敏感度重新变为 `untrusted`。

## 实验结论

P2b 最低目标完成：已建立可复现的非生产 shooting sensitivity 诊断框架，并完成 M=1 和 M=3 的对照证据。

P2b 理想目标未达成：M=1 下 position 和 axis 最佳结果仅为 `diagnostic-only`，没有达到 `usable`。

当前不能进入 production IK，也不应把该路径接入外层 IK 优化器。

本轮比 P2 的改进：

- 条件数从 `1e11-1e12` 降到 `1e0-1e2` 量级。
- M=1 下 position/axis 出现 `diagnostic-only` 信号。

本轮仍未解决：

- 低维 shape surrogate 与真实 FK oracle 的敏感度不一致。
- 姿态/轴向残差仍缺少可靠的物理姿态参数化。
- 中等网格下 cubic Bezier 表达能力不足。

## 下一步计划

建议下一轮进入 `P2c-physical-shooting-state`：

1. 放弃纯几何 Bezier 代理，改为更接近 Cosserat rod 的低维物理 shooting state。
2. 候选变量：
   - 近端内力/内矩。
   - 分段常曲率参数。
   - 每段低阶曲率 mode coefficients。
3. 方向残差不要再用中心线切向近似刚段姿态，应从重构 frame 或 FK rigid midpoint frame 中建立一致约定。
4. 先要求 M=1 position 达到 `usable`，再要求 axis 达到 `usable`。
5. 若物理 shooting state 仍失败，应判断当前 FK oracle 在该电流附近存在分岔或不可微信号，转向采样型/无梯度外层 IK。
