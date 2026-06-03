# P2c5 JAX clean terminal-pose shooting 加速实验

## 实验目的

本轮实验目标是在不向残差中加入轴线参考、刚段中点参考或正则项的前提下，将 clean terminal-pose shooting 的单组热运行求解时间压缩到 30 s 以内，并保持 P2c4 多工况末端位姿稳定收敛。

## 实验方案

实验新增 JAX 版本的 segment-wrench shooting forward，并使用 `jax.jacfwd` 生成显式 Jacobian。残差严格限定为三类：

$$
r(y)=
\begin{bmatrix}
w_p(p_{end}(y)-p^*_{end})\\
w_R\mathrm{Log}(R_{end}^{*T}R_{end}(y))\\
w_b b(y)
\end{bmatrix}
$$

其中 $b(y)$ 是最终内力和内矩边界残差。centerline 和 rigid midpoint 仅作为诊断输出，不参与 residual。

## 实验配置

- 分支：`ik_optimize`
- Conda 环境：`evolution_of_mcrs`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- 基准工况：复用 P2c4 的 4 个工况。
- 初值：每个工况分别测试 `oracle_y0` 和 `zero_y0`。
- 对照：至少一个工况运行旧 SciPy clean shooting baseline。

## 运行命令

见 `commands.sh`。

## 当前状态

实验已完成。最终候选使用 JAX shooting forward、`jax.jacfwd` 显式 Jacobian 和实验级 dense LM/GN。为避免末端位姿达标但物理边界残差偏大的问题，最终 clean residual 权重设置为：

- `tip_weight=120.0`
- `so3_weight=2.0`
- `boundary_weight=0.5`

该调整没有引入新的残差类型，仍只包含 terminal position、terminal SO(3) 和 physical boundary。

## 结果

8 组 P2c4 复用工况全部通过门限。steady solve time 不计首次 JAX 编译。

| case | init | steady_s | compile_s | pos_mm | so3_deg | boundary | iter |
|---|---:|---:|---:|---:|---:|---:|---:|
| n3_mixed_base_i4 | oracle_y0 | 3.357 | 44.363 | 0.0000 | 1.5061 | 0.000430 | 16 |
| n3_mixed_base_i4 | zero_y0 | 2.797 | 0.015 | 0.4296 | 1.0760 | 0.021467 | 17 |
| n3_mixed_rand20 | oracle_y0 | 3.458 | 46.000 | 0.1000 | 1.9114 | 0.075824 | 19 |
| n3_mixed_rand20 | zero_y0 | 4.456 | 0.015 | 0.2006 | 1.1678 | 0.137087 | 22 |
| n2_z_pair_rand12 | oracle_y0 | 3.410 | 61.867 | 0.0240 | 0.0359 | 0.000559 | 17 |
| n2_z_pair_rand12 | zero_y0 | 3.130 | 0.017 | 0.1161 | 0.3559 | 0.002393 | 18 |
| n1_tip_z_rand15 | oracle_y0 | 4.866 | 51.360 | 0.0375 | 0.2273 | 0.007523 | 17 |
| n1_tip_z_rand15 | zero_y0 | 5.849 | 0.015 | 0.2563 | 1.3948 | 0.032670 | 21 |

关键汇总：

- 最大 steady solve time：`5.849 s`
- 最大 terminal position error：`0.4296 mm`
- 最大 terminal SO(3) error：`1.9114 deg`
- 最大 boundary residual：`0.137087`
- `reject_count=0`，`lambda` 均未到上限
- JAX/Python forward parity：tip position 差异约 `9.34e-18 m`
- Jacobian FD directional check：相对误差约 `1.37e-5`
- 旧 SciPy clean baseline 单组耗时：`317.344 s`

## 数据分析

P2c5 的主要加速来自两点：

1. 将 shooting forward 完整 JAX 化，避免 SciPy 数值差分中反复进入 Python RK4 循环。
2. 使用显式 Jacobian 和小规模 dense LM/GN，使每轮迭代只需一次 residual/Jacobian 评估和一个 18 维线性系统求解。

与 P2c4 的 330-780 s 单组耗时相比，P2c5 steady solve time 为 2.8-5.9 s，达到约 56-267 倍热求解加速。旧 SciPy clean baseline 在同一 `n3_mixed_base_i4/oracle_y0` case 上耗时 317.344 s，而 P2c5 对应耗时 3.357 s。

首次编译仍需 44-62 s，这是 JAX/XLA 编译成本，不计入本轮 steady solve gate。后续同 shape、同 case 的 zero-y0 求解 compile_s 约 0.015 s，说明 jit 复用生效。

P2c5 的 clean residual 没有使用 centerline 或 rigid midpoint 参考，因此图中的轴线形态差异仍应解释为 morphology ambiguity，而不是末端 IK 失败。

## 实验结论

本轮目标已达成。P2c5 可标记为 production-candidate clean shooting：在 P2c4 四工况、两类初值共 8 组上均满足 30 s steady solve、末端位置、末端姿态、边界残差和 LM 稳定性门限。

## 下一步计划

下一步应验证该 clean shooting solver 对外部电流扰动的 re-optimized sensitivity 是否与 FK oracle 一致。如果敏感性通过，再考虑接入外层双层 IK；如果敏感性仍失败，应进入 implicit shooting sensitivity 或更强的可观测性约束设计。
