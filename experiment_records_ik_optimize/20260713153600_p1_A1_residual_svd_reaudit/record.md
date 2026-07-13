# P1 A1 三点残差雅可比直接求解再审计记录

## 一、实验目的

本轮承接 `20260713151200_p1_A1_three_point_reaudit` 的失败结果。上一轮发现：在 A1 的三个代表点上，旧求解器响应版、去阻尼高斯牛顿版和精确驻点版三类灵敏度相对有限差分误差普遍接近或超过 `100%`。本轮目标是定位失败原因，并验证“直接求解残差雅可比系统”是否能避免法方程病态放大。

这里的“残差雅可比直接求解”是指不再先构造 \(J_z^T J_z\)，而是直接对矩形残差雅可比 \(J_z\) 做奇异值分解，求解：

```math
J_z \frac{\partial z}{\partial c} = -J_c .
```

然后再通过末端观测链式法则计算：

```math
\frac{\partial p}{\partial c}
=
\frac{\partial p}{\partial z}
\frac{\partial z}{\partial c}.
```

这样可以避免将条件数平方。

## 二、代码修改

### 2.1 `inverse_kinematics/ik_diff.py`

新增 `sensitivity_mode="residual_svd"`。该模式直接对残差雅可比 \(J_z\) 做奇异值分解，不经过 \(J_z^T J_z\) 的法方程。

保留原三类模式：

- `solver_response`：旧求解器响应近似。
- `gn_svd`：去除求解器阻尼后的高斯牛顿驻点近似。
- `stationarity_exact`：对 \(J_z^T E=0\) 直接自动微分。

### 2.2 `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py`

补充审计字段：

- `fd_valid`：比 `fd_ok` 更严格，要求双侧扰动、双侧回环和 FK 平衡均有效。
- `fd_norm_per_unit`：有限差分单位响应范数。
- `jac_norm_per_unit`：隐式灵敏度单位响应范数。
- `norm_ratio`：隐式灵敏度范数与有限差分范数之比。

同时将自动电流通道选择从单一模式改为所有灵敏度模式的前若干通道并集，减少抽样偏差。

## 三、运行命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_12 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13 \
--out-dir experiment_records_ik_optimize/20260713153600_p1_A1_residual_svd_reaudit \
--selector best_accept \
--current-channels auto \
--auto-current-channels 2 \
--current-steps-a 0.25 \
--x-steps-mm 0.5 \
--sensitivity-modes residual_svd \
--solve-mode svd \
--svd-rcond 1e-8 \
--max-points 3
```

## 四、关键诊断证据

在进入 `residual_svd` 前，额外完成了同一状态下的残差偏导验证。对 A1-pt01 的保存状态，`dE/dI` 和 `dE/dx` 的自动微分结果与固定状态有限差分一致到 `1e-14` 量级。这说明电流和推进量确实进入了残差方程，显式残差接口不是本轮主因。

对 A1-pt01 进行本地消融时，直接在 \(J_z\) 上做奇异值分解，在 `rcond=1e-8` 附近可以将两个电流通道误差降到约 `1%`，而旧法方程截断路径约为 `100%`。这说明主要问题是法方程病态和截断策略，而不是磁场参数没有生效。

## 五、A1 三点再审计结果

本轮共得到 9 条有限差分对比，其中 7 条为有效强响应样本，2 条因回环失败标为无效。

### 5.1 有效样本

有效强响应样本的相对误差从小到大为：

```text
0.0044, 0.0086, 0.0249, 0.0306, 0.0524, 0.1507, 0.4490
```

其中：

- A1-pt01 两个电流通道误差分别约 `0.44%` 和 `0.86%`。
- A1-pt01 推进量通道误差约 `3.06%`。
- A1-pt12 推进量通道误差约 `2.49%`。
- A1-pt13 推进量通道误差约 `5.24%`。

这些结果说明 `residual_svd` 已经显著改善了多数有效样本的灵敏度质量。

### 5.2 仍未通过的样本

A1-pt12 的两个电流通道仍然偏差较大：

- 一个通道相对误差约 `15.1%`。
- 一个通道相对误差约 `44.9%`。

A1-pt13 的两个电流通道回环失败，不能作为有效有限差分样本。该点的电流方向更可能受到局部盆地跳变或步长过大的污染。

## 六、当前结论

本轮可以确认三点：

1. 显式残差对电流和推进量的偏导是正确的。
2. 原三类隐式灵敏度失败的主要原因不是漏项或符号错误，而是法方程病态和截断策略把真实响应方向压掉。
3. `residual_svd` 是当前最小可行修正，在 A1-pt01 和大部分有效样本上明显接近分支保持有限差分。

但本轮还不能宣布 P1 完成，因为：

- A1-pt12 仍有电流通道误差超过 `10%`。
- A1-pt13 的电流通道回环失败，说明固定 `0.25 A` 步长可能过大。
- 当前审计只用一个扰动尺度，尚未完成 `h, h/2, h/4` 的步长收敛检查。

## 七、下一步

1. 对 A1-pt12 和 A1-pt13 做自适应步长扫描，至少包括 `0.25 A`、`0.125 A`、`0.0625 A` 和推进量 `0.5 mm`、`0.25 mm`、`0.125 mm`。
2. 对 `residual_svd` 扫描 `svd_rcond`，优先检查 `1e-6`、`1e-8`、`1e-10`。
3. 只有在强响应通道通过 P1 门禁后，才将 `residual_svd` 接入正式外层 IK；在此之前它仍是诊断候选。
4. A2 不应提前进入外层优化；应先用同样方法完成代表点灵敏度门禁。
