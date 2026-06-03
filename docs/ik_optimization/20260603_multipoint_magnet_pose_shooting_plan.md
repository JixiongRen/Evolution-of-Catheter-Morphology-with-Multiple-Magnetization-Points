# 多磁体位姿目标 Shooting 试验方案

## 目标

本方案用于规划一轮约一天的试探性开发：在当前 P2c5 JAX clean shooting 基础上，尝试支持“末端磁体位姿 + 上一段磁体位置或姿态”作为目标。

本轮不是替代主线 IK 的正式路线，而是低风险分支验证：

- 若可行，形成 multipoint shooting（多点打靶）候选。
- 若不可行，保留诊断结论，回到双层 IK 主线。

## 当前能力边界

当前 P2c5 已支持多个非零磁体段参与物理前向计算。`m_body_list` 可以指定多段磁化，JAX shooting forward 会逐段使用 `params.magnet_params_list[i]` 计算磁场外力和外矩。

当前 P2c5 不支持多个磁体段位姿作为目标。现有 residual（残差）只包含：

$$
r(y)=
\begin{bmatrix}
w_p(p_{end}(y)-p^*_{end}) \\
w_R\operatorname{Log}(R_{end}^{*T}R_{end}(y)) \\
w_b b(y)
\end{bmatrix}
$$

其中：

- $p_{end}, R_{end}$：最后刚段远端位姿。
- $b(y)$：最终内力和内矩边界残差。
- $y \in \mathbb{R}^{18}$：三段近端内力和内矩，每段 6 维。

因此，目前验证的是“多磁体配置下的末端位姿 shooting”，不是“多磁体位姿目标 shooting”。

## 关键风险

### 观测点语义不统一

当前仓库中 `tip`、`rigid_mid`、`rigid_distal` 的命名存在历史不一致。后续必须固定唯一 convention（约定）：

- `rigid_prox(i)`：第 `i` 个刚段近端，即前一柔性段远端。
- `rigid_mid(i)`：从 `rigid_prox(i)` 沿刚段推进 `0.5 * L_i` 得到的刚段中点。
- `rigid_dist(i)`：从 `rigid_prox(i)` 沿刚段推进 `L_i` 得到的刚段远端。
- `terminal_end`：最后刚段远端，即物理导管末端。

多点目标必须明确使用哪一种 anchor（观测锚点）。不能把 `query_sites()["rigid_mid_*"]` 直接当成真实中点，除非先完成语义审计。

### 分段 wrench continuity 风险

当前 P2c5 shooting state（打靶状态）为每段近端 wrench（内力/内矩）。积分进入每一段时会重置当前段内力和内矩。最终 boundary residual（边界残差）只约束末端自由边界，不显式约束中间段 wrench continuity（段间内力/内矩连续性）。

直接加入多个磁体段位姿 residual 可能得到“几何上看起来命中，但段间内力漂移”的假解。

### 两点完整位姿易病态

末端完整 pose（位置 + SO(3) 姿态）为 6 维。若上一段磁体也加完整 pose，则额外 6 维。总 residual 维度会明显超过 18 维状态的有效可观测自由度，容易变成软几何拟合问题。

优先顺序应为：

1. 末端完整位姿 + 上一段磁体位置。
2. 末端完整位姿 + 上一段磁体轴向。
3. 末端完整位姿 + 上一段磁体完整 SO(3) 姿态。
4. 多个磁体段完整位姿，仅作为最后可选压力测试。

## 推荐算法路线

### 总体路线

保留当前 18D segment-wrench state（分段内力/内矩状态），先只扩展 residual，不改 FK 主体、不改状态维数、不引入 implicit sensitivity（隐式敏感性）。

新增 `clean_multisite_residual`：

$$
r(y)=
\begin{bmatrix}
r_{terminal\ pose}(y) \\
r_{upstream\ site}(y) \\
r_{boundary}(y)
\end{bmatrix}
$$

其中 `upstream site`（上游观测点）可选：

- position residual（位置残差）：

$$
r_p = \frac{p_i(y)-p_i^*}{\sigma_p}
$$

- axis residual（轴向残差）：

$$
r_a = \frac{a_i^* \times a_i(y)}{\sigma_a}
$$

- SO(3) residual（旋转群残差）：

$$
r_R = \frac{\operatorname{Log}(R_i^{*T}R_i(y))}{\sigma_R}
$$

第一版只实现 `position` 和 `axis`，完整 SO(3) 作为可选项。

### Continuation

Continuation（分阶段延续）用于避免一开始就把所有约束拉满。

建议 stage（阶段）：

| stage | terminal position | terminal SO(3) | upstream position | upstream axis/SO(3) | boundary |
|---|---:|---:|---:|---:|---:|
| 0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 1 | 1.0 | 0.5 | 0.0 | 0.0 | 1.0 |
| 2 | 1.0 | 1.0 | 0.3 | 0.0 | 1.0 |
| 3 | 1.0 | 1.0 | 1.0 | 0.3 | 1.0 |
| 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

残差维度保持固定，通过 stage scale（阶段权重）乘零或放大，避免 JAX 因 shape（张量形状）变化反复编译。

## 实现范围

### 模块修改

优先修改：

- `inverse_kinematics/ik_terminal_shooting_jax.py`

新增内容：

- `SiteTargetSpec`：描述观测点类型、索引、目标位置、目标姿态、目标轴向、sigma。
- `clean_multisite_residual(...)`：多点 residual。
- `extract_site_pose_jax(...)`：从 `CleanShootingOutput` 提取 `terminal_end`、`rigid_mid(i)`、`rigid_dist(i)` 等 site pose。

保留：

- `clean_terminal_residual(...)`，保证 P2c5 回归不受影响。

### 实验 runner

新增实验目录：

```text
experiment_records_ik_optimize/{YYYYMMDDHHMMSS}_p2c6_multipoint_magnet_pose_probe/
```

新增 runner：

```text
run_multipoint_magnet_pose_probe.py
```

职责：

- 复用 P2c4/P2c5 case。
- 生成 FK oracle 目标。
- 构造 multipoint target。
- 运行 JAX LM/GN solver。
- 输出 summary、history、centerline、site diagnostics、图像和中文报告。

## 一天内实验矩阵

### P0：语义 gate

目的：确认 FK oracle 与 JAX shooting 的 site pose 定义一致。

| ID | case | init | 检查 |
|---|---|---|---|
| S0 | `n3_mixed_base_i4` | `oracle_y0` | 比较 `terminal_end`、`rigid_mid(i)`、`rigid_dist(i)` 的 FK/JAX 差异 |

通过条件：

- `terminal_end` 已知约定一致。
- `rigid_mid(i)` 和 `rigid_dist(i)` 的定义可明确复现。
- 若无法确认，不进入 multipoint residual 实验。

### P1：位置多点

残差组合：

```text
terminal full pose + upstream rigid position + boundary
```

Smoke：

| ID | case | init | 目标 |
|---|---|---|---|
| S1 | `n3_mixed_base_i4` | `oracle_y0` | 复跑 tip-only baseline |
| S2 | `n3_mixed_base_i4` | `oracle_y0` | `terminal pose + all rigid position` |
| S3 | `n3_mixed_base_i4` | `zero_y0` | 无 oracle 初值稳定性 |

Core：

| cases | init | residual bundle |
|---|---|---|
| P2c4 四个 case | `oracle_y0`, `zero_y0` | `terminal pose + all rigid position + boundary` |

共 8 组。

### P2：轴向多点

残差组合：

```text
terminal full pose + upstream rigid position + upstream axis + boundary
```

Core：

| cases | init | residual bundle |
|---|---|---|
| P2c4 四个 case | `oracle_y0`, `zero_y0` | `terminal pose + all rigid position + last rigid axis + boundary` |

共 8 组。只有 P1 至少 6/8 通过时才进入 P2。

### P3：完整姿态可选压力测试

残差组合：

```text
terminal full pose + previous magnet full pose + boundary
```

只选两个 case：

- `n3_mixed_base_i4`
- `n3_mixed_rand20`

只做 `oracle_y0` 和 `zero_y0`。该阶段只判断可行性，不作为主结论。

## 成功判据

主判据：

- terminal position error（末端位置误差）≤ `1.0 mm`
- terminal SO(3) error（末端姿态误差）≤ `2.0 deg`
- upstream rigid position RMS（上游刚段位置均方根误差）≤ `1.0 mm`
- upstream axis error（上游轴向误差）≤ `5.0 deg`
- boundary residual norm（边界残差范数）≤ `0.2`
- steady solve time（热启动求解时间）中位数 ≤ `15 s`，最差 ≤ `30 s`
- `reject_count` 不爆炸
- `lambda_final < 1e10`

诊断指标：

- centerline RMS error（中心线均方根误差）
- centerline max error（中心线最大误差）
- Jacobian condition estimate（雅可比条件数估计）
- singular values（奇异值）

诊断指标不进入 residual，不作为 clean target。

## 失败与回退条件

立即回退：

- P0 语义 gate 不通过。
- 必须引入 centerline target 才能稳定。
- `C1` 在 `oracle_y0` 下都不能稳定收敛。
- 多点 residual 导致 terminal gate 明显退化。

停止 P2/P3：

- axis residual 一加入就导致多个 case `lambda` 到上限。
- boundary residual 显著劣化。
- steady solve time 超过 P2c5 baseline 两倍以上且没有形态改善。

保留可用结论：

- 若 P1 通过但 P2/P3 失败，则保留“末端完整位姿 + 上游位置”作为可行候选。
- 若 P1 不通过，则说明当前 18D 分段 shooting 不适合多点几何目标，应回到主线 IK。

## 推荐结论预期

最可能可行方案：

```text
terminal full pose + previous magnet position + boundary
```

更物理但更难的方案：

```text
terminal full pose + previous magnet axis + boundary
```

不建议作为第一天主方案：

```text
terminal full pose + previous magnet full SO(3) pose + boundary
```

原因：完整双点姿态约束容易过约束，且当前 shooting 缺少显式段间 wrench continuity。

## 后续若可行

若 P1/P2 稳定通过，再进入下一阶段：

1. 增加段间 wrench continuity 诊断或 residual。
2. 做 re-optimized current sensitivity（重新优化后的电流敏感性）。
3. 接入外层 IK，验证是否能求解电流。

