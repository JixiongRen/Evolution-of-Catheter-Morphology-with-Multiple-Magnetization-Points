# IK 优化总体方案：面向 Richter 量级的多点磁控导管逆静力学求解

## 1. 背景与目标

当前项目已经完成 FK 求解器的一轮较大幅度优化，固定高难度 FK case 在生产路径下可达到几十秒量级。下一阶段目标是优化 IK，尤其是双层逆静力学路径，使其能够更接近 Richter et al. 2021 在多磁体姿态控制中展示的计算量级。

本方案的目标不是直接复刻 Richter 的实现，而是吸收其有效思想，并结合本项目已有的 JAX 可微 FK、无量纲 Cosserat 残差和隐式伴随灵敏度，形成更适合本项目多段磁体导管的工程路线。

目标分为三层：

1. 物理正确性目标：同一控制输入下，IK 输出必须能通过 FK replay，即使用 FK 重新计算后姿态、平衡残差和中心线形态不能发生不可接受漂移。
2. 数值稳定性目标：IK 不再只依赖末端位置误差，而是引入多磁体位姿、中心线采样点和必要正则项，提高问题正定性，减少奇异和分支跳变。
3. 速度目标：先将完整 IK 稳态求解压缩到数十秒，再进一步向 Richter 式单步雅可比更新 `0.1-0.5 s` 和闭环 Hz 级控制频率靠近。

## 2. Richter 论文要点

Richter et al. 2021 的工作是 `Multi-Point Orientation Control of Discretely-Magnetized Continuum Manipulators`。其核心配置是柔性连续体结构携带两个刚性 NdFeB 永磁体，由 BigMag 六电磁铁系统在非均匀磁场中驱动。

该工作有几个值得迁移的关键点：

1. 控制目标是多磁体姿态，而不是单一末端位置。
2. 使用 Cosserat rod model 计算连续体静力平衡。
3. 构造 orientation Jacobian，即外部电磁系统控制量变化到磁体姿态变化的局部线性映射。
4. 使用 informed guess 改善 BVP 初值。
5. 指出多个 BVP 求解本质上相互独立，可以并行化。

该论文报告的关键耗时为：

- 实验中 orientation Jacobian 计算约 `158 ± 64 ms`。
- 不使用 informed guess 时约 `165 ± 53 ms`。
- 扩展仿真系统中约 `260 ± 41 ms` 或 `420 ± 16 ms`。
- 单个 BVP 平均约 `20 ms`。
- 整体闭环控制频率受形状重建和雅可比计算限制，约 `2 Hz`。

这些数据不能直接等价于本项目完整 IK 耗时。Richter 的 `20 ms` 是单个 BVP，不是完整多初值、多目标、多阶段 IK。其 `158 ms` 主要是姿态雅可比构造时间，也不是全局求解到目标点的完整耗时。因此本项目应将其作为局部线性化和闭环更新的参考量级，而不是完整 IK 立即达到的验收线。

## 3. 当前代码现状

### 3.1 已有基础

当前仓库已经具备迁移 Richter 思路的若干基础：

- `ForwardKinematicsEngine` 已封装可复用 FK 引擎，支持固定几何、磁场、材料配置和 warm start。
- FK 求解器已有 JAX JIT 复用和 `dense_fused_cholesky` 路径。
- `ik_diff.py` 已实现基于 LM/GN 的隐式伴随灵敏度，可计算 tip position 对电流和 protrusion 的导数。
- `ForwardKinematicsEngine.query_sites()` 已能返回 tip pose 和 rigid midpoint pose，可作为多磁体位姿观测的入口。

### 3.2 主要问题

当前 IK 与 Richter 量级仍有明显距离：

1. IK 主要是 position-only，即只约束 tip position。
2. 多磁体姿态没有进入 residual，因此存在末端位置相近但磁体姿态或中心线形态错误的伪解。
3. IK 仍有较多串行 FK 调用，尤其是 trial、line search、有限差分和多初值尝试。
4. 灵敏度层仍默认 materialize dense Jacobian，矩阵自由伴随没有成为生产路径。
5. `inverse_kinematics`、`single_layer_ik` 和 `forward_kinematics_project` 之间包入口不完全统一，存在导入旧 FK 或备份 FK 的风险。
6. 控制维度存在 8 线圈硬编码，未来扩展到更多控制变量或外部磁体 pose 会不方便。

## 4. 物理与数学模型

### 4.1 Cosserat 静力平衡

导管可近似为分段 Cosserat rod。设弧长参数为 \(s\)，中心线位置为 \(p(s)\)，截面姿态为 \(R(s) \in SO(3)\)，内力为 \(n(s)\)，内矩为 \(m(s)\)。静力平衡方程为：

$$
\begin{aligned}
p'(s) &= R(s)v(s), \\
R'(s) &= R(s)[u(s)]_\times, \\
n'(s) + f_{\mathrm{ext}}(s) &= 0, \\
m'(s) + p'(s) \times n(s) + \tau_{\mathrm{ext}}(s) &= 0.
\end{aligned}
$$

其中 \(v(s)\) 是剪切/拉伸应变，\(u(s)\) 是弯曲/扭转应变，\([\cdot]_\times\) 表示反对称矩阵。

本构关系可写为：

$$
\begin{aligned}
n(s) &= R(s)K_{\mathrm{se}}\left(v(s)-v^\ast(s)\right), \\
m(s) &= R(s)K_{\mathrm{bt}}\left(u(s)-u^\ast(s)\right).
\end{aligned}
$$

对本项目而言，柔性段通过离散节点近似，刚性磁体段通过 rigid segment state 近似。当前 FK 的 `rigid_mid_p_dim` 和 `rigid_mid_Q_wxyz` 可解释为刚性磁体中点位姿。

### 4.2 多磁体磁力与磁矩

第 \(i\) 个磁体在局部坐标系下磁矩为 \(\mu_{i,\mathrm{body}}\)，其世界坐标磁矩为：

$$
\mu_i = R_i \mu_{i,\mathrm{body}}.
$$

外部电流或磁场控制量记为 \(I\)，磁场为 \(B(p, I)\)。理想点偶极近似下，磁体受到的磁矩与磁力可写为：

$$
\tau_i = \mu_i \times B(p_i, I),
$$

$$
f_i = \nabla_p \left(\mu_i^\top B(p_i, I)\right).
$$

若磁场近似均匀，则磁力项较弱或为零，主要通过磁矩驱动姿态变化；若磁场有明显梯度，则磁力和磁矩都应进入平衡残差。

这说明多磁体 IK 不能只看末端位置。因为磁体姿态 \(R_i\) 决定 \(\mu_i\)，从而决定磁矩 \(\tau_i\)。如果姿态约束缺失，IK 可能找到末端接近目标但物理磁控响应不稳定的解。

### 4.3 双层逆静力学形式

推荐将 IK 写成 reduced-space inverse statics，即内层满足平衡，外层优化控制量。

内层 FK：

$$
z^\ast(q) = \arg\min_z \frac{1}{2}\|E(z, q)\|^2,
$$

其中 \(z\) 是离散 Cosserat 状态，\(q\) 是控制变量，例如：

$$
q = [I_1,\ldots,I_{n_I}, x]^\top.
$$

外层 IK：

$$
\min_q \frac{1}{2}\|r(z^\ast(q), q)\|^2 + \frac{1}{2}\|r_{\mathrm{reg}}(q)\|^2.
$$

这里 \(r\) 是任务残差，\(r_{\mathrm{reg}}\) 是电流幅值、插入长度、平滑性和上一步控制差分正则项。

## 5. 多点位姿残差设计

### 5.1 位置残差

对观测点 \(i\)，位置残差为：

$$
r_{p,i} = \frac{p_i - p_{i,\mathrm{des}}}{\sigma_{p,i}}.
$$

其中 \(\sigma_{p,i}\) 是位置误差尺度。对 tip 可以设置较小尺度，对中间磁体或中心线 waypoint 可设置较宽尺度。

### 5.2 姿态残差

姿态残差建议使用 \(SO(3)\) 的李代数对数映射：

$$
r_{R,i} = \frac{\log\left(R_{i,\mathrm{des}}^\top R_i\right)}{\sigma_{R,i}}.
$$

这里 \(\log: SO(3)\rightarrow\mathbb{R}^3\)，输出旋转误差轴角向量。

如果当前系统更关注磁体磁化轴方向，而不是完整 roll 姿态，可使用轴向残差：

$$
a_i = R_i a_{i,\mathrm{body}},
$$

$$
r_{a,i} = \frac{a_i \times a_{i,\mathrm{des}}}{\sigma_{a,i}}.
$$

两者的取舍：

- 完整 \(SO(3)\) residual 信息更充分，但对绕磁化轴自旋也施加约束。
- 轴向 residual 更贴近磁偶极方向控制，也更接近 Richter 的 orientation target，但缺少完整姿态约束。

建议早期实现两种模式：

1. `orientation_mode=axis`：磁化轴方向控制，优先用于稳定性实验。
2. `orientation_mode=so3`：完整姿态控制，用于更高阶任务。

### 5.3 总残差

综合残差可写为：

$$
r =
\begin{bmatrix}
w_p r_p \\
w_R r_R \\
w_a r_a \\
w_c r_c \\
w_I I \\
w_x (x-x_{\mathrm{ref}}) \\
w_{\Delta I}(I-I_{\mathrm{prev}}) \\
w_{\Delta x}(x-x_{\mathrm{prev}})
\end{bmatrix}.
$$

其中 \(r_c\) 是中心线形态残差，可用于防止仅末端/磁体点对齐而整体形态异常。

## 6. 隐式伴随灵敏度

### 6.1 精确隐式形式

若平衡方程可写为：

$$
E(z,q) = 0,
$$

并且 \(E_z\) 局部非奇异，则有：

$$
\frac{\partial z^\ast}{\partial q}
=
-E_z^{-1}E_q.
$$

任务残差为 \(r(z^\ast(q),q)\)，其导数为：

$$
\frac{dr}{dq}
=
r_q - r_z E_z^{-1}E_q.
$$

### 6.2 伴随形式

直接求 \(E_z^{-1}E_q\) 代价较高。可对每个任务残差方向求伴随变量：

$$
E_z^\top \lambda = r_z^\top,
$$

得到：

$$
\frac{dr}{dq}
=
r_q - \lambda^\top E_q.
$$

当前项目内层实际是 overdetermined residual 的 LM/GN 问题，因此生产实现可使用 GN/LM 近似：

$$
H = E_z^\top E_z + \lambda_{\mathrm{LM}}D,
$$

$$
H U = r_z^\top,
$$

$$
\frac{dr}{dq}
\approx
r_q - U^\top E_z^\top E_q.
$$

当前 `ik_diff.py` 已使用类似形式，但任务观测量只有 tip position。下一步应推广为任意 observation residual，包括多磁体位置、磁体轴向和 \(SO(3)\) 姿态残差。

### 6.3 为什么不用 Richter 的纯数值雅可比作为主路径

Richter 使用数值扰动方式构造 orientation Jacobian。对每个控制量扰动后重新求 BVP，得到局部差分。该方式概念简单，但对本项目不应作为生产主路径：

1. 控制维度为 8 个线圈再加 protrusion，有限差分至少需要 \(2(n_q)\) 次 FK。
2. 每次 FK 即使已优化仍有明显成本。
3. 差分步长会受到平衡分支、容差和磁场插值噪声影响。
4. 大角度或强非线性区域内，差分雅可比不稳定。

因此建议：

- 生产路径默认使用 implicit adjoint。
- Richter 式 finite-difference Jacobian 仅作为 small case audit，即小规模正确性抽检。

## 7. 初始化与全局化策略

IK 速度主要取决于外层迭代次数、每轮 FK 数量和分支稳定性。优质初始化和 continuation 是核心。

### 7.1 初始化候选

每个目标可生成多个初值：

1. 上一帧 accepted state。
2. 零电流平衡状态。
3. 小电流随机扰动。
4. 磁化感知线性最小二乘初值。
5. 历史数据库近邻初值。
6. coarse mesh 解 lift 到 fine mesh。

磁化感知初值可由简化线性模型得到：

$$
\Delta y \approx A I,
$$

其中 \(\Delta y\) 可以是目标方向、磁体姿态轴误差或 tip displacement。通过带界约束最小二乘：

$$
\min_I \|AI-\Delta y\|^2 + \alpha\|I\|^2,
\quad
I_{\min} \le I \le I_{\max}.
$$

### 7.2 Load continuation

对磁场载荷使用 continuation：

$$
I(t) = t I_{\mathrm{target}}, \quad t \in [0,1].
$$

分阶段求解：

$$
t_0=0 < t_1 < \cdots < t_K=1.
$$

每个阶段使用上一阶段的 \(z^\ast\) 作为 warm start。这样能减少突变导致的分支跳变。

### 7.3 Target continuation

对姿态目标也做 continuation：

$$
R_{\mathrm{des}}(t)
=
R_0 \exp\left(t \log(R_0^\top R_{\mathrm{des}})\right).
$$

对位置目标：

$$
p_{\mathrm{des}}(t)
=
(1-t)p_0 + t p_{\mathrm{des}}.
$$

这样能避免 \(SO(3)\) log 在大角度目标附近突然跨越分支。

### 7.4 Coarse-to-fine

先用低离散化网格求解 basin，再 lift 到目标网格精修：

$$
M_{\mathrm{coarse}} \rightarrow M_{\mathrm{fine}}.
$$

注意早期验收中不应改变最终网格和最终容差；coarse stage 只允许作为初始化过程，最终结果必须在目标配置下 FK replay。

## 8. 代码迁移路径

### 8.1 P0：分支与架构整理

目标：在 `ik_optimize` 分支上建立清晰、安全的 IK 优化工作面。

任务：

1. 确认所有后续 IK 优化仅在 `ik_optimize` 分支进行。
2. 建立 IK 优化文档目录和实验记录目录。
3. 明确生产 FK 后端为 `forward_kinematics_project/forward_kinematics_optimized`。
4. 梳理 `inverse_kinematics` 和 `single_layer_ik` 的 FK 导入路径，防止导入旧副本。

验收：

- `git branch --show-current` 输出 `ik_optimize`。
- 文档和后续实验日志路径固定。
- smoke import 能确认使用生产 FK 后端。

### 8.2 P1：ObservationSpec 与多点位姿观测

目标：把 tip-only observable 扩展为多点位姿 observable。

新增概念：

```text
ObservationSpec:
  site_type: tip | rigid_midpoint | centerline_s | custom
  site_index: int | None
  residual_type: position | axis | so3 | pose
  weight / sigma
```

任务：

1. 在 IK 层新增 observation parser。
2. 复用 `ForwardKinematicsEngine.query_sites()` 输出 tip pose 和 rigid midpoint pose。
3. 实现 quaternion 到 rotation matrix，以及 \(SO(3)\) log。
4. 实现 axis residual 和 so3 residual。
5. 增加离线观测测试：给定 FK 解，能正确抽取磁体中点位置和姿态。

验收：

- 同一 FK state 下，tip position 与旧路径一致。
- rigid midpoint pose 数量与 `m_body_list`/`rigid_lengths` 一致。
- 小角度姿态扰动下，\(SO(3)\) residual 与有限差分方向一致。

### 8.3 P2：多点隐式伴随灵敏度

目标：将 `compute_tip_sensitivities_via_lm_adjoint` 推广为 `compute_task_sensitivities_via_lm_adjoint`。

数学接口：

$$
J_{r,q}
=
\frac{dr}{dq}
\approx
r_q - U^\top E_z^\top E_q.
$$

任务：

1. 将 `_p_of_z_params` 泛化为 task observable function。
2. 输出 \(r_z\)、\(E_z\)、\(E_I\)、\(E_x\)。
3. 同时支持 position、axis、so3 residual。
4. 增加 directional finite difference audit：

$$
\frac{r(q+\epsilon d)-r(q-\epsilon d)}{2\epsilon}
\approx
J_{r,q}d.
$$

5. 记录 condition number、stationarity norm、residual norm 和 sensitivity status。

验收：

- 小 case 中 direction FD 相对误差可接受。
- 弱平衡、病态条件数、FD 不一致时明确降级或标记。
- 不再把病态 adjoint 结果静默当作可信梯度。

### 8.4 P3：多点位姿 IK 外层优化器

目标：构建新的 production candidate IK 外层，不再局限于 tip position。

外层变量：

$$
u = [u_I, u_x],
$$

其中：

$$
I = I_{\max}\tanh(u_I),
$$

$$
x = x_{\min} + (x_{\max}-x_{\min})\sigma(u_x).
$$

任务：

1. 将 residual assembly 从固定 `r_p` 扩展到 observation list。
2. 支持磁体姿态权重、tip 权重、电流正则和平滑项。
3. 支持 Levenberg-Marquardt 或 trust-region Gauss-Newton 外层。
4. 引入 step clipping 和 target continuation。
5. 每次 accepted state 保存 FK replay 所需最小信息。

验收：

- 单目标 tip position 与旧 IK 在相同配置下结果接近。
- 两磁体 axis orientation target 能降低姿态误差。
- 结果通过 FK replay，raw equilibrium residual 不劣化。

### 8.5 P4：Batch FK/BVP 并行

目标：迁移 Richter 中“多个 BVP 可并行”的核心速度思想。

需要批量化的场景：

1. 多初值候选。
2. continuation 多阶段候选筛选。
3. line search 多 trial。
4. finite-difference audit。
5. Richter-style numerical Jacobian 对照实验。

两种实现路径：

1. JAX `vmap` 批量 solve：形状固定、速度潜力高，但需要处理每个样本收敛步数不同的问题。
2. 多进程/多 GPU 并行：工程实现简单，适合作为第一阶段 batch runner，但 JAX 编译复用较弱。

建议顺序：

1. 先实现 Python 层 batch runner，用独立 engine 或 engine pool 并行多 seed。
2. 再实现 JAX fixed-iteration batched LM，用 mask 处理已收敛样本。
3. 最后将 batch FK 接入 IK 外层，使 trial 和 seed selection 同步并行。

验收：

- batch 结果与串行逐个 FK 结果几何一致。
- 多 seed 吞吐至少优于串行。
- 记录每个 candidate 的 wall time、normE、centerline error 和 final status。

### 8.6 P5：矩阵自由伴随与线性算子接口

目标：减少 dense Jacobian materialization，降低显存和大网格成本。

当前底层已有：

- \(Jv\)：residual JVP。
- \(J^\top v\)：residual VJP。
- matrix-free CG 线性解路径。

应新增统一接口：

```text
LinearizedEquilibrium:
  residual(z, q)
  jvp_z(v)
  vjp_z(w)
  jvp_q(dq)
  vjp_q(w)
  normal_matvec(v) = Jz.T @ (Jz @ v) + damping * v
```

伴随方程用迭代法求：

$$
\left(E_z^\top E_z + \lambda D\right)U = r_z^\top.
$$

验收：

- 小网格上 matrix-free adjoint 与 dense adjoint 接近。
- 中网格上显存下降。
- 如果迭代法不稳定，必须 fallback 到 dense path，不得静默失败。

### 8.7 P6：闭环式 Richter benchmark

目标：构造类似 Richter 的局部姿态控制 benchmark，用于客观对标。

实验配置：

1. 选择两段或三段磁体。
2. 固定 \(M=[5,5,5]\) 或项目标准网格。
3. 给出两磁体目标 axis orientation。
4. 从当前 FK 姿态出发，执行小步姿态更新。
5. 每步只做局部 IK 更新，不做完整全局搜索。

统计指标：

- 每步 FK solve time。
- 每步 sensitivity time。
- 每步 outer update time。
- 每步 accepted/rejected 次数。
- 姿态误差角。
- tip position 漂移。
- raw equilibrium residual。
- FK replay 几何误差。

阶段目标：

- 初始目标：每步 `1-5 s`。
- 中期目标：每步 `<1 s`。
- 远期目标：接近 Richter 的 `0.1-0.5 s` 局部更新量级。

## 9. 实验与验收规范

### 9.1 正确性门禁

每个 IK 候选结果必须执行 FK replay：

$$
z_{\mathrm{replay}} = \mathrm{FK}(q_{\mathrm{IK}}).
$$

然后比较：

- tip position error。
- rigid midpoint position error。
- rigid midpoint orientation error。
- centerline RMS error。
- centerline max pointwise error。
- final raw equilibrium residual。

早期可以允许 IK 目标误差较宽，但 FK replay 不能显示明显物理漂移。

### 9.2 速度指标

必须分开记录：

1. compile time：JAX 首次编译时间。
2. steady solve time：不含编译的净求解时间。
3. sensitivity time：雅可比/伴随灵敏度时间。
4. trial FK time：trial/line search 中 FK 时间。
5. I/O time：保存文件和绘图时间。

### 9.3 结果标签

所有加速结论必须分级：

- `speed-only`：速度提升，但物理一致性不足。
- `diagnostic-only`：只用于诊断，不可作为生产结果。
- `physically-comparable`：FK replay 与 baseline 几何可接受。
- `production-candidate`：速度、稳定性、正确性均通过当前门禁。

## 10. 主要风险与反方意见

### 10.1 Richter 指标不可直接类比

Richter 的 `20 ms` 是单个 BVP，不是完整 IK。其 `158 ms` 是 orientation Jacobian，不是全局逆静力学求解。若直接把完整 IK 目标设为 `20 ms`，会导致不现实的优化压力。

应对方式：

- 将 Richter 指标作为局部闭环更新 benchmark。
- 完整 IK 先以几十秒为工程目标，再逐步下降。

### 10.2 姿态残差可能引入新奇异

\(SO(3)\) log 在接近 \(\pi\) 时不光滑，磁体轴向约束也可能存在正负轴歧义。

应对方式：

- 采用 target continuation。
- 先实现 axis residual，再扩展完整 SO(3) residual。
- 对大角度目标分段求解。

### 10.3 隐式灵敏度可能病态

当 \(E_z^\top E_z\) 条件数极大时，adjoint 可能给出错误方向。

应对方式：

- 记录 condition number 和 stationarity norm。
- 使用 ridge 或截断 SVD。
- 对随机方向做 FD spot-check。
- 病态时 fallback 到数值 Jacobian 或更小 continuation step。

### 10.4 多磁体控制可能 rank 不足

多个磁体共享同一外部磁场系统，任务维度可能高于可控自由度。

应对方式：

- 记录 task Jacobian singular values。
- 明确可达性判断。
- 不可达任务应返回 best effort，而不是伪收敛。

### 10.5 包入口混乱可能污染实验

当前存在生产 FK、旧 FK、备份 FK 多套代码。若实验误导入旧模块，结果不可比。

应对方式：

- P0 优先统一 import。
- 每个实验 summary 记录 FK backend path 和 git commit。
- smoke test 验证导入路径。

## 11. 两个月阶段计划

### 第 1 周：P0/P1

目标：

- 完成 FK backend 统一方案。
- 实现 ObservationSpec。
- 实现 position、axis、so3 residual 的离线计算。

交付：

- `ik_optimize` 分支基础文档。
- 多点观测单元测试。
- 小型 FK replay notebook/script。

### 第 2-3 周：P2

目标：

- 完成多点隐式伴随灵敏度。
- 建立 FD audit。
- 给出多磁体姿态任务的 sensitivity correctness report。

交付：

- `compute_task_sensitivities_via_lm_adjoint`。
- Dense adjoint 与 FD 对照。
- 病态诊断输出。

### 第 4 周：P3

目标：

- 完成多点位姿 IK 外层。
- 跑通两磁体姿态目标和 tip position 混合目标。

交付：

- 新 IK runner。
- 至少 10 组目标的成功率、耗时和 FK replay 报告。

### 第 5 周：P4

目标：

- 实现 batch candidate runner。
- 多初值和 trial 并行化。

交付：

- 多 seed benchmark。
- batch 与 serial 等价性报告。
- 完整 IK 速度下降到分钟内或接近分钟级。

### 第 6 周：P5

目标：

- 矩阵自由伴随原型。
- 降低大网格显存压力。

交付：

- Dense vs matrix-free adjoint 对照。
- 大网格内存和耗时报告。

### 第 7-8 周：P6

目标：

- 建立 Richter-style local update benchmark。
- 将局部多磁体姿态更新推向秒级以下。

交付：

- Richter-style benchmark report。
- production-candidate IK 路线建议。
- 下一阶段是否引入 NN initializer 或 surrogate 的判断。

## 12. 推荐优先级

最高优先级：

1. 多点位姿 residual。
2. 多点隐式伴随 sensitivity。
3. FK replay 正确性门禁。
4. batch candidate 并行。

中优先级：

1. Coarse-to-fine 初始化。
2. 历史近邻初值库。
3. 矩阵自由伴随。

低优先级：

1. NN initializer。
2. 完整替换 FK 物理模型。
3. 单层 penalty IK 作为生产主路径。

## 13. 当前推荐的第一轮实验

实验名：

```text
ik_pose_observation_and_sensitivity_probe
```

实验目的：

- 不追求完整 IK 成功，先验证多点位姿观测和多点灵敏度是否可靠。

实验内容：

1. 固定三段磁体配置。
2. 使用当前 FK 生成若干平衡姿态。
3. 抽取 tip、三个 rigid midpoint 的位置和姿态。
4. 对电流方向做小扰动，比较 implicit sensitivity 与 FK finite difference。
5. 记录每个观测项的误差。

通过条件：

- position sensitivity 方向误差可接受。
- axis orientation sensitivity 能反映正确旋转方向。
- 病态 case 能被诊断，而不是静默通过。

失败处理：

- 如果 position sensitivity 已不稳定，先修复现有 `ik_diff.py`。
- 如果 position 稳定但 orientation 不稳定，优先检查 quaternion convention、SO(3) log 和 rigid midpoint pose convention。

