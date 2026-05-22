# FK/IK 效率优化总体方案

创建日期：2026-05-22

本文档是单层 IK 求解效率优化的总体方案，不是某一轮大实验的实验记录。后续每一轮大实验仍需按照规则创建独立的实验编号、实验记录文件和日志数据目录。

## 1. 总体目标

当前 IK 工作流单组目标点求解耗时约为 30-60 分钟。优化目标如下：

- 目标耗时：约 1 分钟/组。
- 可容忍上限：约 3 分钟/组。
- 质量要求：不能明显牺牲物理平衡、目标点精度、接口一致性、形态平滑性和电流可行性。

总体路径为：

1. 优化 FK 求解耗时。
2. 重构 FK 线性化方式，以及 IK 使用 FK 结果和灵敏度的方式。
3. 神经网络暂不作为主求解器，仅作为后续初值估计或 attempt 筛选的辅助工具。

## 2. 当前诊断

当前耗时问题大概率无法通过简单调参解决。主要高成本环节可能包括：

- FK LM 迭代中显式构造 dense Jacobian。
- 通过 `J.T @ J` 构造 dense normal equation。
- 平衡求解器内部使用 dense direct solve。
- IK 外层重复调用 FK，尤其是对 `x` 或导管伸出量相关变量做有限差分灵敏度估计。
- 不必要的诊断输出和高频数据保存，尤其是每次迭代保存完整 centerline 或压缩位姿数据。
- JAX 输入结构或 shape 不稳定导致重复编译。
- 初值质量不足，导致 attempt 次数和外层迭代数过多。

此前短期 GPU/runtime 实验已经说明：

- 实验子进程应显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- `pose_save_every` 应默认保持稀疏，避免每次 callback 都写入 `np.savez_compressed`。
- 短实验中 float32 与 float64 结果接近，但仍需在中等长度代表性案例上验证。

目前最重要的正确性风险是建模一致性。尤其是受控 tip 或任务点定义必须在旧 IK、FK engine 和单层 IK 中保持一致。否则求解器即使变快，也可能是在优化不同的物理点。

## 3. 建模假设

目标对象是 permanent magnet-based magnetic continuum robot。建模层级应保持如下原则：

- 最终验证和最终 IK 精修仍使用高保真 Cosserat 或 GL-style 平衡求解。
- 低阶近似、粗网格或 continuation 只用于快速寻找可行 basin，不作为最终物理解的替代。
- 永磁体段应作为具有固定 body-frame 磁化矢量的物理驱动区域处理，其 world-frame 力和力矩必须与局部坐标系约定和磁场模型一致。

效率优化过程中必须保留以下物理检查：

- 所有求解路径使用同一个 tip 或任务点定义。
- 磁体中心、刚段端点、柔性段节点的几何约定明确。
- `m_body` 到 world frame 的映射一致。
- 磁场插值位于标定范围内，或明确报告越界行为。
- 最终成功不能只看 tip error，还必须满足平衡残差和形态质量要求。

## 4. FK/IK 正确性验证与宽松门禁

当前缺少直接实验条件验证 IK 输出，因此后续优化采用 FK oracle 代理验证。这里的验证不应在前期卡得过严：FK 目前只能被认为“整体求解效果比较靠谱”，因此前期主要检查同配置下 FK 的整体姿态、弯曲方向和曲率变化趋势是否大致一致。IK 输出暂不作为强可信对象，IK replay 主要用于诊断和标记可信度，不作为早期算法探索的硬阻断条件。

### 4.1 分阶段正确性契约

- 在进入主要优化前，必须明确 canonical task point。验证阶段应同时记录最后柔性节点、最后刚段远端或刚段代表点，直到最终确认哪一个是物理受控点。
- FK golden 数据由当前可信 dense FK 生成。后续新 FK 路径、matrix-free 路径或 reduced-space IK 路径都应与该 golden 对比，但前期只要求整体姿态和曲率变化曲线不发生明显偏移。
- IK 成功不能只看内部 `pos_err_norm_mm` 或自报 `p_tip_m`。IK 输出的 `I_A` 和 `x_m` 应重新输入 FK 做 replay，但前期 replay 结果只用于诊断、分类和选择下一步，不直接否定本轮 FK 加速实验。
- 速度结论分两类记录：`FK speedup` 只要求 FK 姿态大致一致；`IK success speedup` 必须额外通过 FK replay 和基本物理质量检查。
- 前期 hard fail 只针对明显错误：FK 姿态整体翻转、弯曲方向明显相反、曲率曲线严重失真、平衡残差大数量级恶化、出现不可解释的形态断裂或数值发散。

### 4.2 FK golden 基线

FK 基线应使用低开销入口生成：

- 首选 `ForwardKinematicsEngine.solve()` 或 `solve_with_stats()`。
- 常规回归默认使用 `solve()`，只额外计算必要 residual 和几何指标。
- 不默认保存完整 Jacobian，不默认绘图，不在 tight loop 中保存大量 NPZ。
- `fk.py` CLI 可用于人工调试，但不作为自动回归主入口，因为它可能引入额外 I/O 和 plotting 开销。

FK golden 每个 case 至少记录：

- case id、git commit、FK module、JAX x64 状态、device。
- `M_list`、`L_protrude`、`L1_dim`、currents、`m_body_list`、gravity、scales。
- `ok`、`normE`、`cost`、`max_abs_E`。
- canonical task point、最后柔性节点、最后刚段或刚体代表点。
- rigid midpoint poses、centerline 下采样点、interface gap、smoothness。
- cold first-call time、steady solve time、residual evaluation time。

`z_bar` 只作为 debug artifact，不作为主要比较对象，因为 nondim scale、内部 packing 或状态布局变化可能造成假差异。

### 4.3 FK 对比容差

初始容差采用宽松分级规则，后续根据实验结果收紧：

- 同 dense x64 路径重复运行仍应较稳定，可保留较严格检查，但这主要用于发现非确定性或实现 bug。
- 新 FK 算法对 dense golden 的前期主指标是形态相似性：重采样 centerline 后，弯曲方向一致、整体曲率峰值位置相近、曲率变化曲线形状相似。
- 建议前期 hard fail 条件：centerline 平均偏移超过 `5-10% * L_ref`，最大偏移超过 `15% * L_ref`，曲率曲线相关性低于 `0.8`，或主要弯曲方向相反。
- task point 偏移前期只作为参考：`3-5 mm` 以内可接受，`5-10 mm` 记为 warning，超过 `10 mm` 再结合形态和 residual 判断是否 hard fail。
- residual 前期不要求严格倍率一致，但如果 `normE_new` 比 `normE_ref` 恶化超过一个数量级，或绝对值进入明显不可信区间，应标记为 hard fail。
- interface gap 和 smoothness 前期作为 warning 指标，只有出现明显断裂、尖峰或非物理折角时才阻断。

### 4.4 IK 代理验证

IK 修复或优化建议执行以下低成本代理验证，但前期不把 IK replay 作为硬约束：

- FK replay：读取 IK 输出的 `I_A`、`x_m` 和配置，重建同配置 FK 并严格重算。记录 `e_ik=||p_ik-p_des||`、`e_fk=||p_fk-p_des||`、`replay_gap=||p_fk-p_ik||`、`normE_fk`、`ok_strict` 和 FK iteration count。
- closed-loop consistency：先采样一批 `(I, x)`，用 FK 生成 reachable target，再让 IK 反求该 target，最后用 FK replay 验证 `FK(IK(p_fk))` 是否回到目标。由于系统冗余，不要求反求得到的 `I` 和 `x` 与原始采样完全一致。
- 目标可达性 sweep：对真实目标先做小规模 FK sweep，估计 `d_min = min ||p_fk-p_des||`。如果目标明显在采样可达包络外，IK 不应报告成功，而应输出 best-effort 和 reachability 或 bound-stall 诊断。
- sensitivity check：对 implicit adjoint 或 matrix-free sensitivity，在 well-conditioned 小 case 上用 directional finite difference 验证；near-singular case 可放宽容差，但必须标记 conditioning。

建议初始判据：

- 如果 IK 自报误差小于 `1 mm`，但 FK replay 误差大于 `2 mm`，该 IK run 标记为“不可信 IK 成功”，但不阻断 FK 加速实验。
- 如果 `replay_gap` 大于 `0.5-1 mm`，判定存在状态解释、task point 或平衡一致性风险；前期只记录风险并辅助定位。
- 如果 FK replay 的 `normE > 1e-1`，或 `L1_clipped=true` 且误差未收敛，判为物理无效。
- `normE <= 1e-2` 作为绿色区间；`1e-2..1e-1` 作为黄色区间；`>1e-1` 作为红色区间。

### 4.5 建议基线 case

第一批 FK golden case 应覆盖低成本 per-commit 回归和更重的 nightly/performance 回归：

| 层级 | case | 配置 |
| --- | --- | --- |
| per-commit | `straight_nomag_nogravity` | `M_list=3,2,2`，`L_protrude=0.10`，gravity off，magnetics off |
| per-commit | `gravity_only_mid` | 同上，gravity on |
| per-commit | `mag_zero_mid` | magnetics on，mixed `m_body_list`，currents 全零 |
| per-commit | `coil0_pos` | currents `[15,0,0,0,0,0,0,0]` |
| per-commit | `coil0_neg` | currents `[-15,0,0,0,0,0,0,0]` |
| per-commit | `coil5_pos` | currents `[0,0,0,0,0,15,0,0]` |
| per-commit | `mixed_balanced` | currents `[10,-8,6,-4,2,-6,8,-10]` |
| per-commit | `ik_snapshot_input` | 使用已有 IK 输出的 `(I, x)` 作为 FK 输入，不以 IK 是否成功作为 oracle |
| nightly | `short_protrude` | `L_protrude=0.075`，mixed current |
| nightly | `long_protrude` | `L_protrude=0.18`，mixed current |
| nightly | `mesh_5` | `M_list=5,5,5`，mixed current |
| weekly/perf | `mesh_8` | `M_list=8,8,8`，mixed current |

默认几何建议复用当前 stress 配置：`flex_lengths_tail=0.03,0.03`、`rigid_lengths=0.003,0.003,0.003`、`axis_body=1,0,0`、`p0=0,0,-0.05`、mixed `m_body_list=0,0,-0.005301;0,0,0.005301;0.005301,0,0`。真实 IK target 只用于覆盖工作空间，不作为 oracle。

## 5. 主策略

推荐主线如下：

1. 正确性校准和 FK/IK 宽松 correctness gate。
2. dense FK profiling。
3. matrix-free FK。
4. implicit adjoint sensitivity。
5. reduced-space IK。
6. coarse-to-fine 以及 load/target continuation。
7. 批量回归和 fallback 策略固化。

单层 penalty IK 可以保留为 fallback 或研究分支，但主生产路径建议转向 reduced-space IK：

```text
minimize task_cost(I, x, z)
subject to E(z, I, x) = 0
```

该路径把平衡方程作为内层硬约束，而不是允许 penalty formulation 用非物理解换取更低的 tip error。

## 6. 成功指标

每一轮大规模效率优化实验至少应记录：

- 总 wall time。
- 可测量时的 compile time。
- steady-state solve time。
- FK solve count。
- IK outer iteration count。
- FK LM iteration count。
- final tip error。
- raw equilibrium residual。
- interface gap。
- smoothness 或 curvature anomaly metric。
- current utilization 和 saturation。
- 可用时记录 control Jacobian rank 或 singular values。
- 使用 float32 还是 float64。
- 本次统计是否包含 first-call compilation。
- FK golden 对比误差，重点关注整体姿态和曲率变化趋势。
- IK final control 的 FK replay 误差，前期作为诊断和可信度分类。
- task point convention 是否一致。
- failed case 的失败分类。

speedup 需要分层解释。前期 `FK speedup` 只要求 FK 姿态和曲率趋势大致一致；`IK success speedup` 才要求 FK replay、目标误差和基本物理质量同时可接受。

## 7. 两个月路线图

| 阶段 | 时间 | 目标 | 主要产物 | 验收标准 |
| --- | --- | --- | --- | --- |
| P0 | 立即 | 建立 workflow 和总体方案 | skill、目录、本文档 | 后续实验具备统一 ID、记录、日志、耗时统计和本地 dev 提交规则 |
| P1 | 第 1 周 | 宽松正确性门禁、基准拆解与建模一致性 | correctness baseline + dense FK/IK profiling 实验 | FK golden、IK replay 和 10-12 个固定 case 耗时分解全部记录 |
| P2 | 第 2 周 | 低风险 runtime 清理 | 配置和 I/O 优化 | 不降低收敛质量的前提下获得 2-5x 提速 |
| P3 | 第 3-4 周 | matrix-free FK 原型 | 带 dense fallback 的 JVP/VJP Krylov FK 路径 | 代表性 case 中 FK 单步成本下降 5-10x |
| P4 | 第 4-5 周 | implicit adjoint sensitivity | 不依赖完整 `J` 和 FK 有限差分循环的灵敏度路径 | IK 线性化成本相对 FK 平衡求解变小 |
| P5 | 第 5-6 周 | reduced-space IK | 基于 `I` 和 `x` 的物理变量 IK | 代表性 near/stress target 可稳定收敛 |
| P6 | 第 6-7 周 | continuation 与初始化 | coarse-to-fine mesh、load continuation、target continuation | 典型 IK 迭代数降到 50-80 以下 |
| P7 | 第 8 周 | 回归与稳定性验证 | 10 组以上极端场景 benchmark suite | FK 姿态一致性通过后评估 p50 是否接近 1 分钟，p95 或极端 case 是否控制在约 3 分钟内 |

## 8. 阶段细化

### P1：宽松正确性门禁、基准拆解与建模一致性

目标：

- 建立 FK golden 和 IK FK-replay 诊断基线。
- 在修改求解器结构前找准真实瓶颈。
- 确保所有 IK 路径优化同一个物理任务点。

任务：

- 审查旧 IK、FK engine 和单层 IK 中的 tip extraction。
- 定义唯一的 canonical task point convention。
- 在 canonical convention 确认前，同时记录最后柔性节点和最后刚段相关点，防止比较对象漂移。
- 用当前 dense FK 生成第一批 golden baseline。
- 对已有成功 IK 输出选取少量 `I_A/x_m` 快照，做 FK replay，验证 IK 自报结果与 FK oracle 是否一致。
- 审查刚段端点、磁体中心和柔性节点约定。
- 增加或运行 profiling，将 residual evaluation、Jacobian construction、`J.T @ J`、dense solve、line-search residual call、stats packing 和 I/O 分开计时。
- 将 first-call compile time 与 steady-state runtime 分开统计。

建议实验：

- `exp_name=fk_correctness_dense_baseline`
- 实验记录：`docs/efficiency_optimization_experiment/{ID}_fk_correctness_dense_baseline_record.md`
- 日志数据：`docs/efficiency_optimization_experiment/{ID}_fk_correctness_dense_baseline_logs_data/`

决策规则：

- 如果 FK 姿态一致性明显不通过，先修正 convention、FK replay 或诊断逻辑，不进入 FK 提速结论。
- 如果只是 IK replay 不通过，前期记录为 IK 不可信或目标不可达风险，不阻断 FK 提速方向。
- 如果 dense Jacobian 加 dense solve 是主瓶颈，下一步进入 matrix-free FK。
- 如果其他部分占主导，则优先优化真实热点。

### P2：低风险 runtime 清理

目标：

- 在重构算法前先移除明确的可避免开销。

任务：

- 所有实验启动命令显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 默认保持稀疏 `pose_save_every`。
- 除非实验目标就是诊断输出，否则不要在紧密循环中写入压缩 centerline。
- 运行中等长度 float32/float64 对比实验。
- 尽可能保持 JAX PyTree shape 稳定。
- 除非显式需要，不计算昂贵的 final Jacobian stats。

验收标准：

- 可比 case 获得 2-5x runtime 改善。
- final tip error、raw equilibrium residual 和形态质量保持可接受。

### P3：matrix-free FK

目标：

- 停止把 dense LM system 作为默认 FK 路径。

任务：

- 使用 JVP 实现 `J v` 的 matrix-free linear operator。
- 使用 VJP 实现 `J.T v` 的 matrix-free adjoint operator。
- 根据条件数和稳定性选择 CG、LSMR、GMRES 或 MINRES 求解 damped normal 或 augmented system。
- 加入简单 preconditioner，例如 diagonal、block diagonal 或 previous-step curvature。
- 保留当前 dense path 作为正确性 fallback。
- 在相同 seed 和 control 下对比 matrix-free 与 dense FK。

验收标准：

- 代表性 case 中每个 FK LM step 成本下降 5-10x。
- final equilibrium residual 和任务点几何与 dense FK 在容差内一致。
- 显存压力下降，极端 case 不再轻易打满显存。
- 所有 per-commit FK golden case 通过对比，失败时自动或人工 fallback 到 dense path。

### P4：implicit adjoint sensitivity

目标：

- 让 IK 灵敏度计算更便宜且保持物理一致。

任务：

- 使用平衡条件 `E(z, I, x) = 0`。
- 通过 implicit differentiation 计算 sensitivity，替代重复 FK 有限差分。
- 不再要求 `lm_stats` 保存完整 `J`。
- 将 protrusion 或 `x` sensitivity 纳入同一框架。
- 尽量复用 matrix-free FK operator。

验收标准：

- IK outer linearization 不再为小扰动执行多个完整 FK solve。
- 灵敏度计算成本可预测，且显著低于旧有限差分路径。
- well-conditioned 小 case 中 directional derivative 与有限差分一致；near-singular case 必须报告 conditioning。

### P5：reduced-space IK

目标：

- 让主 IK 路径在严格平衡约束下优化物理控制变量。

任务：

- 定义 reduced variables，优先考虑 currents `I` 和 protrusion 或 segment parameter `x`。
- 使用 FK 作为带 warm start 的内层平衡求解器。
- 使用 trust-region 或 line-search globalization。
- 在物理变量上处理 current/protrusion bounds，避免过度依赖 saturated parameter transform。
- 显式记录 current utilization 和 saturation。
- 保留单层 penalty IK 作为对照 fallback。

验收标准：

- 代表性 near targets 可稳定求解。
- stress targets 能收敛，或能给出明确 reachability diagnosis。
- tip error 改善时 equilibrium residual 不恶化。
- 正式宣称 IK 成功时必须通过 FK replay；前期 prototype 可只记录 replay 结果，不把 IK replay 作为 hard fail。

### P6：coarse-to-fine 与 continuation

目标：

- 减少迭代数，避免落入差的局部分支。

任务：

- 先用 coarse `M_list` 寻找 basin。
- 将 coarse solution prolong 到 finer mesh。
- 必要时使用 magnetic field 或 current magnitude 的 load continuation。
- 对远距离目标使用 target continuation。
- 只在最终精修阶段使用更严格 tolerance。

验收标准：

- 典型 total IK iteration count 降到 50-80 以下。
- fine solve 初始点足够接近，只需要少量昂贵迭代。
- branch jump 和不自然形态减少。

### P7：回归套件

目标：

- 证明优化后的求解器不是只对单个手选 case 有效。

任务：

- 构造至少 10 个极端但有意义的目标点。
- 包含混合磁化模式：

```text
m_body_list = "0,0,-0.005301;0,0,0.005301;0.005301,0,0"
```

- 覆盖 near、diagonal、lateral、high-curvature、boundary 和 negative-axis targets。
- 在可行时同时运行优化求解器和 dense 或旧求解器 baseline。
- 绘制 convergence、target point、start point、final point 和 pose curves。

验收标准：

- 如果算法改造达到预期，p50 runtime 接近 1 分钟。
- reachable hard cases 的 p95 runtime 或极端 case runtime 控制在约 3 分钟内。
- 所有成功 case 都满足物理质量容差。
- 所有成功 FK case 通过姿态一致性检查；正式宣称 IK 成功的 case 需要通过 FK replay。
- 失败 case 必须分类为 reachability、conditioning、numerical 或 implementation failure。

## 9. 第一轮正式实验设计

第一轮大实验应先建立宽松 correctness baseline，再进行 profiling baseline，而不是直接优化。

实验名称：

```text
fk_correctness_dense_baseline
```

实验组：

| 实验组 | 配置 | 目的 |
| --- | --- | --- |
| Convention audit | 同一 `z_bar` 同时提取最后柔性节点、最后刚段相关点和 centerline end | 固定 canonical task point，发现旧 IK、FK engine、single-layer IK 的定义差异 |
| FK golden baseline | per-commit FK case | 生成当前 dense FK compact JSON 和可选下采样 centerline NPZ |
| IK FK replay | 选取已有成功或代表性 IK 输出的 `I_A/x_m` | 诊断 IK 自报结果能否被 FK replay 复现 |
| Closed-loop reachable | 少量 FK 采样生成 target，再由 IK 反求 | 低成本验证 IK 能否回到 FK 生成的可达目标 |
| FK micro | `M_list=3,3,3`、`5,5,5`、`8,8,8` | 测量 dense FK 的规模扩展趋势 |
| FK repeated call | 相同 shape，改变 currents | 分离 compile、warm-start 和 steady-state 行为 |
| FK repeated protrusion | 尽量保持相同 shape，改变 `x` 或 protrusion | 观察 recompilation 和有限差分成本 |
| IK macro near | near target，较低 outer cap | 测量容易 case 的 IK overhead |
| IK macro stress | far 或 diagonal target，较高 outer cap | 测量困难 case 中 FK call amplification |

必须统计的计时项：

- Residual evaluation time。
- Jacobian construction time。
- `J.T @ J` construction time。
- Linear solve time。
- Line-search residual time。
- Stats and diagnostics time。
- 可测量时的 I/O time。
- Compile time 或 first-call time。
- 不含 compile 的 steady-state time。

dense FK 瓶颈确认条件：

- `jacobian + J.T @ J + linear solve` 至少占 steady-state FK time 的 70%。
- FK time 至少占 IK wall time 的 80%。
- FK per-step time 随 `M_list` 增大明显超线性增长。
- 外层 9D IK solve time 相比 FK time 很小。

如果这些条件不成立，下一轮实验应先处理真实主瓶颈，而不是直接进入 matrix-free FK。

## 10. 本方向工程规则

- 每轮大规模效率优化实验必须使用一个唯一 ID：`eff_opt_exp_YYYYMMDDHHMMSS`。
- 每轮大规模效率优化实验必须使用同一个 ID 创建一个 record 文件和一个 log-data 目录。
- 本方向的记录和日志必须位于 `docs/efficiency_optimization_experiment/`。
- 每轮大实验必须记录 wall-clock runtime。
- 下一轮实验的等待估计应参考上一轮可比实验的真实耗时。
- 长时间等待期间保持低频输出和低频检查。
- 每轮大实验结束后，将相关变更提交到本地 `dev` 分支。
- 不 push。
- 不提交无关的大型生成数据，除非它是本轮实验记录的必要组成部分。
- 任何 FK 算法路径替换、IK 修复、灵敏度重构或初始化策略修改，都应运行对应级别的 correctness smoke 或 regression。
- 前期 correctness 采用宽松分级判据：FK 结果以整体姿态和曲率趋势相似为主；IK replay 以诊断为主。
- 未通过 FK 姿态一致性检查的结果不能用于宣称 FK 有效提速；未通过 IK FK replay 的结果不能用于宣称 IK 成功，但可用于继续分析和改进。
- 文档统一使用中文。
- 代码注释和 docstring 统一使用英文。

## 11. 风险清单

| 风险 | 影响 | 缓解方式 |
| --- | --- | --- |
| tip 定义不一致 | 快速求解器优化了错误物理点 | 在主要求解器改造前解决 |
| golden data drift | baseline 自身随未记录配置变化漂移 | golden 记录 git commit、完整配置、scales、device 和 x64 状态 |
| IK 自报成功但 FK replay 失败 | penalty solver 给出非物理解或错误 task point | 正式宣称 IK 成功时必须通过 FK replay；前期只作诊断 |
| 旧路径本身存在 convention 错误 | golden 可能固化错误约定 | 第一轮先做 convention audit，同时记录多个 candidate task point |
| 弱验证误判 | 只看 tip error 忽略残差和形态 | 同时检查 residual、interface gap、smoothness、current saturation |
| matrix-free solve 不稳定 | FK 可能因线性求解失败而不收敛 | 保留 dense fallback，并持续对比 residual |
| float32 数值漂移 | 可能出现隐性质量退化 | 使用代表性 float32/float64 benchmark |
| coarse mesh 分支不一致 | fine refinement 进入错误分支 | prolongation 前加入形态和残差检查 |
| saturated current transform | 导数消失并拖慢 IK | 优先使用物理 bounds 和 trust-region step |
| 过度日志输出 | runtime 被 I/O 主导 | 默认稀疏保存 |
| 重复编译 | runtime 统计失真 | 分离 compile 和 steady-state timing |
| benchmark 过小 | 对真实场景泛化差 | 使用至少 10 个多样化 stress cases，并包含 FK replay |

## 12. 预期里程碑

- 第 1 周结束：建立 FK golden 和 IK FK-replay 诊断基线；明确真实瓶颈；tip 和 magnet convention 问题被记录或修正。
- 第 2 周结束：低风险 runtime 清理完成；代表性中等 case 有 float32/float64 证据。
- 第 4 周结束：matrix-free FK 原型能产出与 dense FK 可比的平衡解。
- 第 5 周结束：IK 核心 sensitivity 不再依赖昂贵的 FK 有限差分循环。
- 第 6 周结束：reduced-space IK 可在严格平衡约束下求解代表性 case。
- 第 7 周结束：coarse-to-fine 策略能降低 hard targets 的迭代数。
- 第 8 周结束：benchmark suite 在 FK 姿态一致性通过前提下给出 1 分钟目标是否现实、3 分钟上限是否达成的结论。

## 13. 立即下一步

启动第一轮大实验：

```text
exp_name = fk_correctness_dense_baseline
```

启动前必须先创建：

```text
docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_fk_correctness_dense_baseline_record.md
docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_fk_correctness_dense_baseline_logs_data/
```

第一轮实验结论应决定下一轮进入以下哪条路径：

- matrix-free FK implementation。
- I/O 和 diagnostics cleanup。
- magnetic interpolation optimization。
- IK sensitivity restructuring。
- initialization 和 continuation improvement。
