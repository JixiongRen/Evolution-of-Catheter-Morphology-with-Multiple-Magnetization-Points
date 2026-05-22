# eff_opt_exp_20260522110712_fk_correctness_runtime_matrixfree 实验记录

## 实验目的

本轮实验覆盖效率优化总体方案中的 P1、P2、P3：

- P1：建立宽松 FK golden、IK FK-replay 诊断基线，审查 task point convention，并拆解 dense FK/IK runtime 瓶颈。
- P2：移除低风险 runtime 开销，包括显式禁用 XLA 预分配、稀疏位姿保存、避免非必要 final Jacobian stats，并进行 float32/float64 对比。
- P3：实现 matrix-free FK 原型路径，使用 JVP/VJP 计算 `Jv` 和 `J.T v`，保留 dense fallback，并与 dense FK 做同配置对比。

## 基线与动机

当前 IK 单组求解耗时约 30-60 分钟。已有分析认为主要瓶颈可能来自 FK LM 内部 full Jacobian 构造、`J.T @ J` dense normal equation 和 dense direct solve。由于当前无法通过实验条件验证 IK 的物理正确性，本轮前期 correctness 采用宽松判据：重点确认 FK 整体姿态、弯曲方向和曲率趋势不发生明显偏移；IK replay 只作为诊断，不阻断 FK 提速探索。

## 实验方案

1. 审查旧 IK、FK engine、single-layer IK 的 tip extraction 和中心线约定。
2. 新增或使用低开销脚本生成 FK golden，并记录最后柔性节点、最后刚段相关点、centerline、residual、interface gap 和 smoothness。
3. 对已有 IK 输出中的少量 `I_A/x_m` 做 FK replay，标记 IK 自报结果与 FK oracle 的一致性。
4. 为 FK LM 增加 profiling，拆分 residual、Jacobian、`J.T @ J`、linear solve、line-search residual、stats/I/O 相关耗时。
5. 实施 P2 低风险清理，避免默认昂贵 stats 和高频中心线压缩保存。
6. 实现 P3 matrix-free FK 原型，使用 `JVP/VJP + CG` 解 damped normal equation，并保留 dense fallback。
7. 运行 small representative cases，对比 dense 与 matrix-free runtime、residual 和形态差异。

## 实现修改

- 统一当前 FK/IK 诊断中的 canonical task point：默认任务点为最后一段柔性段 distal node，同时在诊断脚本中保留最后刚段状态和 centerline end，避免比较对象漂移。
- 为 `ForwardKinematicsEngine` 增加 `linear_solver` 参数，默认保留 dense path，同时提供显式实验用的 `matrix_free_cg` path。
- 为 FK LM stats 增加 `stats_include_jacobian` 开关；非必要诊断不再默认保留 final dense Jacobian。
- 所有本轮实验命令显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；相关 IK 入口也增加同样的默认环境设置。
- 将单层 IK 中间位姿保存默认频率从每 10 次回调改为每 50 次回调。
- 旧 IK trial 评估默认不生成 final dense Jacobian；只有 trial 被接受并且下一轮 adjoint 需要时才补算 Jacobian。
- 新增 `experiments/efficiency_optimization/fk_correctness_runtime_matrixfree.py`，用于 FK golden、compile/steady 拆解、dense profiling、float32/float64 和 matrix-free 对照。
- 新增 `experiments/efficiency_optimization/ik_fk_replay.py`，用于读取已有 IK 输出并通过 FK oracle replay。

## 实验过程记录

- 2026-05-22 11:07：创建实验 ID、记录文件和日志数据目录。
- 2026-05-22 11:16：完成 smoke，对比 dense 与 naive matrix-free。
- 2026-05-22 11:29：完成 dense compile/steady 拆解和最小 matrix-free 对照。
- 2026-05-22 11:36：完成 float32/float64 对照。
- 2026-05-22 11:42：完成 5 个代表性 case 的 dense golden baseline。
- 2026-05-22 11:47：完成已有 IK 输出的 FK replay 诊断。

## 运行时间统计

- 本轮总耗时约 40 分钟。
- `dense_compile_steady`：3 个 case，`max_iter=4`，首轮调用 53.7-70.1 秒，steady-state 0.37-0.39 秒。
- `dense_golden_baseline`：5 个 case，`max_iter=80`，首轮调用 53.6-65.8 秒；未提前停止的 case steady-state 约 2.4-2.6 秒。
- `matrixfree_minimal`：最小 case 下 matrix-free 首轮 134.3 秒，steady-state 66.1 秒，明显慢于 dense。
- 下轮同量级实验可先按 40 分钟估算等待；若只做 steady-state 或单 case，可按 5-10 分钟估算。

## 实验结果

- Dense FK golden 已写入 `dense_golden_baseline/`。`straight_nomag_nogravity` 和 `mag_zero_mid` 在 7 次迭代收敛到 `normE=2.05e-6`；`gravity_only_mid` 在 80 次迭代后 `normE=6.76e-4`；`coil5_pos` 和 `mixed_balanced` 在 80 次迭代后仍有较高 residual，分别为 `0.262` 和 `0.0237`。
- Dense step profiling 显示 steady-state 单步中 residual、Jacobian、`J.T @ J` 和 dense solve 都是毫秒级；当前代表性规模的 dense Jacobian 大小为 `[459, 442]`，x64 下约 1.62 MB。
- Naive matrix-free CG 在最小 case 中明显失败：dense steady 约 0.25 秒，而 matrix-free steady 约 66.1 秒；低 CG 迭代下几何对比 `hard_fail`。
- float32 将 Jacobian 存储减半到约 0.81 MB，steady-state 约有 1.18-1.21x 加速，但首轮编译未改善；gravity case residual 从 x64 的 `0.0103` 变为 float32 的 `0.0550`。
- IK FK-replay 发现明显不一致：`ik_out/stage1_optimized_v2_20260324` 自报 tip 与 FK replay tip 相差约 110.6 mm；centerline end 相差约 114.3 mm。

## 数据分析

- 当前 FK 在 steady-state 下并不慢。用户观测到的 5-10 分钟 FK 大概率来自首轮 JAX compile、进程重启、engine 重建、shape/static config 变化或 IK 外层重复触发编译，而不是 dense normal equation 本身。
- 对当前问题规模，dense Jacobian 和 dense solve 不是主瓶颈；强行 matrix-free 会把一次 dense AD/solve 替换为多次 JVP/VJP，当前 JAX 实现下更慢且更难稳定。
- float32 可以降低显存，但在当前 case 上没有带来足够稳健的总耗时收益，且 gravity case residual 明显变差，因此暂不适合作为默认。
- IK FK-replay 不通过，说明已有 IK 成功结果不能直接作为物理正确性证据。前期应继续以 FK golden 的整体姿态和曲率趋势作为主要正确性门禁，IK replay 只作为风险标签。
- 下一步的高收益方向应从“重写 dense solve 为 matrix-free”转为“保证同一物理任务复用同一已编译 FK/IK graph，减少外层流程对编译缓存和 final Jacobian 的破坏”。

## 风险与回归

- matrix-free 原型未达到预期；若后续继续专项优化，需要先证明 JVP/VJP 组合的 steady-state 成本低于 dense Jacobian。
- FK golden 当前只作为“整体姿态靠谱”的代理，不代表实验真值。
- IK replay 不通过时，仅说明 IK 可信度不足或 convention 存疑，不阻断 FK 加速方向。
- 需要避免把首轮 JAX compile time 和 steady-state runtime 混在一起解读。
- 本轮 matrix-free 已证明在当前实现下不应作为默认路径；保留代码仅用于后续专项实验和 fallback 对照。
- 旧 IK 中被接受 trial 补算 Jacobian 会保留必要 adjoint 信息；若后续发现 adjoint 缺失，应优先检查接受分支是否经过 `_ensure_lm_jacobian_stats`。

## 实验结论

- P1 完成：建立了 dense FK golden、compile/steady 拆解、dense step profiling、IK FK-replay 诊断和 task point convention 记录。
- P2 完成：显式关闭 XLA 预分配、降低中间位姿保存频率、避免非必要 final Jacobian stats，并完成 float32/float64 对照。
- P3 完成原型验证但不满足启用条件：当前 `matrix_free_cg` 明显慢于 dense 且低迭代下形态不稳定，因此 dense path 继续作为默认；matrix-free 仅保留为显式实验路径。
- 本轮最重要结论：当前首要瓶颈不是 FK steady-state dense 线性代数，而是编译和流程复用问题。后续效率优化应围绕 persistent engine、固定 shape/static config、预编译和减少外层重复 FK/Jacobian 构造展开。

## 下一步计划

- 建立真实 IK 流程的 compile/steady profile，确认哪些外层步骤在重复触发 JAX compile。
- 将 IK/FK 入口统一到持久化 engine 和固定 shape 的执行路径，避免每个 attempt 或 stage 重建编译图。
- 针对旧 IK，继续削减被拒 trial、line-search 和 final stats 的无用 Jacobian 计算。
- 对 `coil5_pos`、`mixed_balanced` 等 residual 较高 case 做单独 FK 收敛性分析，区分物理不可达、阻尼策略问题和初值问题。
- 暂不默认启用 float32；后续可在无重力或弱载荷 case 中作为显存专项继续评估。
