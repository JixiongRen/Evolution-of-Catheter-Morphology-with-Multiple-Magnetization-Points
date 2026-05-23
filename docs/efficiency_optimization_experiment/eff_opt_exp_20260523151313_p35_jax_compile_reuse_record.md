# eff_opt_exp_20260523151313_p35_jax_compile_reuse 实验记录

## 实验目的

修复 P3 后暴露的 JAX 编译和 FK/IK 流程复用问题，使真实 IK 工作流不再被重复 first-call compile、engine 重建、shape/static config 漂移和无用 Jacobian/stats 计算主导。

## 实验方案

- 审查 FK engine、cached solver、旧 IK 和单层 IK 的调用路径。
- 建立真实 IK macro profiling，拆分 FK first-call、steady solve、FD-x、line-search trial、final stats/Jacobian、I/O。
- 修复可避免的 engine 重建和 JIT wrapper 重建。
- 增加 warm-up 或 compile-cache 稳定路径，保证固定 shape 的 residual/Jacobian/solve 先预热。
- 运行 FK repeated-call/rebuild probe 和短 IK macro，验证重复调用不再反复出现 50-70s first-call。
- 与 FK golden 做宽松形态一致性对照。

## 实现修改

- 将 `forward_kinematics_optimized/equilibrium_solver_nondim.py` 和 `single_layer_ik/forward_kinematics_optimized_bak/equilibrium_solver_nondim.py` 中的 residual、Jacobian、JVP、VJP JIT wrapper 提升到模块级，避免每次 `ForwardKinematicsEngine` 重建时重新创建 transformed function。
- 为 `ForwardKinematicsEngine` 增加 `warm_up()`，用于在固定 shape/static config 下预热 residual、Jacobian 和单步 LM solve，同时不污染 warm-start state。
- 为 `inverse_kinematics/ik_multisite_lm_opt.py` 和 `single_layer_ik/inverse_kinematics_bak/ik_multisite_lm_opt.py` 增加 `--runtime_profile`、`--warmup_fk_compile` 和 `--warmup_ik_adjoint`。
- 为 IK macro profile 拆分 `engine_build`、FK solve、FK observe、FD-x FK、adjoint sensitivity、accepted-trial Jacobian stats 和 I/O。
- 旧 IK 和 single-layer 备份 IK 的 FD-x 分支改为 `return_stats=False`，不再构造未使用的 LMStats。
- 被拒 trial 继续使用 `stats_include_jacobian=False`；只有 accepted trial 才延迟补算 dense FK Jacobian 供下一轮 adjoint 使用。
- final snapshot 改为 `stats_include_jacobian=False`，避免最终保存结果时生成无用 dense Jacobian。
- 顶层 IK 脚本补充 repo root 和模块目录到 `sys.path`，使 `python inverse_kinematics/ik_multisite_lm_opt.py ...` 可直接运行，不依赖额外 `PYTHONPATH`。
- 新增 `experiments/efficiency_optimization/p35_jax_compile_reuse_probe.py`，用于同进程 FK 首次、重复调用和 engine rebuild 后的编译复用验证。

## 实验过程记录

- 2026-05-23 15:13：创建 P3.5 实验 ID、记录文件和日志目录。
- 2026-05-23 15:26：运行 FK reuse probe，验证同进程 engine rebuild 后是否复用已编译 graph。
- 2026-05-23 15:29：运行短 IK macro profiling，确认 warm-up 后 FK solve、FD-x、final stats 和 I/O 的耗时分布。
- 2026-05-23 15:35：补充直接运行路径修复、FD-x stats 清理、final no-Jacobian 和可选 IK adjoint 预热。
- 2026-05-23 15:36：运行 3 个 FK golden recheck case，与上一轮 `eff_opt_exp_20260522110712` dense golden 做宽松一致性对比。
- 2026-05-23 15:37：运行 2 轮短 IK macro，验证 `--warmup_ik_adjoint` 的首轮 sensitivity 复用效果。
- 2026-05-23 15:40：完成静态检查、数据整理和报告更新。

## 运行时间统计

- 本轮总耗时约 30 分钟。
- 当前执行环境无法初始化 CUDA，JAX 打印 `cuInit(0) failed: Unknown CUDA error 100` 后使用 CPU backend。因此本轮数值可用于验证编译复用和流程开销结构，但 GPU 绝对耗时需要在 CUDA 正常环境中复测。
- FK reuse probe：首次 solve `14.236 s`；同一 engine repeat solve `0.121 s`；重建 engine 后 solve `0.123 s` 和 `0.118 s`；engine build 本身约 `0.0005-0.0006 s`。
- 短 IK macro 初版：FK warm-up `14.133 s`；warm-up 后 init/trial/final FK solve 合计 `0.368 s`；FD-x 两次 FK 合计 `0.215 s`；adjoint sensitivity first call `1.653 s`；I/O 合计约 `0.548 s`。
- 短 IK macro 加 adjoint 预热后：FK warm-up `14.547 s`，IK adjoint warm-up `1.804 s`；首轮 sensitivity 直接复用，记录耗时约 `1.3e-6 s`；第二轮 adjoint steady `0.0189 s`。
- 3 个 FK golden recheck case：`straight_nomag_nogravity` first/steady 为 `15.748/0.208 s`；`mag_zero_mid` 为 `13.503/0.223 s`；`mixed_balanced` 为 `14.677/1.958 s`。

## 实验结果

- FK reuse probe 证明模块级 JIT wrapper 修复有效：同一进程内，首次 solve 后重复 solve 和 rebuild-engine solve 都稳定在约 `0.12 s`，没有再次出现上一轮 GPU 观测到的 `50-70 s` 级 first-call compile。
- FK reuse probe 的几何输出完全一致：`centerline_end=[0.0996052176, 0, -0.05]`，`last_flex_node=[0.0966077403, 0, -0.05]`，`normE=0.001013`。
- FK golden recheck 覆盖 `straight_nomag_nogravity`、`mag_zero_mid`、`mixed_balanced`。与上一轮 dense golden 对比，端点和最后柔性节点偏移为 `0` 到 `5.5e-14 m`，残差差异为 `6e-18` 到 `3.3e-14`，通过宽松 FK 姿态一致性门禁。
- 短 IK macro 加 adjoint 预热后，2 轮 outer 均接受 trial，误差从 `22.430 mm` 降到 `19.896 mm`，`normE` 从 `3.368e-3` 降到 `2.051e-4`。该 case 只用于 profiling，不作为 IK 成功证据。
- FD-x 清理后仍能正常运行；final snapshot 不再触发额外 dense Jacobian。`jacobian_stats` 只在 accepted trial 上出现，2 轮合计 `0.0207 s`。
- 直接运行 `python inverse_kinematics/ik_multisite_lm_opt.py --help` 已通过，不再需要手动设置 `PYTHONPATH`。

## 数据分析

- P3.5 的主要假设成立：当前代表性规模下，steady-state dense FK 不是主要瓶颈；大开销来自进程冷启动、JAX first-call compile、未预热的 sensitivity transform，以及外层流程是否破坏编译复用。
- 模块级 JIT wrapper 是本轮最关键修复。此前每次 engine 初始化都会创建新的 `jax.jit(jacfwd(...))` 对象，容易使 engine rebuild 无法复用已有 executable；修复后同进程 rebuild solve 仍保持 steady-level 耗时。
- `L_protrude_max=x_max` 的 IK 主路径继续固定 static scales，`with_L1_dim()` 只动态更新第一柔性段长度，当前 P3.5 未发现 x 扫描导致 shape/static config 漂移。
- IK adjoint first-call 约 `1.6-1.8 s`，不是上一轮 `50-70 s` 级主瓶颈，但作为真实 IK macro 的可见尖峰已经通过 `--warmup_ik_adjoint` 转移到 warm-up，并复用首轮 sensitivity，避免重复计算。
- I/O 在短 macro 中占比仍可见，尤其首次 centerline 保存约 `0.4 s`。这支持后续继续保持稀疏保存，不应在 tight loop 中写压缩中心线。
- 本轮 CPU backend 的 first-call 时间约 `13-16 s`，低于上一轮 GPU 环境观测的 `53-70 s`。两者绝对值不可直接比较，但 first-call 与 steady 的数量级差异一致，流程结论可信。

## 风险与回归

- 本轮重点是编译复用和流程开销，不改变默认 dense FK 物理路径。
- IK replay 仍作为诊断，不作为前期硬门禁。
- 当前环境 CUDA 初始化失败，因此 GPU 利用率和 GPU 绝对耗时未在本轮直接验证。后续在 GPU 正常环境下应复跑同一 probe。
- `warm_up()` 预热固定 shape/static config；如果后续切换 `M_list`、打开或关闭不同物理项、改变 PyTree 结构，仍会触发新的合法编译。
- 通用 FK API 若未显式传入 `L_protrude_max`，仍会以首次 `L_protrude` 建立 static cache。IK 主路径已经传入 `x_max`，暂不阻断 P4，但后续可增加文档或 guard。
- 短 IK macro 仍未收敛到目标，且目标 y 方向没有明显改善；本轮只证明流程复用和 profiling，不证明 IK 策略质量。

## 实验结论

- P3.5 完成：真实 IK 工作流已经具备 compile/steady profiling，并能区分 first-call warm-up、steady FK solve、trial FK、FD-x FK、adjoint sensitivity、accepted-trial Jacobian stats 和 I/O。
- 已修复至少一个主要重复编译路径：FK solver 的 JIT/Jacobian/JVP/VJP wrapper 从 per-engine 对象提升到模块级对象；同进程 engine rebuild 后不再重复出现大 first-call compile。
- 已移除若干无用计算：FD-x 不再构造未使用 stats，被拒 trial 不生成 dense Jacobian，final snapshot 不生成 dense Jacobian。
- dense FK 仍是默认路径，matrix-free 仅保留为显式实验选项。
- FK golden 宽松一致性通过，当前修改没有造成可见姿态漂移或 residual 恶化。
- 可以进入 P4，但进入前建议在 CUDA 正常环境复跑 `p35_jax_compile_reuse_probe.py` 和短 IK macro，确认 GPU 下也保持同进程 compile reuse。

## 下一步计划

- P4 应聚焦 implicit adjoint sensitivity 和 IK 线性化成本，而不是继续优化 dense FK steady solve。
- P4 开始前先补一个 GPU 环境 quick gate：同进程 FK first/repeat/rebuild 和短 IK macro profiling，命令继续显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 保留 `--warmup_fk_compile` 和 `--warmup_ik_adjoint` 作为真实 IK macro 默认推荐参数。
- 后续如果要做多 target batch，应优先保持同一进程、同一 engine、同一 `M_list` 和同一 physics switch，不要用 subprocess-per-target 的方式运行。
