# eff_opt_exp_20260523155806_p35_gpu_quick_gate 实验记录

## 实验目的

在 CUDA/JAX GPU backend 正常可用时，复跑 P3.5 quick gate，确认 FK reuse probe 和短 IK macro 不再重复触发 `50-70 s` 级 first-call compile。

## 实验方案

- 确认 `nvidia-smi` 和 JAX backend 均能识别 GPU。
- 使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 运行 FK reuse probe。
- 使用 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 运行短 IK macro，并启用 `--runtime_profile`、`--warmup_fk_compile`、`--warmup_ik_adjoint`。
- 对照 P3.5 CPU quick gate 和上一轮 FK golden，检查 repeat/rebuild solve 是否保持 steady-level，以及 FK 姿态是否无明显漂移。
- 分析 P3 未将 matrix-free 作为默认路径是否符合总体计划的决策规则。

## 实验过程记录

- 2026-05-23 15:57：确认 `nvidia-smi` 可见 NVIDIA GeForce RTX 5060 Laptop GPU，JAX backend 为 `gpu`，设备为 `CudaDevice(id=0)`。
- 2026-05-23 15:58：创建本轮 quick gate 记录和日志目录。
- 2026-05-23 15:58：运行 GPU FK reuse probe，确认 first-call 与 repeat/rebuild solve 的耗时分离。
- 2026-05-23 16:00：运行 GPU 短 IK macro，启用 `--warmup_fk_compile` 和 `--warmup_ik_adjoint`。
- 2026-05-23 16:01：整理 `quick_gate_summary.json`，完成 P3 matrix-free 计划一致性分析。

## 运行时间统计

- GPU FK reuse probe：
  - first solve：`62.043 s`
  - repeat solve：`0.302 s`
  - rebuild engine 后 solve：`0.332 s` 和 `0.338 s`
  - rebuild engine 本身：约 `0.003-0.004 s`
- GPU 短 IK macro：
  - FK compile warm-up：`64.033 s`
  - IK adjoint warm-up：`1.744 s`
  - warm-up 后 init FK solve：`0.350 s`
  - 两次 trial FK solve 合计：`0.270 s`
  - final FK solve：`0.135 s`
  - FD-x plus/minus 各两次合计：`0.251 s` 和 `0.266 s`
  - 第二轮 adjoint steady：`0.029 s`

## 实验结果

- JAX backend 为 `gpu`，设备为 `cuda:0`，`XLA_PYTHON_CLIENT_PREALLOCATE=false` 生效。
- FK first-call compile 只出现一次。repeat solve 和 rebuild-engine solve 均回到 `0.3 s` 量级，没有再次出现 `50-70 s` 级 first-call compile。
- FK probe 输出保持一致：`centerline_end=[0.0996052176, 0, -0.05]`，`normE=0.001013`。
- IK macro 的 warm-up 后 FK solve、FD-x FK 和 adjoint sensitivity 均保持 steady-level，没有重复大编译。
- IK macro 仍未收敛，最终误差约 `19.896 mm`，`normE=2.051e-4`，分类 `WEAK`。本轮不把 IK 收敛作为 gate 条件。
- 本轮输出汇总见 `docs/efficiency_optimization_experiment/eff_opt_exp_20260523155806_p35_gpu_quick_gate_logs_data/quick_gate_summary.json`。

## 数据分析

- GPU 下 P3.5 的核心修复有效：模块级 JIT wrapper 让 engine rebuild 复用已编译 graph，rebuild 后 solve 未回到 first-call 编译耗时。
- GPU first-call 与上一轮观察的 `50-70 s` 区间一致，本轮 first solve `62.043 s`、IK FK warm-up `64.033 s`，说明 quick gate 覆盖到了真实 GPU compile 场景。
- warm-up 后 FK solve 和 FD-x solve 均在亚秒级，当前真实 IK 工作流不再被重复 first-call compile 主导。
- `warmup_ik_adjoint` 将首轮 `dp/dI` sensitivity 编译显式前置，并复用首轮结果；第二轮 adjoint steady 为 `0.029 s`。
- I/O 在短 macro 中仍可见，尤其 centerline 保存，但不是本轮 P4 准入阻塞。后续长实验仍应保持稀疏保存。

## P3 与 Matrix-Free 计划一致性分析

总体方案中 P3 的原始方向是 matrix-free FK，但启用条件是代表性 case 中获得明显 step 成本下降、residual 与几何保持一致、显存压力下降并通过 golden case。`eff_opt_exp_20260522110712` 的结果与这些条件相反：naive matrix-free CG 在最小 case 中 steady 约 `66.1 s`，而 dense steady 约 `0.25 s`，且低 CG 迭代下几何对比失败。

因此 P3 没有把 matrix-free 作为默认路径符合计划中的决策逻辑。当前主瓶颈已从 dense linear algebra 转为 JAX compile、流程复用和 IK 外层 sensitivity 成本；继续保留 dense FK 默认路径、matrix-free 仅作为显式实验选项，是合理的路线修正。

## 风险与回归

- 本轮只验证 GPU backend 下的 compile reuse，不验证 IK 收敛质量。
- 若 GPU quick gate 仍出现重复大编译，应优先定位 shape/static config 漂移、进程级 cache 丢失或未预热的 sensitivity transform。

## 实验结论

- GPU quick gate 通过。P3.5 的 compile reuse 修复不只在 CPU backend 有效，在 CUDA backend 下也能避免 engine rebuild 和重复 FK 调用反复触发 `50-70 s` 级 first-call compile。
- P3 没有把 matrix-free FK 作为默认路径，符合总体方案的验收和决策逻辑：matrix-free 原型未达到速度与几何一致性要求，因此继续使用 dense FK 默认路径是正确路线修正。
- 已具备 P4 准入条件。P4 应转向 implicit adjoint sensitivity 和 IK 线性化成本优化，不应继续把 matrix-free FK 作为默认主线。

## 下一步计划

- 启动 P4 前保留本 quick gate 作为 GPU 环境 sanity check。
- P4 目标应聚焦减少 IK 线性化中的完整 FK 有限差分和 sensitivity 成本。
- P4 需要单独建立 sensitivity correctness gate：well-conditioned 小 case 用 directional finite difference 对比，near-singular case 记录 conditioning 和 fallback。
