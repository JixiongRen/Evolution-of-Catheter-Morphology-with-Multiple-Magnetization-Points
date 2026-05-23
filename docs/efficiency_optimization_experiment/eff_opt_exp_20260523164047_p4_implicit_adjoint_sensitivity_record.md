# eff_opt_exp_20260523164047_p4_implicit_adjoint_sensitivity 实验记录

## 实验目的

实现并验证 P4 implicit adjoint sensitivity，使 IK 线性化不再依赖多个完整 FK 有限差分，并降低真实 IK workflow 的 sensitivity 成本。

## 实验方案

- 审查 `ik_diff.py`、`ik_multisite_lm_opt.py`、`ForwardKinematicsEngine` 和 equilibrium solver 的 sensitivity 调用链。
- 定义基于 `E(z, I, x)=0` 的 implicit sensitivity 接口，统一计算 `dp/dI` 和 `dp/dx`。
- 优先实现不要求 `lm_stats` 默认保存完整 dense `J` 的路径；保留旧 FD/FK sensitivity 作为 fallback。
- 增加 micro sensitivity 校验，使用 directional finite difference 验证 well-conditioned case。
- 运行短 IK macro near case，对比旧 FD-x 路径和新 implicit sensitivity 路径的成本。
- 保护 FK golden 整体姿态、曲率趋势和 residual 质量。

## 实验过程记录

- 2026-05-23 16:40：创建 P4 实验 ID、记录文件和日志目录。
- 2026-05-23 16:43：完成 sensitivity 链路审查，确认旧路径中 `dp/dI` 依赖 `lm_stats.J`，`dp/dx` 依赖两个完整 FK 有限差分。
- 2026-05-23 16:48：实现 `residual_bar_zIx` 和统一 implicit sensitivity 接口，默认同时输出 `dp/dI` 与 `dp/dx`。
- 2026-05-23 16:55：运行 GPU micro sensitivity directional FD 校验，记录 conditioning 与方向误差。
- 2026-05-23 16:56：运行短 IK macro implicit-x 路径。
- 2026-05-23 16:58：运行短 IK macro FD-x legacy 对照。
- 2026-05-23 17:02：复用 FK solver 已预热的 `jacobian_jit`，避免 implicit sensitivity 额外编译 `dE/dz`。
- 2026-05-23 17:04：运行更新后的 FD-x 对照。
- 2026-05-23 17:05：运行更新后的 micro sensitivity probe。
- 2026-05-23 17:10：运行 regularized sanity case，确认强 ridge 可降低 conditioning，但会改变物理导数。

## 运行时间统计

- 更新后的 GPU short IK macro：
  - implicit-x：`warmup_fk_compile=59.150 s`，`warmup_ik_adjoint=12.595 s`。
  - implicit-x steady sensitivity：局部 `Jz` 约 `0.0078 s`，implicit adjoint 约 `0.0448 s`，合计约 `0.0609 s`。
  - FD-x legacy 对照：FD-x 两侧 FK 合计 `0.5783 s`，accepted-trial Jacobian stats 合计 `0.0346 s`，sensitivity+FD 合计约 `0.6863 s`。
  - steady sensitivity speedup：约 `11.27x`。
- GPU micro sensitivity probe：
  - `straight_nomag_nogravity`：局部 `Jz=0.0084 s`，sensitivity first call `12.90 s`。
  - `mag_zero_mid`：局部 `Jz=0.0110 s`，sensitivity first call `16.11 s`。
  - `mixed_balanced`：局部 `Jz=0.0102 s`，sensitivity first call `15.75 s`。
- first-call sensitivity compile 仍存在，但只发生在新 `dE/d(I,x)` transform 首次调用；steady IK macro 中该路径回到约 `0.05-0.06 s`。

## 实验结果

- 新增 `residual_bar_zIx(z, I, L1_dim, params_static)`，统一 residual 对 current 和 protrusion 相关第一柔性段长度的显式接口。
- 新增 `compute_tip_sensitivities_via_lm_adjoint()`，使用 LM/Gauss-Newton stationarity 近似：
  - `H = Jz.T @ Jz + lambda_eff I`
  - `H U = (dp/dz).T`
  - `dp/dq = -U.T @ (Jz.T @ Eq)`，其中 `q=[I,x]`
- IK 默认 `--x_sensitivity_method implicit`，不再为 `dp/dx` 执行两个完整 FK solve。
- IK 初始和 accepted trial 默认不再保存 dense `lm_stats.J`；仅在 `--store_accepted_jacobian` 或 legacy 对照中保存。
- implicit-x 与 FD-x macro 的最终结果基本一致：
  - implicit-x：`err=19.896461 mm`，`normE=2.04975e-4`
  - FD-x legacy：`err=19.896493 mm`，`normE=2.05103e-4`
  - 误差差异约 `-3.2e-5 mm`，residual 差异约 `-1.27e-7`
- directional FD micro 结果：
  - `straight_nomag_nogravity` 两个方向均通过，最大相对误差 `0.0114`，最大绝对误差 `1.49e-6 m`。
  - `mag_zero_mid` 两个方向按绝对误差门限通过，最大绝对误差 `1.27e-4 m`，但相对误差较高，说明 FD 信号较小且磁场路径更敏感。
  - `mixed_balanced` 一个方向通过、一个方向 warning，最大绝对误差 `4.92e-4 m`。
- conditioning：
  - `straight_nomag_nogravity` 与 `mag_zero_mid` 的 dense `H` condition number 约 `6.56e11`，标记 `ill_conditioned`。
  - `mixed_balanced` condition number 约 `5.95e9`，标记 `weak_conditioning`。
  - ridge `1000` 的 regularized sanity case 可把 condition number 降到约 `5.31e5`，但导数被正则化改变，只作为稳定性诊断，不作为物理导数门禁。

## 数据分析

- P4 的主要 runtime 目标达成：IK outer linearization 的 `dp/dx` 不再需要两个完整 FK solve，代表性短 macro 中 sensitivity 相关 steady 成本从约 `0.686 s` 降到约 `0.061 s`。
- P4 的结构目标部分达成：默认不再要求 `lm_stats` 保存 dense `J`；当前 transitional path 仍按需局部物化 `Jz`，但不写回 `lm_stats`，且复用 FK solver 已预热的 `jacobian_jit`。
- 当前没有把 matrix-free adjoint 作为默认路径，符合 P3 的结论：matrix-free FK 原型此前未达启用条件。P4 先消除 FD-x FK amplification，比强行 matrix-free 更稳。
- conditioning 是主要风险。当前 residual/constraint scaling 使 dense `H` condition number 很高，因此 sensitivity 只能作为优化线性化方向使用，不能当作严格物理导数报告。
- directional FD 对照显示，简单无磁 case 方向导数质量较好；磁场和 mixed case 中相对误差会被小 FD 信号、弱平衡和 conditioning 放大。前期采用宽松门禁是合理的。

## 风险与回归

- 本轮只处理 sensitivity 和 IK linearization，不进入 P5 reduced-space IK。
- dense FK 仍为默认路径，matrix-free 仅保留为显式实验选项。
- IK replay 不作为本轮硬门禁；主要保护 FK golden 姿态一致性。
- implicit sensitivity 当前是 `dense_implicit_transitional`，仍物化 `Jz`，但不存入 `lm_stats`。后续若要完全 matrix-free，需要单独解决 P3 中 JVP/VJP 成本问题。
- 所有代表性 micro case 都有 conditioning warning；P5 必须继续记录 `sensitivity.status` 并保留 fallback。

## 实验结论

- P4 工程目标通过，带 conditioning warning。
- 已实现统一 implicit adjoint sensitivity，当前 IK 默认使用 implicit `dp/dx`，不再为小扰动执行两个完整 FK solve。
- 代表性 IK macro 中 sensitivity steady 成本降低约 `11.27x`，最终 tip error 和 equilibrium residual 与 FD-x 对照等价。
- `lm_stats.J` 不再是默认必需状态；dense `Jz` 仅在 sensitivity 内部局部物化，且不写入 stats。
- 因 conditioning warning，后续进入 P5 可以进行，但必须保留 sensitivity 可信度标签、ridge/fallback 策略和 FD reference spot-check。

## 下一步计划

- P5 可以开始 reduced-space IK 原型，但不应移除旧 IK fallback。
- P5 需要继续记录 current utilization、x/protrusion saturation、sensitivity status 和 FK replay。
- 若 P5 中出现不稳定，应优先处理 conditioning 和 globalization，而不是回退到完整 FK FD-x。
