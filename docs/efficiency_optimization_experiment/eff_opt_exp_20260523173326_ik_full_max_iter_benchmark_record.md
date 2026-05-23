# eff_opt_exp_20260523173326_ik_full_max_iter_benchmark 实验记录

## 实验目的

在不推进新优化的前提下，运行一组完整上限的 IK benchmark，估算当前 P3.5 和 P4 修改后的端到端求解耗时。

## 实验方案

- 使用当前 `inverse_kinematics/ik_multisite_lm_opt.py` 主路径。
- 使用 `evolution_of_mcrs` conda 环境。
- 显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 使用 mixed magnet 配置和旧 stress 中可收敛的 moderate off-axis target。
- 使用 `outer_max_iter=240` 作为当前 stress benchmark 的代表性完整迭代上限。
- 使用 implicit `x` sensitivity、FK compile warm-up、IK adjoint warm-up 和 runtime profiling。

## 实验过程记录

- 2026-05-23 17:33：创建实验 ID、记录文件和日志目录。
- 2026-05-23 17:35：启动完整 IK benchmark。
- 2026-05-23 17:37：第一组 natural-stop run 提前停止，原因是 effective step 过小，未跑满 240 次。
- 2026-05-23 17:38：追加 forced full-cap run，关闭步长和 predicted-reduction 的提前停止门槛。
- 2026-05-23 17:50：forced full-cap run 完成 240 次外层迭代。

## 运行命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib conda run -n evolution_of_mcrs python inverse_kinematics/ik_multisite_lm_opt.py \
  --p_des=0.068,-0.031,0.042 \
  --enable_magnetics \
  --calib_file=calib/mpem_calibration_file_sp=40_order=1.yaml \
  --actuation_table_pkl=supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl \
  --m_body_list='0,0,-0.005301; 0,0,0.005301; 0.005301,0,0' \
  --flex_lengths=0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --M_list=3,2,2 \
  --x0=0.1 \
  --x_min=0.06 \
  --x_max=0.18 \
  --axis_body=1,0,0 \
  --I_max=50 \
  --max_iter=80 \
  --outer_max_iter=240 \
  --tol=1e-5 \
  --eps_p=0.001 \
  --sigma_p=0.005 \
  --seed=0 \
  --x_sensitivity_method=implicit \
  --runtime_profile \
  --warmup_fk_compile \
  --warmup_ik_adjoint
```

Forced full-cap run 在同一命令基础上额外加入：

```bash
--step_I_tol=-1 --step_x_tol=-1 --pred_red_tol=-1
```

## 运行时间统计

- Natural-stop run：
  - Wall time：`125.64 s`，约 `2.09 min`。
  - Warm-up：FK compile `69.00 s`，IK adjoint `16.15 s`。
  - 外层迭代：初始化加 8 次 trial 后提前停止。
  - 结论：该 run 不能作为 full max-iter benchmark，只能说明当前配置会快速进入步长停滞。
- Forced full-cap run：
  - Wall time：`725.51 s`，约 `12.09 min`。
  - Warm-up：FK compile `69.11 s`，IK adjoint `20.82 s`。
  - 外层迭代：完整执行 `240` 次 trial。
  - 去掉 warm-up 后的平均耗时：约 `(725.51 - 69.11 - 20.82) / 240 = 2.65 s/iter`。
  - FK solve trial 总耗时：`541.57 s`，约 `2.26 s/iter`。
  - FK observe trial 总耗时：`16.87 s`，约 `0.070 s/iter`。
  - implicit sensitivity trial 总耗时：约 `14.45 s`，约 `0.060 s/iter`。
  - trial pose 和 history I/O 总耗时：约 `31.36 s`，约 `0.131 s/iter`。
  - Max RSS：约 `1.98 GB`。

## 实验结果

- Natural-stop run：
  - `success=false`
  - `err=99.55 mm`
  - `normE=0.336`
  - `cls=REJECT`
  - 实际只进行了 8 次 trial，未达到 `outer_max_iter=240`。
- Forced full-cap run：
  - `success=false`
  - `outer_count=241`，包含初始化 `iter=0` 和 `iter=1..240`。
  - `accept_count=1`，只有初始化接受；后续 `240` 次均为 reject。
  - 初始误差 `99.55 mm`，最后 trial 误差约 `103.98 mm`。
  - 最终保存结果仍对应最后 accepted state，因此 `result.json` 中 `err=99.55 mm`。
  - 最终 `normE=0.336`，`cls=REJECT`。

## 数据分析

- 本轮 benchmark 的核心速度结论是：在 FK inner `max_iter=80` 的较高保真设置下，当前 optimized IK 主路径跑满 `240` 次外层 trial 的 wall time 约为 `12.1 min`。
- 与 P4 短 macro 中 FK inner `max_iter=4` 的估算不同，本轮高保真 FK 使单次 trial FK solve 回到约 `2.26 s/iter`，因此端到端 full-cap 耗时明显高于此前基于短 macro 的 `2-5 min` 乐观估计。
- P4 implicit sensitivity 仍然有效：每次 trial sensitivity 约 `0.060 s`，只占单次 trial 的小部分；当前 full-cap 主要耗时重新由 FK solve 主导。
- I/O 在 full-cap 中可见但不是第一瓶颈：trial pose 和 history 合计约 `31.36 s`，约占 wall time `4.3%`。
- 与旧 stress 中 `c04_upper_front_moderate` 的 `1529.78 s` 记录相比，本轮 forced full-cap `725.51 s` 约快 `2.1x`；若按每次迭代粗略比较，旧路径约 `11.1 s/iter`，当前约 `2.65-3.02 s/iter`，约 `3.7-4.2x`。
- 该比较不是严格 apples-to-apples：旧路径是 staged/auto-init 流程并成功收敛，本轮是当前 standalone optimized IK，且 forced run 不代表可用 IK 解。

## 实验结论

- 当前端到端 IK 性能相比旧 stress 记录已有明显改善，但若采用 FK inner `max_iter=80`，完整 `240` 次 cap 仍约 `12 min`，尚未达到 `1-3 min/组` 的目标。
- P3.5 和 P4 的收益没有消失，但 full-cap 下主瓶颈重新变成高保真 FK solve，而不是 sensitivity 或重复 JAX 编译。
- 当前 standalone optimized IK 在 mixed-magnet target 上存在初始化和 globalization 问题：随机初值导致初始 FK 为 `REJECT`，后续 trial 全部 rejected。该结果不能用于评价 IK 成功质量，只能用于评估 full-cap runtime。

## 下一步计划

- 若继续推进，应优先做 P5 reduced-space IK 和初始化/globalization 修复，而不是继续优化 P4 sensitivity。
- 后续 benchmark 应分两类运行：一类固定 forced full-cap 测最坏耗时，另一类使用可用初始化和正常停止条件测成功收敛耗时。
- 若目标仍是 `1-3 min/组`，需要继续降低高保真 FK solve 单次成本，或减少外层 trial 数量到约 `50-80` 以下。
