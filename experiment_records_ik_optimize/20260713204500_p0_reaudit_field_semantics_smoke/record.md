# P0 复审字段语义烟测记录

日期：2026-07-13

## 一、目的

本轮用于验证灵敏度复审脚本的审计字段语义修正。目标是让有限差分审计结果明确区分计算成功、同分支、弱响应和可进入统计四类概念，避免继续把弱响应样本或跨分支样本误纳入灵敏度相对误差统计。

## 二、代码修改

本轮修改范围：

- `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py`
- `inverse_kinematics/ik_branch_state.py`
- `inverse_kinematics/ik_multisite_lm_opt.py`
- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

主要改动：

- 新增 `step_family`、`displacement_gate_mm`、`step_weak_gate_triggered`、`weak_gate_triggered`、`channel_max_displacement_mm`、`branch_consistent`、`replay_converged`、`used_in_stats`、`stats_exclusion_reason` 字段。
- 保留 `weak_channel` 作为兼容字段，但其语义统一为通道级弱响应门限，而不是单步扰动门限。
- `summary.json` 新增 `fd_gate_summary`，记录有限差分门禁计数。
- 运行环境记录新增 `git_tracked_dirty` 和 `git_untracked_count`，避免同一提交号下混入未提交代码变更。

## 三、验证命令

编译检查：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  inverse_kinematics/ik_multisite_lm_opt.py \
  inverse_kinematics/ik_branch_state.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

最小有限差分复审：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_17 \
--out-dir experiment_records_ik_optimize/20260713204500_p0_reaudit_field_semantics_smoke/A2_pt17_ch4_last_accept_seg1 \
--selector last_accept \
--current-channels 4 \
--current-steps-a 0.0625 \
--x-steps-mm "" \
--sensitivity-modes residual_svd \
--solve-mode svd \
--ridge 0 \
--svd-rcond 1e-9 \
--spectral-filter hard \
--spectral-damping 0 \
--roundtrip-segments 1 \
--terminal-gate-mm 0.25 \
--centerline-gate-mm 0.5 \
--weak-displacement-mm 0.25
```

## 四、结果

编译检查通过。

最小有限差分复审成功完成，GPU 设备为 NVIDIA GeForce RTX 5060 Laptop GPU。输出中已经包含新增字段，且 `summary.json` 中的门禁统计为：

- `fd_ok=1`
- `replay_converged=1`
- `branch_consistent=1`
- `weak_gate_triggered=0`
- `used_in_stats=1`

该样本说明新字段可以正确表达：当前扰动重放成功、回环同分支、不是弱响应，并且可以进入灵敏度统计。

运行环境函数额外验证了 Git 状态字段。在本轮代码尚未提交时，`git_tracked_dirty=true`，说明后续实验可以识别“提交号相同但工作区有未提交修改”的情况。

## 五、结论

P0 的审计字段语义修正已经通过最小烟测。下一步可以在同一批失败强通道上重跑 P1 复审，并使用 `used_in_stats` 和 `stats_exclusion_reason` 生成更可靠的通道级归因表。
