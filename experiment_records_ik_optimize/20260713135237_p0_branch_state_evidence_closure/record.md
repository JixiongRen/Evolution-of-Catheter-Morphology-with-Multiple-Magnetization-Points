# P0 状态证据闭环记录

## 实验目的

本轮目标是为 A1/A2 有监督逆运动学后续灵敏度审计建立可追溯的平衡状态证据链。重点不是提高收敛成功率，而是保证每一次外层候选评估都能保存完整状态，并能从保存状态中重新提取真实末端和中心线。

## 实验条件

- 导管类型：A1、A2 单点磁化导管。
- 目标点：A1 的 `pt_01`、`pt_12`、`pt_13`；A2 的 `pt_05`、`pt_17`、`pt_19`。
- 固定条件：无重力，`M_list=5`，FK 容差 `1e-5`，电流范围 `[-20,20] A`，推进范围 `[0.003001,0.203] m`。
- 成功阈值：真实刚性远端末端位置误差不超过 `5 mm`。
- 环境：`evolution_of_mcrs`，显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 与 `JAX_PLATFORMS=cuda`。

## 代码修改

- 新增 `inverse_kinematics/ik_branch_state.py`，用于保存每次 FK 候选评估的状态快照、哈希、控制量、FK 统计、真实末端、柔性末端和中心线。
- 修改 `inverse_kinematics/ik_multisite_lm_opt.py`，增加 `--branch_state_capture` 开关，并在初始状态、每次试探状态和最终状态保存分支状态。
- 修改 `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`，正式实验默认打开 `--branch_state_capture`。
- 新增 `inverse_kinematics/verify_branch_states.py`，从 `.npz` 快照重新读取 `z_star_bar` 和无量纲尺度，重新提取真实末端与中心线，检查重载一致性。

## 运行命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
--disable-gravity --eps-p-mm 5 --force-formal-20 \
--catheters A1 --target-indices 1,12,13 \
--outer-max-iter 50 --per-point-timeout-s 2700
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
--disable-gravity --eps-p-mm 5 --force-formal-20 \
--catheters A2 --target-indices 5,17,19 \
--outer-max-iter 50 --per-point-timeout-s 2700
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
-m inverse_kinematics.verify_branch_states \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity \
DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity \
--out-csv experiment_records_ik_optimize/20260713135237_p0_branch_state_evidence_closure/reload_gate.csv \
--out-json experiment_records_ik_optimize/20260713135237_p0_branch_state_evidence_closure/reload_gate.json
```

## 实验结果

| 导管 | 点号 | 真实末端误差(mm) | 是否进入 5 mm | FK 状态 | 墙钟时间(s) | 终止原因 |
|---|---:|---:|---|---|---:|---|
| A1 | 1 | 4.939 | 是 | GOOD | 205.6 | reached_threshold_true_tip |
| A1 | 12 | 49.235 | 否 | GOOD | 397.5 | max_outer_iter |
| A1 | 13 | 69.083 | 否 | GOOD | 666.1 | early_stop_or_plateau |
| A2 | 5 | 4.407 | 是 | GOOD | 162.7 | reached_threshold_true_tip |
| A2 | 17 | 47.454 | 否 | GOOD | 740.4 | early_stop_or_plateau |
| A2 | 19 | 70.651 | 否 | GOOD | 527.8 | early_stop_or_plateau |

状态保存完整性检查结果：

| 点位 | 状态快照数 | 历史记录带状态索引数 | 清单首项 | 清单末项 | 完整性 |
|---|---:|---:|---|---|---|
| A1-pt_01 | 16 | 15 | init | final | 通过 |
| A1-pt_12 | 52 | 51 | init | final | 通过 |
| A1-pt_13 | 49 | 48 | init | final | 通过 |
| A2-pt_05 | 14 | 13 | init | final | 通过 |
| A2-pt_17 | 40 | 39 | init | final | 通过 |
| A2-pt_19 | 24 | 23 | init | final | 通过 |

重载门禁检查结果：

- 总状态数：`195`。
- 真实末端重载误差最大值：`0.0 mm`。
- 中心线重载最大误差：`0.0 mm`。
- 所有状态均通过 `0.1 mm` 真实末端门禁和 `0.5 mm` 中心线门禁。

## 数据索引

- A1 代表点输出：`DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/`
- A2 代表点输出：`DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/`
- 重载门禁 CSV：`experiment_records_ik_optimize/20260713135237_p0_branch_state_evidence_closure/reload_gate.csv`
- 重载门禁 JSON：`experiment_records_ik_optimize/20260713135237_p0_branch_state_evidence_closure/reload_gate.json`

## 分析结论

本轮已经完成 P0 的状态保存和重载一致性闭环。六个代表点的每一次外层候选评估均有独立 `.npz` 快照和清单记录，历史轨迹也能回指对应的 `eval_id`。这解决了旧数据只能复现最终末端、不能追踪中间平衡分支的问题。

本轮尚未证明同一平衡分支的差分灵敏度正确性。当前重载验证只证明“保存的状态可重新解析且末端/中心线约定一致”，不等价于“对控制量扰动后仍保持同一分支”。后续 P1 仍需要执行双侧扰动、回环门禁和三种灵敏度公式对比。

## 下一步计划

1. 基于本轮 195 个状态快照，优先对 A1-pt_01、A1-pt_12、A1-pt_13、A2-pt_05、A2-pt_17、A2-pt_19 执行 P1 灵敏度再审计。
2. 对每个有效状态执行电流与推进量差分尺度扫描，严格区分强响应通道与弱响应通道。
3. 若 A2 代表点继续出现大幅灵敏度不一致，先定位电流到磁场、平衡方程偏导、真实末端链式导数和分支跳变四类原因，不直接进入全量 A2 回归。
