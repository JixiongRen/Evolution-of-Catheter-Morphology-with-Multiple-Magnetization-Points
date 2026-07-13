# A1/A2 有监督逆运动学 P0 代表点状态重跑记录

## 一、实验目的

本轮用于补齐旧正式 40 点实验缺失的分支状态证据。旧合并基线没有保存 `branch_state_manifest.jsonl`，因此无法证明“同一平衡分支”。本轮重跑 A1 与 A2 共六个代表点，并开启分支状态捕获，为后续灵敏度审计和同分支门禁提供可追溯状态快照。

## 二、实验设置

- 导管：A1、A2。
- 点位：A1-pt01、A1-pt12、A1-pt13、A2-pt05、A2-pt17、A2-pt19。
- 重力：关闭。
- 离散化：`M_list=5`。
- 正运动学容差：`1e-5`。
- 成功阈值：最后刚性段远端真实末端误差小于 `5 mm`。
- 电流范围：`[-20, 20] A`。
- 推进范围：`[0.003001, 0.203] m`。
- 环境变量：`XLA_PYTHON_CLIENT_PREALLOCATE=false`，`JAX_PLATFORMS=cuda`。
- A1 正式输出目录：`DELIVER_TO_EXP/formal_experiments/20260714035640_A1_A2_5mm_supervised_IK_no_gravity/`。
- A2 正式输出目录：`DELIVER_TO_EXP/formal_experiments/20260714041425_A1_A2_5mm_supervised_IK_no_gravity/`。

## 三、运行说明

首次尝试使用后台启动，进程静默退出；随后用 `timeout 15s` 前台调试确认命令能进入 Python，该调试产生的 `DELIVER_TO_EXP/formal_experiments/20260714035529_A1_A2_5mm_supervised_IK_no_gravity/` 为中断无效目录，不纳入结果。

有效运行采用前台 PTY 会话完成。命令保存在 `run_a1_representative.sh`。

## 四、A1 结果摘要

| 点位 | 真实末端误差(mm) | 是否进入 5 mm | 外层迭代 | 接受/拒绝 | 停止原因 | 热启动来源 |
|---|---:|---|---:|---|---|---|
| A1-pt01 | 3.638917 | 是 | 11 | 12/0 | reached_threshold_true_tip | cold_start_formal |
| A1-pt12 | 50.588841 | 否 | 50 | 38/13 | max_outer_iter | last_success_pt_01 |
| A1-pt13 | 69.829588 | 否 | 50 | 38/13 | max_outer_iter | last_success_before_failed_point |

本轮结果与此前结论一致：pt01 是稳定健康点；pt12 仍远离目标；pt13 在本协议下没有复现历史近解，说明它仍高度依赖具体邻点热启动和分支状态。

## 五、A2 结果摘要

| 点位 | 真实末端误差(mm) | 是否进入 5 mm | 外层迭代 | 接受/拒绝 | 停止原因 | 热启动来源 | FK 分类 |
|---|---:|---|---:|---|---|---|---|
| A2-pt05 | 4.896264 | 是 | 10 | 11/0 | reached_threshold_true_tip | cold_start_formal | GOOD |
| A2-pt17 | 28.487635 | 否 | 29 | 16/14 | reject_stall | last_success_pt_05 | WEAK |
| A2-pt19 | 81.576099 | 否 | 17 | 11/7 | early_stop_or_plateau | last_success_before_failed_point | GOOD |

A2-pt05 在本轮冷启动下仍可进入 5 mm。A2-pt17 未成功且 FK 分类为 `WEAK`，说明该点后续灵敏度审计必须首先确认平衡质量；A2-pt19 误差很大，仍属于困难点。

## 六、分支状态保存与重载验证

状态快照数量：

- A1-pt01：13 个状态。
- A1-pt12：52 个状态。
- A1-pt13：52 个状态。
- A2-pt05：12 个状态。
- A2-pt17：31 个状态。
- A2-pt19：19 个状态。

使用 `inverse_kinematics/verify_branch_states.py` 对保存状态进行重载一致性验证，结果为通过：

- A1 总状态数：117。
- A2 总状态数：62。
- 真实末端重载误差门限：`0.1 mm`。
- 轴线最大重载误差门限：`0.5 mm`。
- 所有状态均满足上述门限。

注意：该验证证明保存的 `z_star_bar` 与记录的末端和轴线数据自洽；它仍不是完整“同分支”证明。完整同分支还需要后续完成逐步轨迹重放和控制量往返回环门禁。

## 七、当前结论

1. A1/A2 六个代表点已具备后续 P1 灵敏度审计所需的状态快照基础。
2. 旧合并基线缺少分支状态，不能直接用于同分支灵敏度结论。
3. pt13 在本轮小回归中退化到约 `69.83 mm`，说明历史近解不是当前协议的稳定产物；后续必须把入盆策略、分支状态继承和局部精修分开诊断。
4. A2-pt17 的 FK 分类为 `WEAK`，后续不能直接把它的灵敏度误差解释为外层优化错误；必须先把平衡质量纳入 P1 门禁。
5. 本轮重载验证仍只证明保存状态自洽，不等于完整同分支证明；完整同分支还需要逐步轨迹重放和控制量往返回环。

## 八、输出文件

- `run_a1_representative.sh`：A1 三点重跑命令。
- `run_a2_representative.sh`：A2 三点重跑命令。
- `a1_run_summary.json`：正式脚本生成的机器摘要。
- `a1_run_summary.csv`：正式脚本生成的表格摘要。
- `a1_run_report.md`：正式脚本生成的中文报告副本。
- `a1_branch_reload_verification.csv`：分支状态重载验证表。
- `a1_branch_reload_verification.json`：分支状态重载验证机器摘要。
- `a2_run_summary.json`：正式脚本生成的机器摘要。
- `a2_run_summary.csv`：正式脚本生成的表格摘要。
- `a2_run_report.md`：正式脚本生成的中文报告副本。
- `a2_branch_reload_verification.csv`：分支状态重载验证表。
- `a2_branch_reload_verification.json`：分支状态重载验证机器摘要。
- `logs/`：后台启动尝试的标准输出和错误输出。本次有效运行来自前台 PTY，会话输出已在本记录中摘录。
