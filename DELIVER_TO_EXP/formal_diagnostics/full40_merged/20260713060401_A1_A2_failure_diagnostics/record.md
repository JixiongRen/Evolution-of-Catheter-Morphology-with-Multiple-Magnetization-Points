# A1/A2 有监督逆运动学失败诊断报告

## 一、实验目的

本报告执行失败诊断计划的前置门禁与历史轨迹审计，目标是解释 A1/A2 单点磁化导管在无重力、5 mm 阈值圆周任务中的失败来源，并把证据转换为后续代码修改任务。

## 二、数据来源

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40`
- 诊断输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/full40_merged/20260713060401_A1_A2_failure_diagnostics`
- git 版本：`e4babc9065e77ec4f446a6884604c891c288877d`
- 当前 JAX 设备：`{'timestamp': '2026-07-13 06:04:01', 'source_run': '/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40', 'git_commit': 'e4babc9065e77ec4f446a6884604c891c288877d', 'python': '/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'jax': {'available': True, 'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}}`

## 三、门禁结论

- 共检查 40 个点，其中 D0 门禁完全有效的点为 40 个。
- 日志中存在 CUDA 初始化失败或 CPU 回退的点数：0。
- 求解器成功标志与真实末端成功标志不一致的点数：0。
- 中心线终点、柔性段末端和报告真实末端存在几何约定冲突的点数：0。

结论：本轮数据的 D0 前置门禁已通过：运行日志未显示 CPU 回退，磁化配置与单磁体结构一致，中心线终点与报告真实末端一致，求解器成功标志也已统一到真实末端。

- 真实末端进入 5 mm 阈值的点数：11/40。
- FK 平衡为 GOOD 但真实末端未达标的点数：27。
- FK 平衡非 GOOD 且真实末端未达标的点数：2。

## 四、代表点诊断

| 点位 | 真实末端误差(mm) | 柔性末端误差(mm) | 末端几何分类 | 主因 | 证据等级 | 可信度 | 结论类型 | 证据 |
|---|---:|---:|---|---|---|---|---|---|
| A1-pt01 | 4.939 | 7.677 | centerline_matches_reported_true_tip | 有效成功 | 强 | 高 | 已排除因素 | 真实末端误差已进入 5 mm 阈值。 |
| A1-pt08 | 24.789 | 27.423 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 强 | 高 | 已确认根因 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A1-pt12 | 8.805 | 9.038 | centerline_matches_reported_true_tip | 局部预测方向错误 | 强 | 高 | 已确认根因 | 多次出现预测下降但实际目标函数上升，需审计灵敏度和步长策略。 |
| A1-pt13 | 28.220 | 27.792 | centerline_matches_reported_true_tip | 局部预测方向错误 | 强 | 高 | 已确认根因 | 多次出现预测下降但实际目标函数上升，需审计灵敏度和步长策略。 |
| A1-pt17 | 53.451 | 55.852 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 强 | 高 | 已确认根因 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt05 | 4.721 | 7.551 | centerline_matches_reported_true_tip | 有效成功 | 强 | 高 | 已排除因素 | 真实末端误差已进入 5 mm 阈值。 |
| A2-pt09 | 8.982 | 10.776 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 中 | 中 | 仅为伴随现象或放大因素 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt10 | 17.939 | 20.159 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 强 | 高 | 已确认根因 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt17 | 46.027 | 48.105 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 强 | 高 | 已确认根因 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt19 | 40.998 | 43.561 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 中 | 中 | 仅为伴随现象或放大因素 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |

## 五、全局主因分布

| 主因 | 点数 |
|---|---:|
| 阻尼过大导致停滞 | 27 |
| 有效成功 | 11 |
| 局部预测方向错误 | 2 |

## 六、对后续算法修改的直接约束

1. 对误差较大的 GOOD 点，首要问题不是内层 FK 崩溃，而是外层控制更新未能把真实末端推向目标；下一步应优先审计灵敏度方向、局部预测下降量和步长接受机制。
2. 对误差较大且 FK 只达到 WEAK 的点，必须先保证 FK 平衡质量，再判断 IK 是否失败；弱平衡状态下的灵敏度不应直接作为算法结论。
3. 成功点可作为正常样本，用于中心差分灵敏度审计；失败点需要补做多初值、反向遍历和局部可达性采样，以区分优化器失败与疑似不可达。
4. 若中心差分显示灵敏度误差超过 10%，优先修复隐式灵敏度；若小步预测准确但大步失败，则优先改信赖域和回溯策略。

## 七、已生成文件

- `diagnostic_summary.csv`：每个点的主因、证据和下一步任务。
- `endpoint_convention_audit.csv`：末端几何约定和目标函数错配审计。
- `trust_region_audit.csv`：历史预测下降、真实下降、阻尼和步长记录。
- `implicit_sensitivity_log_summary.csv`：历史记录中的隐式灵敏度状态摘要。
- `sensitivity_columns.csv`：D1 中心差分灵敏度审计占位文件；本轮已完成 D0 门禁和历史轨迹审计，D1/D3/D4 尚未执行，但已不再受 D0 阻断。
- `branch_trials.csv`：D3 多初值与分支诊断占位文件；本轮已完成 D0 门禁和历史轨迹审计，D1/D3/D4 尚未执行，但已不再受 D0 阻断。
- `reachability_samples.npz`：D4 经验可达图占位文件；本轮已完成 D0 门禁和历史轨迹审计，D1/D3/D4 尚未执行，但已不再受 D0 阻断。
- `runtime_environment.json`：当前运行环境和设备信息。
- `*_error_by_angle.png`：误差随圆周角度变化。
- `endpoint_convention_representative.png`：代表点末端约定冲突图。
- `root_cause_distribution.png`：主因分布图。

## 八、下一步建议

建议下一步只针对本轮代表点进入 D1/D2/D3：先做真实末端中心差分灵敏度审计，再做局部预测审计和多初值分支诊断；暂不扩大到全 40 点，避免在根因未定位前消耗大量算力。

## 九、闭环补充证据

已追加闭环补充诊断，输出位于：

- `DELIVER_TO_EXP/formal_diagnostics/closure_sensitivity/20260713061913_A1_A2_diagnostic_closure`
- `DELIVER_TO_EXP/formal_diagnostics/closure_sensitivity/20260713062448_A1_A2_diagnostic_closure`

补充结论如下：

1. 基于 1181 条历史外层轨迹状态构建经验可达集后，A1 中 7 个目标自身达标、3 个失败点历史近可达、6 个失败点历史边界可达、4 个失败点历史探索未覆盖；A2 中 4 个目标自身达标、1 个失败点历史近可达、8 个失败点历史边界可达、7 个失败点历史探索未覆盖。
2. A1-pt12、A1-pt13 和 A2-pt17 均可通过历史轨迹重放回到原 IK 分支；其中 A2-pt17 重放末端差异约 0.003 mm，说明至少部分失败点可以做分支保持审计。
3. 分支保持中心差分显示，A1-pt12 和 A1-pt13 的推进量方向灵敏度较可靠，但多个电流通道误差超过 10%；A2-pt17 的电流通道和推进量通道均明显不可靠。
4. 因此，A1 的下一轮重点应是外层信赖域、回溯和电流弱通道正则化；A2 的下一轮重点应先修复隐式灵敏度和病态正则化，再优化外层路径策略。

已进一步完成全 40 点接受状态分支重放筛查，输出位于：

- `DELIVER_TO_EXP/formal_diagnostics/closure_branch_full/20260713070419_A1_A2_diagnostic_closure`

筛查结论：

1. 全 40 点中，19 点可以仅凭接受状态重放回原分支，21 点不能回到原分支。
2. A1 中 12 点可重放、8 点不可重放；A2 中 7 点可重放、13 点不可重放。
3. 该结果说明旧实验记录存在明显分支路径依赖，且没有保存完整平衡状态 `z_star`；因此不能对所有历史点都做低成本严格分支保持灵敏度审计。
4. 下一轮实验必须在每个接受状态保存平衡状态、尺度和参数哈希，否则灵敏度诊断会被“分支无法复现”污染。
5. `sensitivity_columns.csv` 已更新为全 40 点 D1 状态表：18 点可审计但未直接差分，19 点当前历史记录下无物理意义，2 点部分可信，1 点不可信。
