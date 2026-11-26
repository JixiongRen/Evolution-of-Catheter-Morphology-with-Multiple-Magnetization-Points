# A1/A2 有监督逆运动学失败诊断报告

## 一、实验目的

本报告执行失败诊断计划的前置门禁与历史轨迹审计，目标是解释 A1/A2 单点磁化导管在无重力、5 mm 阈值圆周任务中的失败来源，并把证据转换为后续代码修改任务。

## 二、数据来源

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity`
- 诊断输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/full40/20260713045430_A1_A2_failure_diagnostics`
- git 版本：`e4babc9065e77ec4f446a6884604c891c288877d`
- 当前 JAX 设备：`{'timestamp': '2026-07-13 04:54:30', 'source_run': '/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity', 'git_commit': 'e4babc9065e77ec4f446a6884604c891c288877d', 'python': '/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'jax': {'available': True, 'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}}`

## 三、门禁结论

- 共检查 40 个点，其中 D0 门禁完全有效的点为 37 个。
- 日志中存在 CUDA 初始化失败或 CPU 回退的点数：0。
- 求解器成功标志与真实末端成功标志不一致的点数：0。
- 中心线终点、柔性段末端和报告真实末端存在几何约定冲突的点数：3。

结论：本轮数据尚不能直接进入完整灵敏度审计。需要先处理环境、磁化配置、末端定义或成功判据中的前置问题。

- 真实末端进入 5 mm 阈值的点数：10/40。
- FK 平衡为 GOOD 但真实末端未达标的点数：25。
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
| 阻尼过大导致停滞 | 25 |
| 有效成功 | 10 |
| 运行未完整结束 | 3 |
| 局部预测方向错误 | 2 |

## 六、对后续算法修改的直接约束

1. 先统一物理末端定义。需要确认 `x_rigid_list_bar` 表示刚段中点还是刚段远端；若中心线提取逻辑正确，则当前 `_true_rigid_distal_tip()` 很可能把刚段远端又向前平移了半个刚段长度。
2. 外层位置残差、自然终止条件、`result.json` 中的 `success` 字段、正式实验统计必须使用同一个末端点。否则会出现柔性段末端达标但真实末端未达标的假成功。
3. GPU 回退点必须作废并重跑。现有部分 A1/A2 点的日志显示 CUDA 初始化失败，这些点不能用于耗时或分支稳定性结论。
4. D1 中心差分灵敏度审计应在上述三项修复后再执行；否则差分对象和隐式灵敏度对象并不一致，误差结论没有物理意义。
5. 对已存在的历史轨迹，阻尼过大和步长接近零的停滞现象确实存在，但它应作为第二优先级处理，排在末端定义和环境门禁之后。

## 七、已生成文件

- `diagnostic_summary.csv`：每个点的主因、证据和下一步任务。
- `endpoint_convention_audit.csv`：末端几何约定和目标函数错配审计。
- `trust_region_audit.csv`：历史预测下降、真实下降、阻尼和步长记录。
- `implicit_sensitivity_log_summary.csv`：历史记录中的隐式灵敏度状态摘要。
- `sensitivity_columns.csv`：D1 中心差分灵敏度审计占位文件；本轮因 D0 门禁未完全通过，D1/D3/D4 暂不执行。
- `branch_trials.csv`：D3 多初值与分支诊断占位文件；本轮因 D0 门禁未完全通过，D1/D3/D4 暂不执行。
- `reachability_samples.npz`：D4 经验可达图占位文件；本轮因 D0 门禁未完全通过，D1/D3/D4 暂不执行。
- `runtime_environment.json`：当前运行环境和设备信息。
- `*_error_by_angle.png`：误差随圆周角度变化。
- `endpoint_convention_representative.png`：代表点末端约定冲突图。
- `root_cause_distribution.png`：主因分布图。

## 八、下一步建议

建议先执行一个小修复批次：统一末端定义、修正成功标志、在日志中强制记录 JAX 设备和工作空间越界信息。随后只重跑十个代表点，确认 D0 门禁通过，再开展 D1 灵敏度中心差分审计和 D3 多初值分支诊断。