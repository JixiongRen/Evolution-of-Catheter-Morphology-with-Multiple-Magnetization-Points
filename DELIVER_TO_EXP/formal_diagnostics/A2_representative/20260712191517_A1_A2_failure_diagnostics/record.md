# A1/A2 有监督逆运动学失败诊断报告

## 一、实验目的

本报告执行失败诊断计划的前置门禁与历史轨迹审计，目标是解释 A1/A2 单点磁化导管在无重力、5 mm 阈值圆周任务中的失败来源，并把证据转换为后续代码修改任务。

## 二、数据来源

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712183547_A1_A2_5mm_supervised_IK_no_gravity`
- 诊断输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/A2_representative/20260712191517_A1_A2_failure_diagnostics`
- git 版本：`385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe`
- 当前 JAX 设备：`{'timestamp': '2026-07-12 19:15:17', 'source_run': '/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712183547_A1_A2_5mm_supervised_IK_no_gravity', 'git_commit': '385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe', 'python': '/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'jax': {'available': True, 'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}}`

## 三、门禁结论

- 共检查 5 个点，其中 D0 门禁完全有效的点为 5 个。
- 日志中存在 CUDA 初始化失败或 CPU 回退的点数：0。
- 求解器成功标志与真实末端成功标志不一致的点数：0。
- 中心线终点、柔性段末端和报告真实末端存在几何约定冲突的点数：0。

结论：本轮数据的 D0 前置门禁已通过：运行日志未显示 CPU 回退，磁化配置与单磁体结构一致，中心线终点与报告真实末端一致，求解器成功标志也已统一到真实末端。

- 真实末端进入 5 mm 阈值的点数：1/5。
- FK 平衡为 GOOD 但真实末端未达标的点数：4。
- FK 平衡非 GOOD 且真实末端未达标的点数：0。

## 四、代表点诊断

| 点位 | 真实末端误差(mm) | 柔性末端误差(mm) | 末端几何分类 | 主因 | 证据 |
|---|---:|---:|---|---|---|
| A2-pt05 | 4.407 | 7.042 | centerline_matches_reported_true_tip | 有效成功 | 真实末端误差已进入 5 mm 阈值。 |
| A2-pt09 | 48.210 | 49.024 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt10 | 57.852 | 58.211 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt17 | 47.486 | 48.723 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |
| A2-pt19 | 70.652 | 70.299 | centerline_matches_reported_true_tip | 阻尼过大导致停滞 | 后期步长接近零且阻尼急剧增大，外层局部模型已失效。 |

## 五、全局主因分布

| 主因 | 点数 |
|---|---:|
| 阻尼过大导致停滞 | 4 |
| 有效成功 | 1 |

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