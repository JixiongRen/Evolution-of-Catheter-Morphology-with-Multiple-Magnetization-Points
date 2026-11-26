# A1/A2 有监督逆运动学失败诊断报告

## 一、实验目的

本报告执行失败诊断计划的前置门禁与历史轨迹审计，目标是解释 A1/A2 单点磁化导管在无重力、5 mm 阈值圆周任务中的失败来源，并把证据转换为后续代码修改任务。

## 二、数据来源

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity`
- 诊断输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/20260712165017_A1_A2_failure_diagnostics`
- git 版本：`385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe`
- 当前 JAX 设备：`{'timestamp': '2026-07-12 16:50:17', 'source_run': '/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity', 'git_commit': '385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe', 'python': '/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python', 'XLA_PYTHON_CLIENT_PREALLOCATE': None, 'jax': {'available': True, 'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}}`

## 三、门禁结论

- 共检查 40 个点，其中 D0 门禁完全有效的点为 0 个。
- 日志中存在 CUDA 初始化失败或 CPU 回退的点数：14。
- 求解器成功标志与真实末端成功标志不一致的点数：1。
- 中心线终点、柔性段末端和报告真实末端存在几何约定冲突的点数：40。

结论：当前数据不能直接进入完整灵敏度审计。主要原因不是某一个目标点的局部失败，而是前置定义尚未统一：外层残差优化的是柔性段末端，而正式实验报告统计的是另一套“真实末端”定义；同时现有中心线终点与报告真实末端之间通常存在约 1.5 mm 的系统偏移，说明刚段末端计算公式需要先核对。

## 四、代表点诊断

| 点位 | 真实末端误差(mm) | 柔性末端误差(mm) | 末端几何分类 | 主因 | 证据 |
|---|---:|---:|---|---|---|
| A1-pt01 | 2.640 | 4.990 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A1-pt08 | 2.060 | 4.835 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A1-pt12 | 4.053 | 1.171 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A1-pt13 | 48.202 | 52.442 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A1-pt17 | 138.856 | 137.430 | reported_true_tip_likely_shifted_beyond_centerline_end | 环境无效 | 日志显示 CUDA 初始化失败或回退 CPU，该点不能用于耗时和分支稳定性结论。 |
| A2-pt05 | 1.664 | 3.943 | reported_true_tip_likely_shifted_beyond_centerline_end | 环境无效 | 日志显示 CUDA 初始化失败或回退 CPU，该点不能用于耗时和分支稳定性结论。 |
| A2-pt09 | 5.493 | 3.598 | reported_true_tip_likely_shifted_beyond_centerline_end | 目标函数与判据不一致 | 求解器按柔性段末端判断成功，但正式报告按最后刚段远端统计成功。 |
| A2-pt10 | 3.965 | 2.769 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A2-pt17 | 54.474 | 57.986 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |
| A2-pt19 | 158.516 | 154.870 | reported_true_tip_likely_shifted_beyond_centerline_end | 末端几何约定不清 | 中心线终点、柔性段末端和报告的真实末端互不一致，需先统一物理末端定义。 |

## 五、全局主因分布

| 主因 | 点数 |
|---|---:|
| 末端几何约定不清 | 23 |
| 环境无效 | 14 |
| 运行未完整结束 | 2 |
| 目标函数与判据不一致 | 1 |

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
- `sensitivity_columns.csv`：D1 中心差分灵敏度审计占位文件；本轮因 D0 未通过未执行。
- `branch_trials.csv`：D3 多初值与分支诊断占位文件；本轮因 D0 未通过未执行。
- `reachability_samples.npz`：D4 经验可达图占位文件；本轮因 D0 未通过未执行。
- `runtime_environment.json`：当前运行环境和设备信息。
- `*_error_by_angle.png`：误差随圆周角度变化。
- `endpoint_convention_representative.png`：代表点末端约定冲突图。
- `root_cause_distribution.png`：主因分布图。

## 八、下一步建议

建议先执行一个小修复批次：统一末端定义、修正成功标志、在日志中强制记录 JAX 设备和工作空间越界信息。随后只重跑十个代表点，确认 D0 门禁通过，再开展 D1 灵敏度中心差分审计和 D3 多初值分支诊断。