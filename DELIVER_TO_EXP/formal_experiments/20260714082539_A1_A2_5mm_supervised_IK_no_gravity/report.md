# A1/A2 单点磁化导管 5.0 mm 有监督 IK 烟测报告

- 实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082539_A1_A2_5mm_supervised_IK_no_gravity`
- 目标圆：x = 60.0 mm，半径 = 100.0 mm，20 点均匀采样。
- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。
- 成功阈值：真实末端位置误差不超过 5.0 mm；同时保留 3 mm、5 mm、10 mm 参考统计。
- git 版本：`6ae088099be156aa97d370ced11f8a2642527bd7`
- 实验类型：烟测；本批次实际运行 1 个目标点，请求点号为 [1]。

## 实验设置

- 数值设置：M=5，关闭重力，FK 容差 1e-5，电流限幅 20 A。
- 外层步长限制：单步电流不超过 5 A，单步推进不超过 20 mm。
- 推进搜索：本批次使用 0.003001 m 到 0.203 m 的局部搜索范围；配置文件中的物理长度上限仍保留为 1 m。
- 初始分支状态：首个运行点从 `/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082312_A1_A2_5mm_supervised_IK_no_gravity/A1/smoke/pt_01/branch_states/states/eval_000014_final_iter_0014.npz` 读取内部平衡状态作为正运动学初值。
- 控制通道门禁：已启用控制通道门禁；门禁表 `/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/audit_output/a1_representative_channel_gate.csv`；缺失通道策略 `block_all`；允许有限级通道。
- 灵敏度设置：模式 `gn_svd`，线性求解 `svd`，奇异值阈值 `1e-09`，谱滤波 `hard`，推进灵敏度 `implicit`。
- 外层优化器设置：信赖域 `True`，真实末端优先 `True`，可靠子空间 `True`，最大回溯次数 `3`。
- 自动超时：单个目标点超过 45.0 分钟未自然退出时，标记为 auto_timeout，不计入成功。
- A1：末端单磁体沿局部 z 轴轴向磁化。
- A2：末端单磁体沿局部 x 轴径向磁化。
- 阶段门控：本轮强制执行指定目标点子集，不使用代表点结果阻断；本轮不作为正式 20 点实验结论。

## 总览结果

| 导管 | 门控阈值成功数 | 决策 | 本批次运行点数 | 5.0mm 成功数 | 3mm 成功数 | 5mm 成功数 | 10mm 成功数 | 最大真实误差(mm) | 平均真实误差(mm) | 总耗时(s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0/0 | smoke_forced | 1 | 0 | 0 | 0 | 0 | nan | nan | 22.0 |

## 分导管结果

### A1

- 阶段门控结果：本轮运行指定目标点子集，未使用代表点结果阻断。
- 控制通道门禁：已启用控制通道门禁；门禁表 `/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/audit_output/a1_representative_channel_gate.csv`；缺失通道策略 `block_all`；允许有限级通道。
- 决策：已执行烟测，不得作为正式 20 点统计。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082539_A1_A2_5mm_supervised_IK_no_gravity/A1_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 决策标签 | 纳入正式统计 | 控制通道门禁 | 状态来源 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 1 | smoke | 烟测，不纳入正式20点统计 | False | 审计通道门禁 | 初始状态 | nan | nan | False | False | False | False | None | 0 | 22.0 | early_stop_or_plateau |

## 输出索引

- 批次元数据：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082539_A1_A2_5mm_supervised_IK_no_gravity/batch_metadata.json`
- 汇总 JSON：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082539_A1_A2_5mm_supervised_IK_no_gravity/summary.json`
- 汇总 CSV：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714082539_A1_A2_5mm_supervised_IK_no_gravity/summary.csv`
