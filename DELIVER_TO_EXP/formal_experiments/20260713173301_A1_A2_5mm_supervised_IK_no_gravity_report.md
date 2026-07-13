# A1/A2 单点磁化导管 5.0 mm 有监督 IK 正式实验报告

- 实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity`
- 目标圆：x = 60.0 mm，半径 = 100.0 mm，20 点均匀采样。
- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。
- 成功阈值：真实末端位置误差不超过 5.0 mm；同时保留 3 mm、5 mm、10 mm 参考统计。
- git 版本：`991671407db4a2937fe7db3e24c36d9fbb586661`

## 实验设置

- 数值设置：M=5，关闭重力，FK 容差 1e-5，电流限幅 20 A。
- 推进搜索：本批次使用 0.003001 m 到 0.203 m 的局部搜索范围；配置文件中的物理长度上限仍保留为 1 m。
- 灵敏度设置：模式 `residual_svd`，线性求解 `svd`，奇异值阈值 `1e-09`，谱滤波 `hard`，推进灵敏度 `implicit`。
- 外层优化器设置：信赖域 `True`，真实末端优先 `True`，可靠子空间 `True`，最大回溯次数 `3`。
- 自动超时：单个目标点超过 15.0 分钟未自然退出时，标记为 auto_timeout，不计入成功。
- A1：末端单磁体沿局部 z 轴轴向磁化。
- A2：末端单磁体沿局部 x 轴径向磁化。
- 运行规则：本轮按用户要求强制执行完整 20 点正式圆周实验，不使用门控结果阻断。

## 总览结果

| 导管 | 门控阈值成功数 | 决策 | 正式/诊断点数 | 5.0mm 成功数 | 3mm 成功数 | 5mm 成功数 | 10mm 成功数 | 最大真实误差(mm) | 平均真实误差(mm) | 总耗时(s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0/0 | formal_20pt_forced | 1 | 0 | 0 | 0 | 0 | 42.967 | 42.967 | 126.3 |

## 分导管结果

### A1

- 门控结果：本轮强制完整 20 点，未使用门控阻断。
- 决策：已执行 20 点正式圆周实验。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity/A1_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 1 | formal_20pt | 42.967 | 45.818 | False | False | False | False | GOOD | 2 | 126.3 | max_outer_iter |

## 输出索引

- 批次元数据：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity/batch_metadata.json`
- 汇总 JSON：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity/summary.json`
- 汇总 CSV：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity/summary.csv`
