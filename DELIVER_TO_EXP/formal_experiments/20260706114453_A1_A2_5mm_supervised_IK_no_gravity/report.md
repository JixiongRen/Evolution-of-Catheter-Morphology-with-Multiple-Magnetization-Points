# A1/A2 单点磁化导管 5.0 mm 有监督 IK 正式实验报告

- 实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity`
- 目标圆：x = 60.0 mm，半径 = 100.0 mm，20 点均匀采样。
- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。
- 成功阈值：真实末端位置误差不超过 5.0 mm；同时保留 3 mm、5 mm、10 mm 参考统计。
- git 版本：`385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe`

## 实验设置

- 数值设置：M=5，关闭重力，FK 容差 1e-5，电流限幅 20 A。
- 推进搜索：本批次使用 0.003001 m 到 0.203 m 的局部搜索范围；配置文件中的物理长度上限仍保留为 1 m。
- 自动超时：单个目标点超过 45.0 分钟未自然退出时，标记为 auto_timeout，不计入成功。
- A1：末端单磁体沿局部 z 轴轴向磁化。
- A2：末端单磁体沿局部 x 轴径向磁化。
- 运行规则：本轮按用户要求强制执行完整 20 点正式圆周实验，不使用门控结果阻断。

## 总览结果

| 导管 | 门控阈值成功数 | 决策 | 正式/诊断点数 | 5.0mm 成功数 | 3mm 成功数 | 5mm 成功数 | 10mm 成功数 | 最大真实误差(mm) | 平均真实误差(mm) | 总耗时(s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0/0 | formal_20pt_forced | 20 | 7 | 4 | 7 | 9 | 138.856 | 36.731 | 16153.8 |
| A2 | 0/0 | formal_20pt_forced_resumed | 20 | 4 | 2 | 4 | 7 | 158.516 | 39.364 | 18943.4 |

## 分导管结果

### A1

- 门控结果：本轮强制完整 20 点，未使用门控阻断。
- 决策：已执行 20 点正式圆周实验。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity/A1_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 0 | formal_20pt | 100.125 | 100.001 | False | False | False | False | GOOD | 50 | 92.9 | max_outer_iter |
| 1 | formal_20pt | 2.640 | 4.990 | True | True | True | True | GOOD | 14 | 173.2 | reached_threshold_true_tip |
| 2 | formal_20pt | 2.281 | 4.492 | True | True | True | True | GOOD | 12 | 232.5 | reached_threshold_true_tip |
| 3 | formal_20pt | 7.197 | 11.011 | False | False | False | True | GOOD | 50 | 657.6 | reached_10mm_only |
| 4 | formal_20pt | nan | nan | False | False | False | False | None | 18 | 2701.1 | auto_timeout |
| 5 | formal_20pt | 30.393 | 29.278 | False | False | False | False | GOOD | 28 | 647.8 | early_stop_or_plateau |
| 6 | formal_20pt | 3.465 | 4.470 | True | False | True | True | GOOD | 13 | 441.6 | reached_threshold_true_tip |
| 7 | formal_20pt | 11.334 | 8.187 | False | False | False | False | GOOD | 37 | 593.2 | early_stop_or_plateau |
| 8 | formal_20pt | 2.060 | 4.835 | True | True | True | True | GOOD | 6 | 260.6 | reached_threshold_true_tip |
| 9 | formal_20pt | 8.378 | 8.239 | False | False | False | True | GOOD | 33 | 1259.5 | reached_10mm_only |
| 10 | formal_20pt | 27.248 | 28.644 | False | False | False | False | GOOD | 32 | 1973.4 | early_stop_or_plateau |
| 11 | formal_20pt | 3.596 | 4.443 | True | False | True | True | GOOD | 14 | 625.9 | reached_threshold_true_tip |
| 12 | formal_20pt | 4.053 | 1.171 | True | False | True | True | GOOD | 4 | 600.5 | reached_threshold_true_tip |
| 13 | formal_20pt | 48.202 | 52.442 | False | False | False | False | WEAK | 22 | 1753.8 | reject_stall |
| 14 | formal_20pt | nan | nan | False | False | False | False | None | 8 | 2700.3 | auto_timeout |
| 15 | formal_20pt | 28.870 | 33.334 | False | False | False | False | GOOD | 38 | 402.7 | early_stop_or_plateau |
| 16 | formal_20pt | 2.891 | 4.942 | True | True | True | True | GOOD | 17 | 272.6 | reached_threshold_true_tip |
| 17 | formal_20pt | 138.856 | 137.430 | False | False | False | False | WEAK | 8 | 263.9 | reject_stall |
| 18 | formal_20pt | 126.457 | 125.758 | False | False | False | False | WEAK | 8 | 249.5 | reject_stall |
| 19 | formal_20pt | 113.117 | 113.104 | False | False | False | False | WEAK | 9 | 251.0 | reject_stall |

### A2

- 门控结果：本轮强制完整 20 点，未使用门控阻断。
- 决策：已执行 20 点正式圆周实验。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity/A2_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 0 | formal_20pt | 24.163 | 28.382 | False | False | False | False | GOOD | 23 | 116.0 | reject_stall |
| 1 | formal_20pt | 41.217 | 45.553 | False | False | False | False | WEAK | 22 | 584.0 | reject_stall |
| 2 | formal_20pt | 74.505 | 76.088 | False | False | False | False | WEAK | 16 | 175.4 | reject_stall |
| 3 | formal_20pt | 17.448 | 20.998 | False | False | False | False | GOOD | 35 | 350.0 | early_stop_or_plateau |
| 4 | formal_20pt | 3.108 | 4.427 | True | False | True | True | GOOD | 21 | 151.5 | reached_threshold_true_tip |
| 5 | formal_20pt | 1.664 | 3.943 | True | True | True | True | GOOD | 12 | 197.1 | reached_threshold_true_tip |
| 6 | formal_20pt | 9.278 | 13.105 | False | False | False | True | GOOD | 42 | 377.5 | reached_10mm_only |
| 7 | formal_20pt | 39.392 | 43.517 | False | False | False | False | WEAK | 50 | 2164.2 | max_outer_iter |
| 8 | formal_20pt | 51.529 | 54.165 | False | False | False | False | WEAK | 21 | 477.4 | reject_stall |
| 9 | formal_20pt | 5.493 | 3.598 | False | False | False | True | GOOD | 21 | 1651.9 | flex_tip_only_reached |
| 10 | formal_20pt | 3.965 | 2.769 | True | False | True | True | GOOD | 13 | 754.3 | reached_threshold_true_tip |
| 11 | formal_20pt | 7.386 | 9.901 | False | False | False | True | GOOD | 50 | 1104.9 | reached_10mm_only |
| 12 | formal_20pt | 81.016 | 80.546 | False | False | False | False | GOOD | 26 | 790.5 | early_stop_or_plateau |
| 13 | formal_20pt | 1.227 | 4.528 | True | True | True | True | GOOD | 17 | 1061.9 | reached_threshold_true_tip |
| 14 | formal_20pt | 44.742 | 47.451 | False | False | False | False | GOOD | 29 | 1584.8 | early_stop_or_plateau |
| 15 | formal_20pt | 63.800 | 65.882 | False | False | False | False | WEAK | 20 | 1823.4 | reject_stall |
| 16 | formal_20pt | 60.514 | 62.448 | False | False | False | False | WEAK | 20 | 1896.0 | reject_stall |
| 17 | formal_20pt | 54.474 | 57.986 | False | False | False | False | GOOD | 50 | 1671.4 | max_outer_iter |
| 18 | formal_20pt | 43.842 | 46.305 | False | False | False | False | GOOD | 26 | 1076.6 | early_stop_or_plateau |
| 19 | formal_20pt | 158.516 | 154.870 | False | False | False | False | WEAK | 9 | 934.8 | reject_stall |

## 输出索引

- 批次元数据：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity/batch_metadata.json`
- 汇总 JSON：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity/summary.json`
- 汇总 CSV：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706114453_A1_A2_5mm_supervised_IK_no_gravity/summary.csv`
