# A1/A2 单点磁化导管 5.0 mm 有监督 IK 正式实验报告

- 实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity`
- 目标圆：x = 60.0 mm，半径 = 100.0 mm，20 点均匀采样。
- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。
- 成功阈值：真实末端位置误差不超过 5.0 mm；同时保留 3 mm、5 mm、10 mm 参考统计。
- git 版本：`e4babc9065e77ec4f446a6884604c891c288877d`

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
| A1 | 0/0 | formal_20pt_forced | 20 | 7 | 0 | 7 | 9 | 100.001 | 28.531 | 13165.5 |
| A2 | 0/0 | formal_20pt_forced | 20 | 3 | 1 | 3 | 4 | 80.151 | 33.751 | 22688.1 |

## 分导管结果

### A1

- 门控结果：本轮强制完整 20 点，未使用门控阻断。
- 决策：已执行 20 点正式圆周实验。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity/A1_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 0 | formal_20pt | 100.001 | 100.031 | False | False | False | False | GOOD | 50 | 97.3 | max_outer_iter |
| 1 | formal_20pt | 4.939 | 7.677 | True | False | True | True | GOOD | 14 | 174.8 | reached_threshold_true_tip |
| 2 | formal_20pt | 4.396 | 7.200 | True | False | True | True | GOOD | 12 | 243.7 | reached_threshold_true_tip |
| 3 | formal_20pt | 8.617 | 11.263 | False | False | False | True | GOOD | 43 | 541.4 | reached_10mm_only |
| 4 | formal_20pt | 21.206 | 23.213 | False | False | False | False | WEAK | 26 | 897.7 | reject_stall |
| 5 | formal_20pt | 23.327 | 25.678 | False | False | False | False | GOOD | 45 | 807.6 | early_stop_or_plateau |
| 6 | formal_20pt | 4.498 | 6.990 | True | False | True | True | GOOD | 13 | 416.7 | reached_threshold_true_tip |
| 7 | formal_20pt | 4.866 | 7.494 | True | False | True | True | GOOD | 16 | 366.6 | reached_threshold_true_tip |
| 8 | formal_20pt | 24.789 | 27.423 | False | False | False | False | GOOD | 33 | 1012.8 | early_stop_or_plateau |
| 9 | formal_20pt | 3.959 | 6.935 | True | False | True | True | GOOD | 11 | 533.7 | reached_threshold_true_tip |
| 10 | formal_20pt | 4.763 | 7.728 | True | False | True | True | GOOD | 13 | 542.9 | reached_threshold_true_tip |
| 11 | formal_20pt | 4.624 | 5.238 | True | False | True | True | GOOD | 4 | 255.8 | reached_threshold_true_tip |
| 12 | formal_20pt | 8.805 | 9.038 | False | False | False | True | GOOD | 50 | 307.6 | reached_10mm_only |
| 13 | formal_20pt | 28.220 | 27.792 | False | False | False | False | GOOD | 50 | 652.5 | max_outer_iter |
| 14 | formal_20pt | 60.436 | 60.094 | False | False | False | False | GOOD | 24 | 814.2 | early_stop_or_plateau |
| 15 | formal_20pt | 91.469 | 91.519 | False | False | False | False | GOOD | 26 | 1073.2 | early_stop_or_plateau |
| 16 | formal_20pt | 40.141 | 43.119 | False | False | False | False | GOOD | 37 | 846.7 | early_stop_or_plateau |
| 17 | formal_20pt | 53.451 | 55.852 | False | False | False | False | GOOD | 26 | 820.8 | reject_stall |
| 18 | formal_20pt | 40.169 | 42.644 | False | False | False | False | GOOD | 36 | 1687.8 | early_stop_or_plateau |
| 19 | formal_20pt | 37.937 | 40.590 | False | False | False | False | GOOD | 44 | 1071.8 | early_stop_or_plateau |

### A2

- 门控结果：本轮强制完整 20 点，未使用门控阻断。
- 决策：已执行 20 点正式圆周实验。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity/A2_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 5.0mm | 3mm | 5mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---|---|---:|---:|---|
| 0 | formal_20pt | 27.503 | 30.326 | False | False | False | False | GOOD | 31 | 256.0 | early_stop_or_plateau |
| 1 | formal_20pt | 39.176 | 41.506 | False | False | False | False | GOOD | 21 | 430.2 | reject_stall |
| 2 | formal_20pt | 73.052 | 74.312 | False | False | False | False | WEAK | 15 | 501.3 | reject_stall |
| 3 | formal_20pt | 21.059 | 23.469 | False | False | False | False | GOOD | 34 | 819.7 | early_stop_or_plateau |
| 4 | formal_20pt | 4.710 | 7.277 | True | False | True | True | GOOD | 20 | 367.7 | reached_threshold_true_tip |
| 5 | formal_20pt | 4.721 | 7.551 | True | False | True | True | GOOD | 14 | 676.4 | reached_threshold_true_tip |
| 6 | formal_20pt | nan | nan | False | False | False | False | None | 21 | 2700.2 | auto_timeout |
| 7 | formal_20pt | nan | nan | False | False | False | False | None | 46 | 2700.2 | auto_timeout |
| 8 | formal_20pt | nan | nan | False | False | False | False | None | 16 | 2700.2 | auto_timeout |
| 9 | formal_20pt | 8.982 | 10.776 | False | False | False | True | GOOD | 50 | 1000.3 | reached_10mm_only |
| 10 | formal_20pt | 17.939 | 20.159 | False | False | False | False | GOOD | 50 | 521.4 | max_outer_iter |
| 11 | formal_20pt | 39.192 | 41.118 | False | False | False | False | GOOD | 44 | 1056.7 | early_stop_or_plateau |
| 12 | formal_20pt | 2.573 | 4.956 | True | True | True | True | GOOD | 4 | 477.6 | reached_threshold_true_tip |
| 13 | formal_20pt | 34.217 | 36.031 | False | False | False | False | GOOD | 33 | 1174.1 | early_stop_or_plateau |
| 14 | formal_20pt | 43.851 | 45.425 | False | False | False | False | GOOD | 30 | 1210.6 | reject_stall |
| 15 | formal_20pt | 80.151 | 80.386 | False | False | False | False | GOOD | 21 | 867.8 | reject_stall |
| 16 | formal_20pt | 51.798 | 53.574 | False | False | False | False | GOOD | 19 | 1548.9 | reject_stall |
| 17 | formal_20pt | 46.027 | 48.105 | False | False | False | False | GOOD | 37 | 837.6 | early_stop_or_plateau |
| 18 | formal_20pt | 37.815 | 39.798 | False | False | False | False | GOOD | 31 | 1766.0 | early_stop_or_plateau |
| 19 | formal_20pt | 40.998 | 43.561 | False | False | False | False | GOOD | 50 | 1075.0 | max_outer_iter |

## 输出索引

- 批次元数据：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity/batch_metadata.json`
- 汇总 JSON：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity/summary.json`
- 汇总 CSV：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712191913_A1_A2_5mm_supervised_IK_no_gravity/summary.csv`
