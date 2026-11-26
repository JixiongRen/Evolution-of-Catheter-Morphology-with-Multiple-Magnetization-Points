# A1/A2 单点磁化导管 3 mm 有监督 IK 正式实验报告

- 实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK`
- 目标圆：x = 60.0 mm，半径 = 100.0 mm，20 点均匀采样。
- 约束口径：只约束真实刚性远端末端位置，不约束末端姿态。
- 成功阈值：真实末端位置误差不超过 3 mm；10 mm 仅作为历史宽松口径参考。
- git 版本：`385d1ba5b3f8bb73cd16be5b921a95b9b7ce4efe`

## 实验设置

- 数值设置：M=5，开启重力，FK 容差 1e-5，电流限幅 20 A。
- 推进搜索：本批次使用 0.003001 m 到 0.203 m 的局部搜索范围；配置文件中的物理长度上限仍保留为 1 m。
- 自动超时：单个目标点超过 45.0 分钟未自然退出时，标记为 auto_timeout，不计入成功。
- A1：末端单磁体沿局部 z 轴轴向磁化。
- A2：末端单磁体沿局部 x 轴径向磁化。
- 门控规则：上、下、左、右 4 个代表点中至少 3 点达到 3 mm，才进入该导管 20 点正式实验；否则执行 8 点可达性诊断。

## 总览结果

| 导管 | 门控 3mm 成功数 | 决策 | 正式/诊断点数 | 3mm 成功数 | 10mm 成功数 | 最大真实误差(mm) | 平均真实误差(mm) | 总耗时(s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A1 | 1/4 | diagnostic_8pt | 8 | 1 | 1 | 124.709 | 80.461 | 11715.0 |
| A2 | 1/4 | diagnostic_8pt | 8 | 1 | 1 | 78.840 | 50.798 | 12803.2 |

## 分导管结果

### A1

- 门控结果：1/4 点达到 3 mm。
- 决策：门控未通过，未执行 20 点正式实验，已执行 8 点可达性诊断。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK/A1_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 3mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---:|---:|---|
| 0 | diagnostic_8pt | 124.709 | 128.001 | False | False | GOOD | 19 | 1796.9 | early_stop_or_plateau |
| 2 | diagnostic_8pt | 78.944 | 79.063 | False | False | GOOD | 37 | 1733.5 | early_stop_or_plateau |
| 5 | diagnostic_8pt | 76.677 | 79.972 | False | False | GOOD | 33 | 790.5 | early_stop_or_plateau |
| 7 | diagnostic_8pt | 71.143 | 74.590 | False | False | GOOD | 34 | 2243.4 | early_stop_or_plateau |
| 10 | diagnostic_8pt | 124.709 | 127.999 | False | False | GOOD | 19 | 1823.5 | early_stop_or_plateau |
| 12 | diagnostic_8pt | 95.068 | 93.560 | False | False | GOOD | 34 | 1413.5 | early_stop_or_plateau |
| 15 | diagnostic_8pt | 2.784 | 2.943 | True | True | GOOD | 12 | 713.2 | reached_3mm_true_tip |
| 17 | diagnostic_8pt | 69.654 | 69.012 | False | False | GOOD | 32 | 1200.5 | early_stop_or_plateau |

### A2

- 门控结果：1/4 点达到 3 mm。
- 决策：门控未通过，未执行 20 点正式实验，已执行 8 点可达性诊断。
- 总览图：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK/A2_target_circle_true_tip_overview.png`

| 点号 | 阶段 | 真实误差(mm) | 柔性节点误差(mm) | 3mm | 10mm | cls | 外层迭代 | 耗时(s) | 停止原因 |
|---:|---|---:|---:|---|---|---|---:|---:|---|
| 0 | diagnostic_8pt | 51.835 | 53.691 | False | False | GOOD | 50 | 1885.1 | max_outer_iter |
| 2 | diagnostic_8pt | 45.190 | 46.367 | False | False | GOOD | 22 | 1325.4 | reject_stall |
| 5 | diagnostic_8pt | 78.840 | 79.282 | False | False | WEAK | 22 | 1974.0 | reject_stall |
| 7 | diagnostic_8pt | 32.657 | 32.076 | False | False | WEAK | 27 | 2425.0 | early_stop_or_plateau |
| 10 | diagnostic_8pt | 58.715 | 58.463 | False | False | GOOD | 28 | 1457.9 | early_stop_or_plateau |
| 12 | diagnostic_8pt | 74.274 | 74.030 | False | False | GOOD | 20 | 1498.0 | reject_stall |
| 15 | diagnostic_8pt | 1.843 | 2.692 | True | True | GOOD | 10 | 766.5 | reached_3mm_true_tip |
| 17 | diagnostic_8pt | 63.030 | 62.215 | False | False | GOOD | 12 | 1471.4 | reject_stall |

## 输出索引

- 批次元数据：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK/batch_metadata.json`
- 汇总 JSON：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK/summary.json`
- 汇总 CSV：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260706001028_A1_A2_3mm_supervised_IK/summary.csv`
