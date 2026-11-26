# A1/A2 有监督逆运动学诊断闭环补充报告

## 一、目的

本报告补充 D1、D3 和 D4 的关键证据，重点回答失败点是否已有经验可达证据，以及旧结果能否支持分支保持灵敏度审计。

## 二、数据边界

- 源实验目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40`
- 输出目录：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/closure_screen/20260713065244_A1_A2_diagnostic_closure`
- git 版本：`4b86e28f86f5205958224e14ef90288e3e656019`
- JAX 设备：`{'devices': ['cuda:0'], 'platforms': ['gpu'], 'has_gpu': True}`

## 三、经验可达性结论

### A1

- 目标自身已经达标：7 点。
- 历史经验边界可达：6 点。
- 历史经验未覆盖：4 点。
- 历史经验近可达：3 点。
- 历史探索仍未覆盖的失败点：pt14, pt15, pt16, pt19。这些点不能直接判定不可达，只能判定为现有探索证据不足。

### A2

- 历史经验边界可达：8 点。
- 历史经验未覆盖：7 点。
- 目标自身已经达标：4 点。
- 历史经验近可达：1 点。
- 历史探索仍未覆盖的失败点：pt01, pt02, pt08, pt14, pt15, pt16, pt17。这些点不能直接判定不可达，只能判定为现有探索证据不足。

## 四、分支与灵敏度结论

- A1-pt12：历史重放状态为 `ok`，末端差异 0.000 mm。
- A1-pt12：全局灵敏度相对误差 0.037，状态 `ok`；2 个已审计电流通道中 1 个未通过 10% 阈值，推进量通道状态为 `ok`。

## 五、行动结论

1. 现有历史轨迹可作为经验可达集，但它不是均匀控制空间采样；对“历史经验未覆盖”的点，结论只能是探索不足，不能写成物理不可达。
2. 若历史重放无法回到原分支，旧结果不适合做严格灵敏度审计；下一轮必须在 IK 运行时直接保存平衡状态。
3. 若分支保持差分仍显示灵敏度误差超过 10%，优先修复隐式灵敏度；若灵敏度可用，则优先修复外层信赖域、回溯和多初值分支管理。

## 六、文件索引

- `reachability_history_summary.csv`：每个目标到历史经验可达集的最近距离。
- `reachability_samples.npz`：历史外层迭代产生的末端点云。
- `sensitivity_columns.csv`：分支保持灵敏度审计结果。
- `branch_trials.csv`：历史轨迹重放分支检查。
- `A1_history_reachability.png`、`A2_history_reachability.png`：三维经验可达集图。