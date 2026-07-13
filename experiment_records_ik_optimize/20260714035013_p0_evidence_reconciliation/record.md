# A1/A2 有监督逆运动学 P0 证据对账记录

## 一、实验目的

本轮不重新运行逆运动学，而是对现有 A1/A2 40 点正式结果、诊断结论和可达性采样进行口径对账。目标是固定后续修正所依据的可信基线，并明确哪些旧报告不能单独作为结论来源。

## 二、输入数据

- 合并正式结果：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40`
- 合并诊断结果：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/full40_merged/20260713060401_A1_A2_failure_diagnostics`
- 可达性采样结果：`/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/reachability_sampling/20260713123103_A1_A2_reachability_sampling`

## 三、对账结论

- 点位总数：40。
- 正式汇总表与诊断表逐点成功状态不一致数量：0。
- A1 当前合并口径为 7/20 进入 5 mm，成功点为 [1, 2, 6, 7, 9, 10, 11]。
- A2 当前合并口径为 4/20 进入 5 mm，成功点为 [4, 5, 6, 12]。
- A2-pt06 来自补跑并使用人工初始状态，应与独立求解成功率分账统计。
- 原合并目录中的 `report.md` 存在口径漂移，不得单独作为 A2 成功率结论来源。

## 四、报告冲突

- 合并目录 report.md 仍写 A2 为 3/20，未吸收 A2-pt06 补跑成功。
- 合并目录 report.md 仍保留 A2-pt06/07/08 的原始超时行。

## 五、分导管摘要

| 导管 | 5 mm 成功数 | 成功点 | 实验模式统计 | 失败主因统计 |
|---|---:|---|---|---|
| A1 | 7/20 | [1, 2, 6, 7, 9, 10, 11] | {"独立或冷启动": 2, "顺序热启动": 18} | {"阻尼过大导致停滞": 11, "有效成功": 7, "局部预测方向错误": 2} |
| A2 | 4/20 | [4, 5, 6, 12] | {"独立或冷启动": 5, "顺序热启动": 12, "人工分支种子": 1, "补跑合并": 2} | {"阻尼过大导致停滞": 16, "有效成功": 4} |

## 六、后续使用规则

- 后续 P1/P2/P3 只能引用本目录 `p0_point_reconciliation.csv`、合并 `summary.csv/summary.json` 和 `full40_merged` 诊断表作为正式基线。
- 若需要引用旧 `report.md`，必须同时说明它未吸收 A2-pt06/07/08 补跑结果。
- 独立运行、顺序热启动、人工分支种子和补跑合并必须分账统计。
- 对 A2 的阶段性结论必须同时给出“合并成功率”和“不含人工初始状态的成功率”。

## 七、输出文件

- `p0_point_reconciliation.csv`：逐点对账表。
- `p0_discrepancies.csv`：正式汇总表与诊断表不一致的点位。本轮为空表示两者逐点一致。
- `summary.json`：机器可读摘要。
- `commands.sh`：生成本记录的命令。
