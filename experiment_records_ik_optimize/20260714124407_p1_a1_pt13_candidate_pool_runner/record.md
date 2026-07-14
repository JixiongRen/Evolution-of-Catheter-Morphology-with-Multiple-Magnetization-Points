# A1-pt13 候选池执行记录

## 一、目的

本脚本用于把 A1-pt13 的候选池运行从手动命令固化为可复现实验流程。候选池覆盖标准冷启动、前向邻点、反向邻点、确定性扰动、历史控制量重放和历史分支状态重放。

## 二、执行状态

- dry_run：`False`
- 候选数量：`8`

## 三、候选摘要

| 候选 | 中文名 | 目标点序列 | 真实末端误差(mm) | 5mm | 10mm | 停止原因 | 运行目录 |
|---|---|---|---:|---|---|---|---|
| cold_start | 标准冷启动 | 13 | 49.203 | False | False | effective_step_below_tolerance | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714124731_A1_A2_5mm_supervised_IK_no_gravity |
| pt01_forward_control | pt01 前向控制量候选 | 13 | 7.436 | False | True | effective_step_below_tolerance | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714125902_A1_A2_5mm_supervised_IK_no_gravity |
| reverse_neighbor_pt14_to_pt13 | 反向邻点候选 pt14->pt13 | 14,13 | 49.203 | False | False | effective_step_below_tolerance | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714130259_A1_A2_5mm_supervised_IK_no_gravity |
| pt01_perturb_seed1 | pt01 确定性扰动候选 1 | 13 | nan | False | False | early_stop_or_plateau | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714133113_A1_A2_5mm_supervised_IK_no_gravity |
| pt01_perturb_seed2 | pt01 确定性扰动候选 2 | 13 | 7.616 | False | True | effective_step_below_tolerance | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714134601_A1_A2_5mm_supervised_IK_no_gravity |
| historical_5778_control_only | 历史 5.778 mm 控制量重放 | 13 | 176.565 | False | False | max_outer_iter | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714135515_A1_A2_5mm_supervised_IK_no_gravity |
| historical_5778_branch_state | 历史 5.778 mm 分支状态重放 | 13 | 5.778 | False | True | max_outer_iter | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714135656_A1_A2_5mm_supervised_IK_no_gravity |
| historical_2810_control_only | 历史 2.810 mm 控制量重放 | 13 | 180.966 | False | False | max_outer_iter | /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714135156_A1_A2_5mm_supervised_IK_no_gravity |

## 四、判读原则

1. 历史分支状态重放只证明该平衡分支存在，不能计入冷启动成功率。
2. 只给控制量重放若失败，说明历史结果不能由控制量单独复现。
3. 非分支状态候选若进入 5 mm，必须再做只给控制量的重放确认。
4. 候选停在 5--10 mm 时，应进入小信赖域精修或标记为候选入盆不足。

## 五、诊断结论

1. 本轮 8 个候选均未达到真实末端 `5 mm` 成功阈值。
2. 标准冷启动停在 `49.203 mm`，反向邻点候选也回到同一误差水平，说明当前默认路径无法稳定进入 A1-pt13 的近目标分支。
3. `pt01` 前向控制量候选和确定性扰动候选 2 分别达到 `7.436 mm` 和 `7.616 mm`，证明相邻成功点控制量具有入盆价值，但仍不足以形成正式成功。
4. 历史 `5.778 mm` 控制量单独重放退化到 `176.565 mm`，历史 `2.810 mm` 控制量单独重放退化到 `180.966 mm`，说明历史近阈值形态不能由控制量单独复现。
5. 加载历史分支状态后可恢复到 `5.778 mm`，该结果只能说明历史平衡分支存在，不能计入冷启动算法成功率。
6. 候选池台账新口径显示：严格冷启动成功 `0`，分支状态重放成功 `0`，后验分支回环停滞 `4`，数值异常 `1`，GPU 无效 `0`。

## 六、后续修正指向

1. A1-pt13 下一步不应继续盲目增加外层迭代次数，而应优先处理近阈值小信赖域精修、实际步长塌缩和分支连续性门禁。
2. 若后续非分支候选进入 `5 mm`，必须执行只给控制量重放；若控制量重放失败，只能记为分支依赖成功。
3. 若多个非分支候选持续停在 `5--10 mm`，应停止刷随机种子，输出“强分支依赖、候选入盆不足和局部停滞”的阻塞证据包。
