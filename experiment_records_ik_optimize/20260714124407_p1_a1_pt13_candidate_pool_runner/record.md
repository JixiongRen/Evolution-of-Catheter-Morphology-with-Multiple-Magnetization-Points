# A1-pt13 候选池执行记录

## 一、目的

本脚本用于把 A1-pt13 的候选池运行从手动命令固化为可复现实验流程。候选池覆盖标准冷启动、前向邻点、反向邻点、确定性扰动、历史控制量重放和历史分支状态重放。

## 二、执行状态

- dry_run：`True`
- 候选数量：`8`

## 三、候选摘要

| 候选 | 中文名 | 目标点序列 | 真实末端误差(mm) | 5mm | 10mm | 停止原因 | 运行目录 |
|---|---|---|---:|---|---|---|---|
| cold_start | 标准冷启动 | 13 |  | None | None | None | None |
| pt01_forward_control | pt01 前向控制量候选 | 13 |  | None | None | None | None |
| reverse_neighbor_pt14_to_pt13 | 反向邻点候选 pt14->pt13 | 14,13 |  | None | None | None | None |
| pt01_perturb_seed1 | pt01 确定性扰动候选 1 | 13 |  | None | None | None | None |
| pt01_perturb_seed2 | pt01 确定性扰动候选 2 | 13 |  | None | None | None | None |
| historical_5778_control_only | 历史 5.778 mm 控制量重放 | 13 |  | None | None | None | None |
| historical_5778_branch_state | 历史 5.778 mm 分支状态重放 | 13 |  | None | None | None | None |
| historical_2810_control_only | 历史 2.810 mm 控制量重放 | 13 |  | None | None | None | None |

## 四、判读原则

1. 历史分支状态重放只证明该平衡分支存在，不能计入冷启动成功率。
2. 只给控制量重放若失败，说明历史结果不能由控制量单独复现。
3. 非分支状态候选若进入 5 mm，必须再做只给控制量的重放确认。
4. 候选停在 5--10 mm 时，应进入小信赖域精修或标记为候选入盆不足。
