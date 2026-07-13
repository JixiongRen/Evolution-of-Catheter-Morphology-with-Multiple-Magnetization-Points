# A1-pt13 同分支通道策略复审

## 实验目的

本轮用于推进 A1/A2 有监督逆运动学修正主线中的最小闭环：先固定 A1-pt13 的近解分支，重审八路线圈电流和推进量通道，再生成点级策略候选表。该实验不改变正运动学物理模型、网格、容差、真实末端定义和控制边界。

## 实验设置

- 导管：A1。
- 点位：pt13。
- 重力：关闭。
- 离散化：`M_list=5`。
- 正运动学容差：`1e-5`。
- 成功目标：最后刚性段远端真实末端误差小于 `5 mm`。
- 源目录：`DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13`。
- 源近解误差：5.778240 mm。
- 源状态哈希：`a53053354c540a304ae01cfd45658ca0acfbcba192f19b92b06d68391d1a32fa`。
- 运行模式：`smoke`。

## 输出文件

- `channel_audit_full.csv`：逐通道有限差分与灵敏度对比。
- `state_replay_gate.csv`：基准状态重载门禁与灵敏度求解统计。
- `channel_policy_candidate.csv`：按通道、状态选择器、回环分段和灵敏度模式汇总的候选策略表。
- `audit_summary.json`：机器可读摘要。
- `channel_policy_summary.png/svg`：中文分类图。

## 初步结果

- 子任务数量：1。
- 成功退出子任务：1。
- 有限差分记录行数：2。
- 灵敏度记录行数：1。
- 策略候选行数：2。

## 当前结论

本记录主要保存本轮实验编排和初步输出结构。若本轮为 `smoke` 模式，只能证明脚本链路可运行，不能替代全量复审结论。全量结论必须基于 `full` 模式下的 `I0-I7+x`、多扰动尺度、多回环分段结果。

## 下一步计划

若 `smoke` 通过，应在无人值守阶段运行 `full` 模式；随后根据 `channel_policy_candidate.csv` 选择 I5-only、x-only、I5+x 三臂同分支精修消融，并判断 A1-pt13 的主要失败原因。
