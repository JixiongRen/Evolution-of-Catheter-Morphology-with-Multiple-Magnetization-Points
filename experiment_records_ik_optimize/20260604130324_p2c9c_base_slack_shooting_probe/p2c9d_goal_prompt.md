# P2c9d Goal Prompt

```text
/goal 推进 P2c9d：hard-continuity multiple shooting + KKT/IFT sensitivity，目标是修复 P2c9/P2c9c 中 active coil 方向错配和 hard current baseline 不稳定问题，获得更可信的 shooting-based IK 任务灵敏度 dg/dI。

Context:
- P2c9 cont_1.0 soft continuity 为 partially-usable，但 n3_mixed_base_i4 active_coil_5 稳定失败，n3_mixed_rand20 baseline 不稳。
- P2c9b 纯 6D true continuous shooting 全四 case baseline 失败，自由度不足。
- P2c9c base wrench + interface slack 仍无法恢复 n3_mixed_rand20，也未修复 active coil 方向。
- 结论：软罚项/弱正则路线不够，需要把段间 wrench continuity 作为硬约束，并审计 KKT/IFT 灵敏度。

Success Criteria:
- 至少覆盖 n3_mixed_base_i4 和 n3_mixed_rand20；若时间允许覆盖四个 P2c4 case。
- 对每个 case 报告 terminal residual、boundary residual、continuity constraint residual、KKT stationarity、rank、condition number、branch consistency。
- 对通过 baseline 的 case，完成 active coil + 至少一个随机方向的 re-optimized FD vs KKT/IFT chain 审计。
- 若 n3_mixed_rand20 仍失败，必须分类为 infeasible、conditioning、basin、constraint conflict 或 model mismatch，并给下一步 continuation/pseudo-arclength 方案。

Constraints:
- 文档中文；代码注释/docstring 英文。
- clean residual 不加入 centerline oracle。
- FK oracle 仅生成 terminal target 和弱对照。
- re-optimized FD 只做 audit，不作为 runtime 梯度。
- 不接入 production IK。
- 本地提交到 ik_optimize，不 push。

Checklist:
1. 实现 hard continuity residual c(y,I)=0，约束段间 propagated wrench 与下一段 proximal wrench 一致。
2. 构造 KKT residual: grad_y L=0, c=0，并保留 terminal/boundary least-squares objective。
3. 使用 scipy root/least_squares 或增广拉格朗日求解 KKT baseline。
4. 用 IFT 解线性系统得到 dy/dI，并计算 task sensitivity。
5. 与 re-optimized FD 做 eps={0.02,0.05,0.10} 审计。
6. 输出 record.md、summary.json/csv、heartbeat.jsonl、failure_events.jsonl、sensitivity_audit.json。
```
