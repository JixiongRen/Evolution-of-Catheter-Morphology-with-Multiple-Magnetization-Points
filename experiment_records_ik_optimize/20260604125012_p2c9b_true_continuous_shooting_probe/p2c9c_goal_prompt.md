# P2c9c Goal Prompt

```text
/goal 继续推进 shooting-based IK 灵敏度 dg/dI，实施 P2c9c：base wrench + interface slack formulation，并完成严格 sensitivity audit。目标是在不使用 centerline oracle 的前提下，获得比 18D soft penalty 更物理、比 6D true continuous shooting 更有表达能力的可微 shooting 分支。

Context:
- P2c9 strict gate: cont_1.0 soft continuity 为 partially-usable；多数方向通过，但 n3_mixed_base_i4 active coil 失败，n3_mixed_rand20 baseline 不稳。
- P2c9b true continuous 6D IVP 全四 case baseline 失败，说明纯 base wrench 自由度不足。
- 下一步需要中间 formulation：主链连续传播，同时允许接口小 slack。

Success Criteria:
- 覆盖 P2c4 四个 case。
- 每 case 记录 residual、stationarity、boundary、slack norm、rank、condition number、eps plateau、branch consistency。
- 至少一个 hard case 的 baseline 从 F1 恢复到低 stationarity 可审计状态。
- 对通过 baseline 的 case，完成 active coil + 随机方向 re-optimized FD vs implicit chain 审计。
- 若仍失败，分类为 freedom 부족、slack 权重污染、branch jump、KKT mismatch 或 conditioning blow-up，并给出下一步 hard-continuity KKT 方案。

Constraints:
- 文档中文；代码注释/docstring 英文。
- clean residual 不加入 centerline oracle。
- FK oracle 仅生成目标和弱对照。
- re-optimized FD 只做 audit，不作为 runtime 梯度。
- 不接入 production IK，只输出 diagnostic/shadow-ready 结论。
- 本地提交到 ik_optimize，不 push。

Checklist:
1. 实现 forward_base_slack_wrench：变量 y=[w0, δw_1, δw_2]，段间默认传播，接口处 w_i = w_i^- + δw_i。
2. residual = terminal position + terminal SO(3) + final boundary + slack regularization。
3. 对 slack weight 做小网格扫描，优先恢复 n3_mixed_rand20 baseline。
4. 对最佳候选跑 strict sensitivity gate。
5. 输出 record.md、summary.json/csv、heartbeat.jsonl、failure_events.jsonl、sensitivity_audit.json。
```
