# P2c9g Goal Prompt

```text
/goal 推进 P2c9g：把 shooting sensitivity 主线切换到 exact stationarity IFT，并完成 baseline recovery + shadow replay。目标是在不依赖 FK oracle 作为唯一真值的前提下，获得可用于外层 IK 的高质量 shadow gradient。

Context:
- P2c9f 证明 GN chain 是主要错配来源；exact stationarity IFT 将 n3_mixed_base_i4 active_coil_5 的 rel_err 从约 0.415 降到约 0.021。
- n2_z_pair_rand12、n1_tip_z_rand15、n3_mixed_base_i4 的 active/random 方向均通过。
- n3_mixed_rand20 仍 baseline unstable，需要单独恢复。

Success Criteria:
- 将实验级 audit harness 默认灵敏度公式改为 exact stationarity IFT，保留 GN 作为对照。
- 对 n3_mixed_base_i4、n2_z_pair_rand12、n1_tip_z_rand15 输出 usable gradient summary。
- 对 n3_mixed_rand20 执行 continuation/multi-start baseline recovery；若失败，明确分类。
- 在 n3_mixed_base_i4 上做一次 production shadow replay，记录 predicted/actual task reduction，不改默认 production IK。

Constraints:
- 文档中文；代码注释/docstring 英文。
- 不加入 centerline oracle。
- FK oracle 只作 secondary comparator。
- re-optimized FD 只做 audit，不作为 runtime 梯度。
- 不默认接 production IK。
- 本地提交到 ik_optimize，不 push。

Checklist:
1. 提炼 exact stationarity IFT 工具函数。
2. 更新 strict sensitivity gate，使用 exact IFT 作为主梯度。
3. 对 hard case 做 baseline recovery。
4. 做 shadow replay，并记录 predicted/actual reduction。
5. 输出 record.md、summary.json/csv、heartbeat.jsonl、failure_events.jsonl。
```
