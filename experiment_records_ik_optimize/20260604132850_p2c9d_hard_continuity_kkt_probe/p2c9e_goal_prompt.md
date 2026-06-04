# P2c9e Goal Prompt

```text
/goal 推进 P2c9e：soft-to-hard continuation 与 shooting 物理耦合审计，目标是解释 P2c9d hard-continuity KKT baseline 失败的根因，并给出下一步可实现的灵敏度修复路径。

Context:
- P2c9 soft continuity partially-usable，但 n3_mixed_base_i4 active coil 和 n3_mixed_rand20 失败。
- P2c9b 6D true continuous baseline 全失败。
- P2c9c base slack 未恢复 hard case。
- P2c9d hard-continuity KKT baseline 在 n3_mixed_base_i4 和 n3_mixed_rand20 均失败，constraint/boundary/objective 同时较高。

Success Criteria:
- 对 n3_mixed_base_i4 和 n3_mixed_rand20 完成 soft-to-hard continuation 曲线。
- 报告每个 continuation stage 的 terminal residual、boundary residual、continuity constraint residual、stationarity/KKT residual、condition number、wall time。
- 判断 hard continuity 失败属于 basin、conditioning、constraint conflict，还是 rigid-load coupling/model mismatch。
- 审查并记录 `_advance_rigid` 与 `rigid_state_along_dim` 中外载对刚段姿态传播的影响路径。
- 若发现外载不影响刚段 pose，只影响 distal wrench，给出修复方案：修改 rigid propagation、重定义 shooting state，或将刚段载荷积分入 pose update。

Constraints:
- 文档中文；代码注释/docstring 英文。
- 不加入 centerline oracle。
- FK oracle 只作为 secondary comparator。
- re-optimized FD 只做 audit。
- 不接入 production IK。
- 本地提交到 ik_optimize，不 push。

Checklist:
1. 实现 continuation runner：从 soft continuity penalty 开始，逐步提高 continuity weight/constraint scale，并记录指标。
2. 复用 P2c9/P2c9d 的 baseline 与 KKT residual。
3. 对刚段传播代码做只读审计，确认磁力/磁矩是否进入 pose propagation。
4. 生成 record.md、summary.json/csv、heartbeat.jsonl、failure_events.jsonl。
5. 给出下一轮具体修复方向。
```
