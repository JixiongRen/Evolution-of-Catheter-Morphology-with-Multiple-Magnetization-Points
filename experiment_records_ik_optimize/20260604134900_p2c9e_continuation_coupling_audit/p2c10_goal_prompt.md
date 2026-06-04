# P2c10 Goal Prompt

```text
/goal 推进 P2c10：修正或验证刚段载荷到 pose propagation 的物理耦合，目标是让磁场电流对 terminal task 存在合理显式路径，并重新审计 shooting-based IK 灵敏度 dg/dI。

Context:
- P2c9e 证明 active coil 在 fixed-y 下 terminal task response 恒为 0。
- 代码审计显示 rigid_state_along_dim 中 p_sigma/Q_sigma 不依赖 f_ext_total_dim/tau_ext_total_dim，外载只更新 distal wrench。
- P2c9/P2c9c/P2c9d 的 active coil mismatch 与 hard case baseline 失败可能来自该物理耦合缺失。

Success Criteria:
- 明确当前 rigid propagation 是否符合项目物理假设；若不符合，提供实验级修复开关。
- 修复原型必须保留旧路径 fallback。
- 在 n3_mixed_base_i4 与 n3_mixed_rand20 上报告 fixed-y explicit task response、terminal pose、boundary、stationarity。
- 重新运行至少 active coil + 一个随机方向的 re-optimized FD vs implicit chain 审计。
- 若显式路径仍为 0，必须给出原因并转向更高阶刚段模型或重新定义 shooting state。

Constraints:
- 文档中文；代码注释/docstring 英文。
- 不加入 centerline oracle。
- FK oracle 只作 secondary comparator。
- 不默认替换 production FK/IK；先做实验级开关。
- 本地提交到 ik_optimize，不 push。

Checklist:
1. 审查刚段刚体受集中力/矩时的姿态传播公式。
2. 实现实验级 rigid-load pose coupling 变体，至少让 tau_ext 对 Q_sigma 有合理一阶影响。
3. 对比旧/新路径的 FK 或 shooting terminal pose。
4. 运行 P2c9 strict sensitivity 子集，检查 fixed-y explicit response 与 chain-FD 一致性。
5. 输出 record.md、summary.json/csv、heartbeat.jsonl、failure_events.jsonl。
```
