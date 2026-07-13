# 后续阶段性 goal prompt

以下提示词控制在 4000 字以内，可直接复制使用。

```text
/goal 在无人值守窗口内完成 A1/A2 有监督逆运动学诊断审计修正的下一阶段，目标是建立可信证据链、修复审计口径、完成失败通道复审，并形成可进入外层优化器修正的明确结论。

Context:
- 当前分支为 ik_optimize，使用 conda 环境 evolution_of_mcrs。
- 文档依据：docs/ik_optimization/20260713202616_A1_A2_supervised_IK_audit_correction_full_plan.md。
- 固定物理条件：无重力，M_list=5，FK 容差 1e-5，电流范围 [-20,20] A，推进范围 [0.003001,0.203] m。
- 目标误差统一使用最后刚段远端，即真实导管末端；柔性段末端只作为兼容观测点。
- 所有命令显式设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 和 JAX_PLATFORMS=cuda。若 CPU 回退，该实验作废。
- 文档用中文；代码注释和 docstring 用英文。
- 所有 goal prompt 必须少于 4000 字。

Success Criteria:
- 修复审计字段语义，新增或输出 used_in_stats、branch_consistent、weak_gate_triggered、displacement_gate_mm、selector、step_family。
- 明确区分“计算成功”和“可用于灵敏度统计”。
- 补全 last_accept 与 best_accept 双状态选择器复审。
- 对 A1-pt13 电流 3/4、A2-pt05 电流 1/3、A2-pt17 电流 4、A2-pt19 电流 5 给出唯一主归因：分支不稳、弱响应、线性窗口过窄、谱截断失配、可有限使用或可生产使用。
- 弱响应样本不得只用相对误差定罪，必须同时报告绝对误差和方向一致性。
- 回环失败样本不得进入同分支灵敏度统计。
- 给出冷启动与热启动的代表点对照，至少覆盖 A1 pt01/pt12/pt13。
- 输出中文报告、聚合 CSV/JSON、关键图像，并记录代码提交、运行点数量、请求点数量、GPU、耗时和 FK 调用次数。

Constraints:
- 只修改工作区内文件，不 push。
- 不把代表点子集称作完整 20 点正式实验。
- 不直接扩大到 A1/A2 全量 40 点，除非 P0/P1/P1.5 均通过。
- 不把 solver_response 和 stationarity_exact 作为生产灵敏度，只保留诊断用途。
- residual_svd 可作为生产候选，但必须经过回环、强响应和步长一致性门禁。
- 每轮实验结束后本地提交到当前分支。
- 等待长任务时降低输出频率，节省 token。

Checklist:
1. 建立本轮实验目录，记录 commands.sh、stdout/stderr、summary、图像和 record.md。
2. 完成 P-1 口径修正：报告中区分代表点子集与全量点，记录请求点和实际运行点。
3. 完成 P0：审计字段、真实末端一致性断言、GPU 与版本追溯。
4. 完成 P1：失败强通道双选择器复审和通道级归因表。
5. 完成 P1.5：对失败点做分支签名、冷/热启动对照和小规模可达性检查。
6. 根据结果决定是否进入 P2；若进入，先拆分 P2 开关，修正推进耦合项开关和接受准则，不整包推广。
7. 汇总根因、修正建议、风险边界和下一轮实验方案。
8. 本地提交所有必要代码、文档和实验记录。
```
