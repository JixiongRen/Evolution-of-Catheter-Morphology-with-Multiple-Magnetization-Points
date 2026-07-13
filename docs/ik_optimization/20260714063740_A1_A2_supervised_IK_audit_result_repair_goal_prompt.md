# A1/A2 有监督逆运动学审计修正 goal prompt

```text
/goal 在无人值守阶段内完成 A1/A2 有监督逆运动学诊断审计修正，并拿到可追溯的代表点结论。

Context:
- 当前分支为 ik_optimize，使用 conda 环境 evolution_of_mcrs。
- 文档参考：docs/ik_optimization/20260714063740_A1_A2_supervised_IK_audit_result_repair_plan.md。
- 可信基线：A1 为 7/20 进入 5 mm；A2 合并口径为 4/20，其中 A2-pt06 需与独立求解分账。
- P1 部分审计已完成 A1-pt01、A1-pt12、A1-pt13、A2-pt05；A2-pt17 停滞，A2-pt19 未完成。
- 当前证据显示：只有正则化最小二乘灵敏度可作为候选生产灵敏度；其他模式只保留为对照。

Success Criteria:
- 每轮输出完整运行包：record.md、commands.sh、runtime_environment.json、planned_points.csv、completed_points.csv、run_status.json、summary.csv/json 和 failures/。
- 灵敏度审计脚本支持断点续跑、通道级或扰动级落盘、单点或单通道超时记录。
- 六个代表点 A1-pt01、A1-pt12、A1-pt13、A2-pt05、A2-pt17、A2-pt19 均获得唯一主结论。
- 每个结论必须区分：生产可用、仅诊断可用、弱平衡无效、分支不连续、证据不足。
- 外层求解器能读取通道策略表，只使用通过审计的可信通道，并输出合法停止原因。
- A1 三点回归完成：pt01 不退化，pt12/pt13 给出可解释结果；若 pt13 存在近解，必须验证其能否稳定精修到 5 mm 内。

Constraints:
- 不放宽 FK 容差、离散化网格和 5 mm 成功阈值。
- 不修改工作区外文件，不 push。
- 实验命令显式设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 和 JAX_PLATFORMS=cuda。
- 若 GPU 不可用，实验作废并记录原因。
- 文档和实验报告使用中文；代码注释和 docstring 使用英文。
- goal prompt、报告和汇报尽量用中文学术论文语言，必要英文名词需附中文解释。
- 每轮大实验结束后提交到本地 ik_optimize 分支。
- 长任务等待期间少输出，按既定间隔检查。

Checklist:
1. 为本轮创建独立实验目录，保存命令、日志、CSV/JSON、图像和 record.md。
2. 建立计划点、完成点、失败原因和结束哨兵；缺失点必须有结构化原因。
3. 修复 DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py：断点续跑、单项超时、失败行落盘、字段口径固定。
4. 对已完成四点生成正式中文摘要；对 A2-pt17/A2-pt19 分点续跑审计。
5. 只把正则化最小二乘灵敏度纳入候选生产策略，其他模式写入对照结论。
6. 生成绑定控制量哈希和内部状态哈希的通道策略表。
7. 接入外层 IK：可信通道子空间、同分支门禁、真实末端优先接受、合法停止原因。
8. 运行 A1 三点回归并绘制误差曲线、控制量变化和最终姿态。
9. 若失败，调用多智能体分析原因，修正计划后继续推进，不因单点失败直接退出。
10. 更新中文实验报告和计划状态，提交本地 git。
```
