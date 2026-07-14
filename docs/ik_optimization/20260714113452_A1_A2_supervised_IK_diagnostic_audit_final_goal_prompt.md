# 下一阶段 Goal Prompt

```text
/goal 在约 12 小时无人值守窗口内，基于 docs/ik_optimization/20260714113452_A1_A2_supervised_IK_diagnostic_audit_final_repair_plan.md 推进 A1/A2 有监督逆运动学修正：完成 A1 代表点主因闭合、A2 三哨兵风险封存，并形成可追溯实验报告和本地提交。

Context:
- 当前分支：ik_optimize。
- 使用 conda 环境 evolution_of_mcrs。
- 所有命令显式设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 和 JAX_PLATFORMS=cuda。
- 禁止 CPU 回退；若 GPU 不可用，该实验作废并记录。
- 固定物理条件：无重力、M_list=5、FK 容差 1e-5、真实末端为最后刚性段远端、成功阈值 5 mm、电流范围 [-20,20] A、推进范围 [0.003001,0.203] m。
- 不改变 FK 容差、网格、重力、真实末端定义、控制边界或成功阈值。
- 文档和汇报使用中文；代码注释和 docstring 使用英文。

Success Criteria:
- 重算 A1/A2 成功台账，分清冷启动、冷重启、邻点延拓、分支状态重放、热启动流程验证和辅助阶段结果。
- A1-pt01 不退化，保持 5 mm 内或给出明确回归原因。
- A1-pt13 进入 5 mm，或闭合唯一主因：分支选择、候选耗尽、步长塌缩、局部模型失效或疑似可达边界。
- A1-pt12 必须给出唯一主因，不允许只写“未收敛”。
- A2-pt05、pt17、pt19 完成风险封存：有限使用、弱平衡隔离、分支不连续、证据不足或疑似可达边界。
- 所有结论必须有证据路径、数值指标、停止原因、初值来源和是否计入算法统计。

Constraints:
- 只修改工作区内文件。
- 不 push 到远程。
- 每轮大实验在 experiment_records_ik_optimize/{YYYYMMDDHHMMSS}_{exp_name}/ 下保存 record.md、commands.sh、stdout/stderr、summary.csv/json、runtime 环境、误差曲线、电流曲线、推进量曲线、接受/拒绝曲线、最终姿态图。
- 干净交付结果复制到 DELIVER_TO_EXP/，但历史诊断结果不得混入正式成功率。
- 分支状态重放、邻点延拓、热启动流程验证不得计入冷启动成功率。
- 若进入长时间等待，每 30 分钟检查一次，等待期间保持静默。
- 每轮大实验结束后本地提交到 ik_optimize 分支。

Checklist:
1. 先检查当前证据字段是否完整：ik_stop_reason、report_stop_reason、warm_start_source、initial_branch_state、algorithm_statistics_included、branch/state hash。
2. 建立成功台账重分类脚本或汇总表，纠正历史成功率口径。
3. 针对 A1-pt13 建最小候选池：标准冷启动、pt01 前向候选、反向邻点候选、确定性小扰动候选、历史分支状态候选；历史分支候选只作诊断。
4. 实现或验证分支回环门禁：重复分支哈希、误差改善小于 0.1 mm 且有效步长跌破阈值时，标记为分支回环停滞。
5. 针对 A1-pt12 做多初值短跑和低成本可达性归因，输出唯一主因。
6. 修改外层求解器时优先处理有效步长塌缩：回溯后重算预测下降、连续拒绝后重新线性化或切换候选，不要只增大阻尼。
7. 封存 A2-pt05、pt17、pt19 风险，不运行 A2 扩点。
8. 若 A1 三点闭合，再运行 A1 3 点回归；若通过且时间充足，再进入 A1 5 点小回归。
9. 更新中文实验报告，图中文字必须正常显示。
10. 提交本地 git commit，不 push。
```
