# A1/A2 有监督逆运动学诊断审计修正 Goal Prompt

```text
/goal 在约 12 小时无人值守窗口内，完成 A1/A2 有监督逆运动学诊断审计后的修正闭环：先硬化证据口径，再完成 A1 三点因果矩阵和 A2 三点失败闭环；若算法未达标，不中止于“失败”，必须给出唯一主因并继续按预案推进。

Context:
- 参考计划：docs/ik_optimization/20260714090907_A1_A2_supervised_IK_audit_result_correction_plan.md
- 环境：conda evolution_of_mcrs；必须使用 GPU；所有实验命令显式设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 和 JAX_PLATFORMS=cuda。
- 固定条件：无重力，M_list=5，FK 容差 1e-5，真实末端为最后刚性段远端，成功阈值 5 mm，电流 [-20,20] A，推进量 [0.003001,0.203] m。
- 当前事实：A1-pt01 是旧成功点；严格门禁禁用推进量会使 pt01 退化；两阶段烟测只证明热启动和落盘可用，不能证明算法成功；A2 主要风险是弱平衡和分支不连续。

Success Criteria:
- P0：summary/report/index/图像/命令/环境记录一致；明确区分记录修复、热启动流程验证、冷启动算法验证、辅助阶段行和最终判定行；零迭代通过不得进入算法成功统计。
- P1：完成 A1-pt01/pt12/pt13 因果矩阵：普通基线、精修门禁、入盆门禁、自动入盆到精修。pt01 必须保持 5 mm 内；pt12/pt13 无论是否成功，都必须给出唯一主因。
- P2：若 A1 失败主因仍是门禁，重审推进量角色，输出入盆可用、精修可用、仅诊断可用或禁用的明确分类。
- P3：完成 A2-pt05/pt17/pt19 失败闭环，输出风险等级：可小范围使用、弱平衡隔离、分支不连续、证据不足或疑似不可达。
- 输出中文 record.md、commands.sh、stdout/stderr、runtime_environment.json、summary.csv/json、误差曲线、电流曲线、推进量曲线、接受/拒绝曲线、最终姿态图。
- 完成后提交到本地 ik_optimize 分支，不 push。

Constraints:
- 只修改工作区内文件；文档中文；代码注释和 docstring 英文。
- 不改变 FK 容差、网格、重力设置、真实末端定义、成功阈值或控制边界来换取成功率。
- 不把热启动零迭代通过写成算法成功；不把 A1 三点结论外推到 20 点；不把 A1 策略直接套到 A2。
- 若 GPU 回退、门禁缺键、真实末端定义漂移、摘要字段缺失或索引引用不存在文件，对应实验作废并先修工程问题。

Checklist:
1. 建立独立实验目录，记录 commit、工作区状态、GPU、命令、门禁哈希、初始状态哈希。
2. 实现或检查证据口径字段：run_type、warm_start、zero_iter_pass、auxiliary_row、final_decision_row、stop_reason。
3. 跑 A1 三点四组因果矩阵，保存全部图表和结构化摘要。
4. 根据结果判断 A1 主因：门禁过保守、入盆失败、局部模型失效、分支跳变、弱平衡或疑似不可达。
5. 跑 A2 三点失败闭环，只做归因和风险封存，不做扩点。
6. 若 A1 三点未闭环，不进入 5/10/20 点；若 A2 三点未闭环，不进入 6/20 点。
7. 更新中文实验报告和计划执行记录，本地 commit。
```

