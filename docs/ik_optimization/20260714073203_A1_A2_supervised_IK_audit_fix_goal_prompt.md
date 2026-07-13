# A1/A2 有监督逆运动学修正 goal prompt

```text
/goal 在无人值守阶段内，基于最新 P1b 审计结论完成 A1/A2 有监督逆运动学的下一轮受控修正与验证，重点不是直接扩大正式回归，而是验证 A1 外层稳定化是否有效，并封存 A2 弱平衡与分支风险。

Context:
- 当前分支：ik_optimize；环境：conda evolution_of_mcrs。
- 计划文档：docs/ik_optimization/20260714073203_A1_A2_supervised_IK_audit_fix_plan.md。
- 审计证据：experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/。
- 当前结论：A1 三个代表点局部灵敏度门禁后可用，但 pt12/pt13 仍远离目标，优先怀疑入盆和外层策略；A2 存在弱平衡与分支不连续，暂不进入正式回归。
- 正则化最小二乘灵敏度是唯一候选生产灵敏度；其他灵敏度只作为对照或失败案例。

Success Criteria:
- 完成 P0 生产路径闭环核查：真实末端定义、FK 容差、M_list、灵敏度模式、截断阈值、推进量处理、外层策略、GPU 环境和代码提交号均可追溯。
- 完成 A1 三点受控验证：pt01、pt12、pt13 分别对比普通外层与受控外层。
- 受控外层至少包含：可信通道子空间、真实末端误差优先接受、有限步长信赖域、必要时回溯重算、合法停止原因。
- pt01 不退化并保持 5 mm 内；pt12/pt13 若未成功，必须给出主根因：通道不足、入盆失败、弱响应、分支不连续、FK 弱收敛或疑似不可达。
- A2 只做风险封存：pt05、pt17、pt19 明确标记为有限通道可用、弱平衡隔离、分支不连续、仅诊断可用或证据不足，不进入正式小回归。
- 输出中文 record.md、commands.sh、stdout/stderr、runtime_environment.json、summary.csv/json、history、图像和失败归因表。
- 每轮大实验完成后提交到本地 ik_optimize 分支，不 push。

Constraints:
- 不放宽 FK 容差 1e-5、M_list=5、5 mm 成功阈值。
- 所有实验显式设置 XLA_PYTHON_CLIENT_PREALLOCATE=false 和 JAX_PLATFORMS=cuda；GPU 不可用则实验作废。
- 文档和汇报使用中文学术表达；代码注释和 docstring 使用英文。
- 不把 A1 代表点门禁外推到全部 20 点；不把 A2 结果套用 A1 策略；不宣称灵敏度问题已彻底解决。
- goal prompt 不超过 4000 字符；等待长任务时少输出。

Checklist:
1. 创建唯一实验目录，记录命令、环境、日志、摘要、图像和中文报告。
2. 核查正式入口、审计入口和绘图入口是否均使用最后刚性段远端作为真实末端。
3. 固化 A1 三点通道门禁，明确其只绑定当前源状态。
4. 运行 A1 三点普通外层基线与受控外层对照。
5. 绘制误差迭代、电流/推进量变化、接受/拒绝统计、最终姿态与目标圆环。
6. 若 A1 受控外层无改善，调用多智能体分析是否应转入入盆策略或推进量通道复审。
7. 封存 A2 风险表，不进入 A2 正式回归。
8. 更新计划状态和实验记录，提交本地 git。
```

