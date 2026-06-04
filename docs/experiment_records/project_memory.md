# 项目记忆

## 运行约定

- 本仓库用于长期无人值守实验。
- 实验目标确定后，应自动持续执行，除非当前大实验完成或出现真实阻塞。
- 长时间运行任务按约定低频检查。
- 等待期间保持安静，使用低频轮询以节省 token。
- 每轮大实验结束后必须完成：
  - 更新实验记录。
  - 在本地 `dev` 分支提交本轮相关变更。
- 不允许 push 到远程。
- 文件修改范围必须限制在本仓库工作区内。

## 实验文档约定

- 通用实验记录目录：`docs/experiment_records/`。
- 通用文件名格式：`{YYYYMMDD}_exp_{exp_name}_record.md`。
- 效率优化专项记录目录：`docs/efficiency_optimization_experiment/`。
- 效率优化专项文件名格式：`{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_record.md`。
- 效率优化专项日志数据目录格式：`{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_logs_data/`。
- 每份实验记录至少包含：
  - 实验目的。
  - 实验方案。
  - 实验过程记录。
  - 实验结果。
  - 数据分析。
  - 实验结论。
  - 下一步计划。

## 自主推进规则

- 每轮实验结束后，应根据最新证据选择下一轮最合理的大实验并继续推进。
- 只有在缺少关键资产、需求矛盾、仓库状态不安全或其他真实阻塞出现时才停止。

## 优化优先级记录

- 神经网络暂时只作为 IK 初值估计和 attempt 筛选的后续辅助工具。
- 推荐的 NN 作用是预测更好的 `I_init`、`I_ref` 和 `x_init`，再由基于物理的 LM 求解器验证和精修。
- 近期不优先考虑用 NN 完全替代 FK、残差或 LM，因为当前数据量不足，且物理一致性风险较高。
- 当前高优先级任务是大幅提升 LM/FK 求解器速度：减少 full Jacobian 成本、避免重复诊断、复用编译、改善 coarse-to-fine 求解，并探索 matrix-free 或 reduced-control update。

## 效率优化实验约定

- 大规模 FK/IK runtime 优化实验使用 `efficiency-optimization-experiment` skill。
- 效率优化记录和对应日志统一存放在 `docs/efficiency_optimization_experiment/`。
- 每轮大实验必须使用唯一 ID：`eff_opt_exp_YYYYMMDDHHMMSS`。
- 每轮大实验必须记录实际 wall-clock runtime，并用于估计下一轮可比实验的等待时间。
- 每轮大规模效率优化实验结束后，将相关变更提交到本地 `dev` 分支；不要 push。

## 语言规范

- 仓库文档和实验记录统一使用中文。
- 代码注释和 docstring 统一使用英文。
- 面向用户的阶段汇报和实验记录应使用中文学术论文式表达，要求严谨、专业、条理清楚。
- 避免堆叠英文术语和缩写；必要英文术语只在首次出现时给出中文定义，后续优先使用中文名称。
- 汇报算法进展时优先解释“问题、方法、证据、结论、局限、下一步”，不要使用口语化或工程黑话式表述。

## Goal Prompt 约定

- 以后所有 goal prompt 必须控制在 4000 字符以内。
- goal prompt 应优先保留目标、成功标准、关键约束和执行清单，删去冗余背景。
