# P1 灵敏度再审计脚本烟测记录

## 一、实验目的

本轮不是完整 P1 结论，而是验证新增审计脚本能否从 P0 保存状态出发，完成以下最小闭环：

1. 读取 `branch_states` 中的状态快照。
2. 根据 `run_config.json` 重建 FK 求解器。
3. 从保存状态重放基准平衡。
4. 计算不同隐式灵敏度公式。
5. 对指定控制通道执行双侧有限差分和往返回环门禁。
6. 输出可追溯 CSV 和 JSON。

## 二、实验条件

- 测试点：A1-pt01。
- 基准状态选择：`best_accept`，对应 `eval_000014`。
- 控制通道：第 0 个线圈电流通道。
- 电流扰动：`0.25 A`。
- 推进量扰动：未启用。
- FK 条件：继承原运行配置，无重力，`M_list=5`，FK 容差 `1e-5`。
- 运行环境：`evolution_of_mcrs`，显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `JAX_PLATFORMS=cuda`。

## 三、运行命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
--out-dir experiment_records_ik_optimize/20260713150141_p1_sensitivity_reaudit_smoke \
--selector best_accept \
--current-channels 0 \
--current-steps-a 0.25 \
--x-steps-mm "" \
--sensitivity-modes solver_response,gn_svd \
--solve-mode svd \
--max-points 1
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
--out-dir experiment_records_ik_optimize/20260713150141_p1_sensitivity_reaudit_exact_smoke \
--selector best_accept \
--current-steps-a "" \
--x-steps-mm "" \
--sensitivity-modes stationarity_exact \
--solve-mode svd \
--max-points 1
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
--out-dir experiment_records_ik_optimize/20260713150141_p1_sensitivity_reaudit_exact_fd_smoke \
--selector best_accept \
--current-channels 0 \
--current-steps-a 0.25 \
--x-steps-mm "" \
--sensitivity-modes stationarity_exact \
--solve-mode svd \
--max-points 1
```

## 四、结果摘要

基准状态重载通过：

- 真实末端重载误差：`0.0 mm`。
- 中心线重载最大误差：`0.0 mm`。
- 基准真实末端误差：`4.939 mm`。

扰动回环门禁通过：

- `+0.25 A` 后回到基准控制量的真实末端回环误差：约 `0.0129 mm`。
- `-0.25 A` 后回到基准控制量的真实末端回环误差：约 `0.0075 mm`。
- 两侧中心线回环误差与末端回环误差一致，均小于 `0.5 mm`。

第 0 电流通道的有限差分响应位移约为 `2.023 mm`，不属于弱通道。但三类灵敏度对该通道的相对误差均接近 `100%`：

- 旧求解器响应版：相对误差约 `0.99999`。
- 去阻尼高斯牛顿版：相对误差约 `0.99999`。
- 精确驻点版：相对误差约 `0.99998`。

三类线性系统均显示严重病态，并触发截断奇异值求解：

- 旧求解器响应版有效秩约 `113`，条件数约 `1.14e11`。
- 去阻尼高斯牛顿版有效秩约 `93`，条件数约 `1.66e17`。
- 精确驻点版有效秩约 `93`，条件数约 `1.10e17`。

## 五、解释

本轮烟测只覆盖一个成功点的一个电流通道，不能作为 A1 的全局结论。但是它已经证明新增审计脚本可以区分两个事实：

1. 分支回环门禁可以通过，说明有限差分样本没有明显跨分支。
2. 即使回环门禁通过，三类当前灵敏度公式仍可能严重低估该电流通道的真实末端响应。

这说明下一步 P1 不能只比较不同隐式公式之间是否一致，还必须把每个通道与分支保持有限差分做直接对照，并单独记录有效秩、条件数和弱通道标签。

## 六、输出文件

- `sensitivity_modes.csv`：轻量模式灵敏度摘要。
- `finite_difference_audit.csv`：轻量模式有限差分对比。
- `summary.json`：轻量模式运行设置。
- `../20260713150141_p1_sensitivity_reaudit_exact_smoke/`：精确驻点仅灵敏度烟测。
- `../20260713150141_p1_sensitivity_reaudit_exact_fd_smoke/`：精确驻点有限差分烟测。

## 七、下一步

1. 将该脚本扩展到六个代表点，但每点先只审一个关键状态和少数通道，避免有限差分成本失控。
2. 优先检查电流通道灵敏度为何接近零：需要拆分 `dE/dI`、`dp/dz`、线性系统有效秩和截断阈值的影响。
3. 若多个代表点重复出现“回环通过但灵敏度严重失真”，应先修 `ik_diff.py` 的隐式灵敏度链，而不是进入外层优化器 P2。
