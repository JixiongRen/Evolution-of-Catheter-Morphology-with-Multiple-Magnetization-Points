# P2c144 两点磁控候选工作点真实基线门禁包

## 实验目的

P2c143 从既有目标闭环诊断中筛出 5 个 n2 新工作点候选。本轮建立真实基线门禁执行包，用于下一步无人值守运行。该包只验证候选是否能形成可信物理解分支，不直接发布灵敏度。

## 执行状态

本轮只生成可复现实验包，未执行长耗时求解。

## 候选与门禁

- 目标偏移：`1.0e-04 m`
- 延拓路径：`(0.0, 0.5, 1.0, 0.5, 0.0)`
- 每步最大迭代：`160`
- 通过门禁：一阶最优性残差不超过 `1e-5`，载荷传播比例不超过 `1.5`，返回状态漂移不超过 `1e-3 m`，返回位置误差不超过 `5e-6 m`。

## 长跑命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/run_p2c144_n2_candidate_baseline_gate.py --run
```

预计耗时：This command can take roughly 30-45 minutes on the current environment.

## 解释边界

该门禁即使通过，也只能说明候选工作点具备进一步做奇异值谱和第一主方向扰动重求解的资格。它不能直接证明三维末端位置灵敏度可发布。

## 下一步计划

1. 在无人值守窗口执行 `--run`。
2. 对通过候选计算末端位置灵敏度矩阵奇异值谱。
3. 只对条件数和有效秩可接受的候选做第一主方向扰动重求解。
