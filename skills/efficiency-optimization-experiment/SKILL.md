---
name: efficiency-optimization-experiment
description: 在本仓库执行大规模 FK/IK 效率优化实验时使用，要求唯一实验 ID、配套实验记录与日志数据目录、自适应等待时间、本地 dev 分支提交，以及 docs/efficiency_optimization_experiment 下的中文文档。
---

# 效率优化实验

本 workflow 用于以降低 FK/IK runtime 为目标的大规模实验。

## 适用范围

- 目标：将 IK runtime 从 30-60 分钟/组压缩到约 1 分钟/组，3 分钟/组作为可容忍上限。
- 实验记录、计划和分析文档统一使用中文。
- 代码注释和 docstring 统一使用英文。
- 主要技术路径：
  - 优化 FK 求解耗时。
  - 重构 FK 线性化方式，以及 IK 使用 FK 的方式。
  - 优先考虑 matrix-free、reduced-space、coarse-to-fine 和 compilation reuse，而不是直接用神经网络替代物理求解器。
- 神经网络仅作为后续初值估计或 attempt 筛选辅助工具。

## 必需 ID 和路径

每轮大规模效率优化实验开始时，创建一个唯一 ID：

```text
eff_opt_exp_YYYYMMDDHHMMSS
```

同一轮实验的所有产物必须使用同一个 ID：

- 实验记录：
  - `docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_record.md`
- 日志和数据：
  - `docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_logs_data/`

不要跨实验复用 ID。

## 实验记录章节

实验记录需要从实验开始维护到实验结束，至少包含：

- 实验目的。
- 基线与动机。
- 实验方案。
- 实现修改。
- 实验过程记录。
- 运行时间统计。
- 实验结果。
- 数据分析。
- 风险与回归。
- 实验结论。
- 下一步计划。

## 运行时间记录

- 记录每次运行和整轮实验的 wall-clock runtime。
- 使用上一轮可比实验的 runtime 估计下一轮等待时间。
- 当新任务明显更小或更大时，基于任务量调整等待估计。
- 等待期间保持安静，低频检查以节省 token。

## Git 规则

- 每轮大实验结束后，将相关代码、实验记录和生成的实验文档提交到本地 `dev` 分支。
- 不要 push。
- 不要提交无关的大型生成数据，除非它是本轮实验的必要组成部分。
- 如果当前不在 `dev` 分支，提交前停止并报告阻塞。

## 执行循环

1. 定义 `exp_name` 并创建唯一 `eff_opt_exp_YYYYMMDDHHMMSS`。
2. 创建实验记录文件和日志数据目录。
3. 写入实验目的、基线、方案、预估耗时和成功标准。
4. 进行范围明确的代码或参数修改。
5. 按需带显式 runtime 环境设置启动实验。
6. 根据预估 runtime 等待，并低频静默检查。
7. 收集 wall time、solver iteration count、FK call count、收敛质量和输出路径。
8. 更新分析、结论和下一步计划。
9. 在本地 `dev` 分支提交。

## 质量指标

FK/IK 提速工作必须同时跟踪速度和物理质量：

- total wall time。
- 可测量时的 compile time。
- steady-state solve time。
- FK solve count。
- IK iteration count。
- final tip error。
- raw equilibrium residual。
- interface gap。
- smoothness 或 curvature anomaly metric。
- current utilization 和 saturation。
- 可用时记录 target reachability 或 control Jacobian rank。

只有物理质量不超过实验设定容差时，speedup 才视为有效。
