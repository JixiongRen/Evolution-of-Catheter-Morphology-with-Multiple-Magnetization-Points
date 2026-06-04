# P2c9c base wrench + interface slack 实验记录

## 实验目的

P2c9 表明 18D soft continuity 只能部分可用；P2c9b 表明纯 6D true continuous IVP shooting 自由度不足。本轮实验验证中间 formulation：`base wrench + interface slack`。它让 wrench 默认沿导管连续传播，只在段间接口允许小的 slack jump，从而兼顾物理连续性和表达能力。

## 实验方案

变量：

- `w0`：基座内力/内矩，6D。
- `δw_i`：段间接口 slack，每个内部接口 6D。
- 三段导管总变量仍为 18D，但语义不同于原始 segment-wrench 任意重置。

残差：

- terminal position residual。
- terminal SO(3) residual。
- final boundary residual。
- weak base-wrench anchor。
- slack regularization。

本轮不加入 centerline oracle，也不加入刚性段中点监督。FK oracle 只用于生成 terminal target 和初始化参考。

## 实验过程记录

运行命令见 `commands.sh`。

输出文件：

- `summary.json`
- `summary.csv`
- `baseline_summary.csv`
- `direction_summary.csv`
- `heartbeat.jsonl`
- `failure_events.jsonl`
- `sensitivity_audit.json`

## 实验结果

总体分类：`partially-usable`，但相对 P2c9 的 `cont_1.0` soft continuity 没有改善，甚至在部分 case 上退步。

主结果：

- 总审计行数：`12`。
- usable 行数：`9`。
- 可用方向数：`3/4`。
- median relative error：约 `0.146`。
- `n3_mixed_base_i4` 的 random 方向可用，median relative error 约 `0.108`。
- `n3_mixed_base_i4` 的 active coil 方向稳定失败，三档 eps 的 relative error 均约 `1.05`，cosine 约 `0.007`。
- `n3_mixed_rand20` baseline 未恢复，最佳记录为 `slack_mu=1.0`，但 `residual_norm≈0.165`、`stationarity_norm≈10944`。
- `n2_z_pair_rand12` baseline 未通过 stationarity 门，最佳记录为 `slack_mu=0.01`，`residual_norm≈3.0e-4`，但 `stationarity_norm≈0.089`。
- `n1_tip_z_rand15` 两个方向均可用，relative error 约 `0.097-0.187`。

## 数据分析

重点判断：

- hard case `n3_mixed_rand20` 的 baseline 是否能从 P2c9 的 F1 中恢复。
- slack norm 是否过大，避免退化为任意 segment reset。
- 通过 baseline 的 case 是否能在 active coil 与随机方向上通过 re-optimized FD vs implicit chain。
- 若失败，区分是自由度仍不足、slack 权重污染、conditioning blow-up，还是 sensitivity mismatch。

关键分析：

- P2c9c 的 baseline 表现介于 P2c9 soft continuity 和 P2c9b 纯 6D true continuous shooting 之间，但没有解决主问题。
- `n3_mixed_rand20` 仍为 F1 baseline unstable，说明 interface slack 并没有恢复 hard current 工况的低 stationarity 可线性化点。
- `n2_z_pair_rand12` 几何 residual 很小但 stationarity 不过关，说明“末端拟合好”仍不能等价于“灵敏度可信”。
- `n3_mixed_base_i4 active_coil_5` 的 eps plateau 稳定但方向导数完全错配，说明这不是有限差分步长问题，而是 formulation 或隐式链路问题。
- 与 P2c9 的 `cont_1.0` 相比，P2c9c 没有带来 hard case recovery，也没有修复 active coil 方向。因此它不应继续作为主线。

## 实验结论

P2c9c 没有达到长期目标。它提供了有价值的反证：`base wrench + interface slack` 比纯 6D 有更强表达能力，但仍不足以获得稳定、全工况可用的 `dg/dI`。

当前最可信结论是：soft penalty、6D true IVP、base-slack 三条轻量路线均无法完整解决问题。下一步应进入 hard-continuity multiple shooting + KKT/IFT，把段间 continuity 作为等式约束处理，而不是继续作为权重罚项或 slack 正则。

## 下一步计划

下一步应推进 P2c9d：

- 将段间 wrench continuity 作为硬等式约束。
- 使用 KKT 或增广拉格朗日系统求解内层 shooting。
- 对 KKT stationarity 与 constraint residual 建立前置门禁。
- 基于 KKT/IFT 计算 `dy*/dI`，再审计 re-optimized FD。
- 优先覆盖 `n3_mixed_base_i4 active_coil_5` 和 `n3_mixed_rand20`，因为它们是当前两类核心失败。
