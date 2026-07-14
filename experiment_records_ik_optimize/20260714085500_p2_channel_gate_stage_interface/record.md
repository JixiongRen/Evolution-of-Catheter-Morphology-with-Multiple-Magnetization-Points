# P2 通道门禁阶段选择接口验证记录

## 一、实验目的

本轮用于验证 P2 两阶段求解器所需的门禁阶段选择接口。目标是让同一门禁表可以在精修阶段使用 `allowed_mask`，在入盆阶段使用 `basin_mask`。

本轮只实现接口与透传，不引入完整两阶段自动流程。

## 二、代码修改

修改文件：

- `inverse_kinematics/ik_multisite_lm_opt.py`
- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

主要修改：

- IK 内核新增 `--channel_gate_stage {refine,basin}`。
- 默认阶段为 `refine`，保持旧实验行为。
- `basin` 阶段使用 `basin_mask` 和 `basin_step_abs_max`。
- 正式实验入口新增 `--channel-gate-stage {refine,basin}` 并透传给 IK 内核。
- 点级摘要和 CSV 增加门禁阶段、实际激活通道、入盆通道等字段。

## 三、验证配置

- 导管：A1。
- 目标点：pt01。
- 初始状态：上一轮 A1-pt01 入盆成功状态。
- 门禁表：P1b A1 代表点门禁表。
- 门禁阶段：`basin`。
- 运行环境：GPU，`JAX_PLATFORMS=cuda`，`XLA_PYTHON_CLIENT_PREALLOCATE=false`。

完整命令见 `commands.sh`。

## 四、验证结果

输出目录：

- `DELIVER_TO_EXP/formal_experiments/20260714085145_A1_A2_5mm_supervised_IK_no_gravity/`

关键结果：

| 指标 | 结果 |
|---|---|
| 真实末端误差 | 4.383 mm |
| 是否进入 5 mm | 是 |
| 门禁阶段 | basin |
| 精修可用通道 | I0-I7 |
| 入盆可用通道 | I0-I7, x |
| 实际激活通道 | I0-I7, x |
| 运行设备 | cuda:0 |

日志中记录：

`control channel gate enabled: key=A1_pt01, stage=basin, active=['I0', ..., 'I7', 'x'], blocked=[]`

## 五、结论

门禁阶段选择接口已可用。后续 P2 两阶段求解器可以在入盆阶段显式使用推进量通道，在精修阶段继续使用保守精修通道。

本轮仍未自动执行“入盆后切换精修”的完整策略，因此不能据此声称 A1-pt12 或 A1-pt13 已改善。
