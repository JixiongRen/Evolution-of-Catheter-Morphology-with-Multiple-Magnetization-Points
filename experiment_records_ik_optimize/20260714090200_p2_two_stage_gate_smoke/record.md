# P2 两阶段门禁烟测记录

## 一、实验目的

本轮用于验证正式实验入口的两阶段门禁流程是否可用。两阶段流程定义为：

1. 入盆阶段：读取 `basin_mask`，允许推进量等入盆通道参与。
2. 精修阶段：从入盆阶段选中状态启动，读取 `allowed_mask`，只使用精修通道。

本轮只验证流程和记录口径，不用于宣称 A1-pt12 或 A1-pt13 已改善。

## 二、代码修改

修改文件：

- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

主要修改：

- 新增 `--two-stage-gate`。
- 新增 `_run_one_target_two_stage()`。
- 入盆阶段输出到 `{phase}_basin`。
- 精修阶段输出到 `{phase}_refine`。
- 批次级 `summary.csv/json` 保留入盆辅助行和精修最终行。
- 点级行增加 `two_stage_status`、入盆误差、入盆分支状态路径等字段。

## 三、验证配置

- 导管：A1。
- 目标点：pt01。
- 初始状态：上一轮 A1-pt01 入盆成功状态。
- 门禁表：P1b A1 代表点门禁表。
- 两阶段：开启。
- 环境：GPU，`JAX_PLATFORMS=cuda`，`XLA_PYTHON_CLIENT_PREALLOCATE=false`。

完整命令见 `commands.sh`。

## 四、验证结果

输出目录：

- `DELIVER_TO_EXP/formal_experiments/20260714085926_A1_A2_5mm_supervised_IK_no_gravity/`

关键结果：

| 阶段 | 门禁阶段 | 激活通道 | 真实末端误差 | 是否进入 5 mm |
|---|---|---|---:|---|
| smoke_basin | basin | I0-I7, x | 4.383 mm | 是 |
| smoke_refine | refine | I0-I7 | 4.383 mm | 是 |

日志确认：

- 入盆阶段：`stage=basin`，激活通道包括 `x`。
- 精修阶段：`stage=refine`，`x` 被屏蔽。
- 两个阶段均为 GPU 运行。

## 五、结论

两阶段门禁流程已可运行，并能在批次摘要中同时保留入盆阶段和精修阶段证据。下一步应在 A1-pt01、A1-pt12、A1-pt13 上运行代表点对照，判断该流程是否能改善原先的入盆失败和门禁过保守问题。
