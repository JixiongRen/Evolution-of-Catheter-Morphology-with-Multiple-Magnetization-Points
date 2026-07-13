# P1 last_accept 状态选择器烟测记录

## 一、实验目的

本轮用于验证灵敏度审计脚本新增的 `last_accept` 状态选择器是否可运行，并确认输出表中能够记录状态来源字段。该实验只覆盖 A1-pt01 的一个预注册电流通道，不作为 P1 正式门禁。

## 二、代码修改

修改文件：

- `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py`

新增能力：

- 支持 `--selector last_accept`。
- 在 `sensitivity_modes.csv` 和 `finite_difference_audit.csv` 中记录 `selector`、`phase` 和 `accept` 字段。

这样后续可同时审计 `last_accept` 和 `best_accept`，避免只使用历史最小末端误差状态导致乐观偏差。

## 三、运行设置

- 输入点：A1-pt01。
- 数据来源：`DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01`。
- 选择器：`last_accept`。
- 审计通道：电流通道 3。
- 电流扰动：`0.0625 A`。
- 推进扰动：未运行。
- 灵敏度模式：残差空间奇异值分解。
- 奇异值阈值：`1e-9`。
- 回环分段数：1。
- 命令显式设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `JAX_PLATFORMS=cuda`。

完整命令见 `commands.sh`。

## 四、结果

输出文件：

- `sensitivity_modes.csv`
- `finite_difference_audit.csv`
- `summary.json`
- `stdout.log`
- `stderr.log`

关键结果：

- 被选状态：`eval_000015`。
- 状态阶段：`final`。
- 接受状态：是。
- 真实末端重载误差：`0.0 mm`。
- 轴线重载最大误差：`0.0 mm`。
- 回环门禁：通过。
- 弱响应：否。
- 灵敏度相对误差：约 `1.93%`。
- 方向夹角：约 `0.68 deg`。

## 五、结论

`last_accept` 状态选择器和状态来源字段工作正常。该结果只证明审计脚本的新选择器路径可用，不证明 A1 或 A2 的 P1 门禁已经完成。下一步应使用同一脚本分别运行 `last_accept` 和 `best_accept`，并采用预注册通道完成六个代表点审计。
