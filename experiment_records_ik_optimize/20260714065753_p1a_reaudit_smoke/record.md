# P1a 灵敏度审计流程烟雾测试记录

## 一、实验目的

本轮用于验证 `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py` 的 P1a 工程修正是否可用。重点检查：

- 是否生成完整运行包。
- 是否使用图形处理器。
- 是否能完成健康点的轻量有限差分审计。
- 是否能在 `--resume` 模式下跳过已完成点。
- CSV 重新读入后布尔统计是否正确。

## 二、实验设置

- 测试点：A1-pt01。
- 状态选择：`best_accept`。
- 灵敏度模式：正则化最小二乘灵敏度。
- 电流通道：第 0 路。
- 电流扰动：`0.0625 A`。
- 推进量扰动：`0.125 mm`。
- 图形处理器：`runtime_environment.json` 记录为 `cuda:0`。
- 环境变量：`XLA_PYTHON_CLIENT_PREALLOCATE=false`，`JAX_PLATFORMS=cuda`。

## 三、结果

首次运行完成 1 个点，产生 1 行灵敏度摘要和 2 行有限差分审计记录：

- 电流通道记录通过回环门禁，可进入统计。
- 推进量记录被判定为弱响应，不进入统计。
- 运行包包含 `planned_points.csv`、`completed_points.csv`、`run_status.json`、`progress.jsonl`、`runtime_environment.json`、`summary.json` 和 CSV 明细。

随后使用同一输出目录执行 `--resume`。脚本没有重新进入正运动学重放，直接跳过已完成点并输出摘要。此前发现 CSV 中 `"False"` 字符串会被 Python 误判为真值，导致摘要统计错误；已修正为显式布尔解析。修正后摘要恢复为：

- 有限差分有效记录：2 行。
- 回环一致记录：2 行。
- 弱响应记录：1 行。
- 可进入统计记录：1 行。

## 四、结论

P1a 审计脚本已具备健康点轻量审计、运行包生成和点级断点续跑能力。该测试不代表完整 A1/A2 灵敏度审计完成，仅证明新的审计流程基础功能可用。

## 五、输出文件

- `commands.sh`：首次运行命令。
- `stdout.txt`、`stderr.txt`：运行输出。
- `audit_output/`：运行包与审计结果。

