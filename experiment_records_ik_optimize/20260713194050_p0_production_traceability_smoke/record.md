# P0 生产路径追溯烟测记录

## 一、实验目的

本轮用于验证生产路径的参数默认值、命令复现字段和运行环境记录是否闭环。该实验只运行 A1 的 `pt_01` 单点，并将外层迭代数限制为 1，因此不评价逆运动学收敛成功率。

## 二、代码修正

修改文件：

- `inverse_kinematics/ik_multisite_lm_opt.py`
- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`
- `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py`

修正内容：

1. 将 IK 主脚本独立运行默认灵敏度改为残差谱分解灵敏度，奇异值阈值为 `1e-9`。
2. 将正式批量入口默认灵敏度改为残差谱分解灵敏度，奇异值阈值为 `1e-9`。
3. 将灵敏度审计脚本默认纳入残差谱分解灵敏度，奇异值阈值为 `1e-9`。
4. 在 IK 点级 `result.json`、`run_config.json` 和 `runtime_profile.json` 中记录代码版本、JAX 版本、GPU 设备、`JAX_PLATFORMS` 和 `XLA_PYTHON_CLIENT_PREALLOCATE`。
5. 正式批量入口生成的 `commands.sh` 显式写入 `JAX_PLATFORMS=cuda`。

## 三、运行设置

命令见 `stdout.log` 对应输出目录中的 `commands.sh`。本轮外部命令显式设置：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cuda
```

固定条件：

- 无重力。
- A1 单点轴向磁化导管。
- 目标点：`pt_01`。
- `M_list=5`。
- 前向运动学容差 `1e-5`。
- 真实末端阈值 `5 mm`。
- 外层最大迭代数：1。

## 四、输出位置

烟测日志：

`experiment_records_ik_optimize/20260713194050_p0_production_traceability_smoke/`

正式输出：

`DELIVER_TO_EXP/formal_experiments/20260713194051_A1_A2_5mm_supervised_IK_no_gravity/`

## 五、检查结果

- 子进程返回码：`0`。
- GPU 设备：`cuda:0`，`NVIDIA GeForce RTX 5060 Laptop GPU`。
- 默认灵敏度：残差谱分解灵敏度。
- 求解模式：奇异值分解。
- 奇异值阈值：`1e-9`。
- `result.json` 已记录 `runtime_environment`。
- `run_config.json` 已记录 `runtime_environment`。
- `runtime_profile.json` 已记录 `environment` 与 FK 调用统计。
- `commands.sh` 已显式包含 `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda`。

FK 调用统计：

- 初始 FK：1 次，约 `57.51 s`。
- 试探 FK：1 次，约 `6.42 s`。
- 最终快照 FK：1 次，约 `0.03 s`。

单点运行结果：

- 真实末端误差：`98.228 mm`。
- 本轮未进入 `5 mm`，原因是外层迭代数被限制为 1，不作为收敛失败证据。

## 六、结论

P0 生产路径追溯烟测通过。后续正式实验可以从点级结果直接追溯默认灵敏度、GPU 环境、代码版本、FK 调用统计和复现实验命令。

下一步应进入失败强通道复审，重点处理 A1-pt13、A2-pt05、A2-pt17 和 A2-pt19 的分支与灵敏度归因。
