# P1 分支状态重载入口实现与烟测记录

## 一、实验目的

本轮目标是补齐 A1/A2 有监督逆运动学中“同分支近解精修”的前置能力。此前 A1-pt13 只复用无门控近解的电流和推进量，初始形态跳到远离目标的另一平衡分支，最终误差达到 `175.910 mm`。这说明仅传控制量不能证明局部精修是否有效。

本轮实现从保存的 `.npz` 分支状态读取内部平衡状态，并将其作为正运动学第一次求解的初值。

## 二、代码修改

修改文件：

- `inverse_kinematics/ik_multisite_lm_opt.py`
- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

主要变化：

- 底层 IK 命令行新增 `--initial_branch_state_npz`。
- 正式 A1/A2 批处理脚本新增 `--initial-branch-state-npz`。
- 状态文件只在本批次第一个运行点使用，避免跨目标误用内部平衡状态。
- `run_config.json`、点级摘要和总表新增初始分支状态路径、状态哈希、控制量哈希、控制量差异等字段。
- 默认不传状态文件时，原有求解路径不变。

## 三、验证命令

完整命令见 `commands.sh`。

静态验证：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

入口验证：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m inverse_kinematics.ik_multisite_lm_opt --help
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --help
```

烟测对象：

- 导管：A1。
- 点位：pt13。
- 源状态：`DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz`
- 外层迭代：`0`。
- 重力：关闭。
- GPU：有效。

## 四、结果

烟测输出目录：

`DELIVER_TO_EXP/formal_experiments/20260714003850_A1_A2_5mm_supervised_IK_no_gravity/`

关键结果：

| 指标 | 数值 |
|---|---:|
| 真实末端误差 | 5.7782399566 mm |
| 初始状态哈希 | `a53053354c54...` |
| 初始 warm-start 哈希 | `a53053354c54...` |
| 源控制量与请求控制量最大电流差 | `3.39e-12 A` |
| 源控制量与请求推进量差 | `9.84e-14 m` |
| 运行设备 | GPU |
| 停止原因 | `reached_10mm_only` |

分支状态重载验证：

- 验证文件：`branch_reload_verify.json`、`branch_reload_verify.csv`。
- 验证结果：`reload_ok=True`。
- 保存状态数量：2。

## 五、结论

本轮证明新的分支状态重载入口可以复用 A1-pt13 的历史近解内部状态，使 0 次外层迭代的初始形态回到 `5.778 mm` 附近，而不是此前只复用控制量时的 `175.910 mm`。这说明后续可以开展真正意义上的“同分支局部门控精修”实验。

本轮尚未证明 A1-pt13 可进入 `5 mm`，也尚未证明门控策略有效。它只完成了 P1 的基础工程入口和最小证据闭环。

## 六、下一步

下一步应以该入口为基础，运行 A1-pt13 四组对照：

- 只复用控制量；
- 只复用内部状态；
- 同时复用控制量和内部状态；
- 完全冷启动。

只有同时复用控制量和内部状态通过重载门禁后，才允许评估局部门控小步精修是否能把 `5.778 mm` 推进到 `5 mm` 内。

