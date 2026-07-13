# P0 正式实验灵敏度配置追溯门禁记录

## 一、实验目的

本轮不是数值回归实验，而是 P0 工程门禁修复。目的在于确保 A1/A2 正式圆周实验入口能够显式传入并记录当前可信灵敏度配置，避免后续正式实验仍然使用旧默认灵敏度路径。

当前可信灵敏度基线为：

- 灵敏度模式：残差空间奇异值分解。
- 谱滤波方式：硬截断。
- 奇异值截断阈值：`1e-9`。

## 二、修改内容

### 2.1 正式实验入口

修改文件：

- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

新增内容：

- 增加正式实验命令行参数：灵敏度模式、线性求解方式、奇异值截断阈值、谱滤波方式、谱阻尼、推进量灵敏度方式和伴随正则项。
- 将这些参数传入 `inverse_kinematics.ik_multisite_lm_opt`。
- 将灵敏度配置写入批次元数据、单点 `formal_point_summary.json`、汇总 CSV 和中文报告。
- 在 `commands.sh` 中写出完整复现实验命令。

### 2.2 IK 主脚本

修改文件：

- `inverse_kinematics/ik_multisite_lm_opt.py`

新增内容：

- 在 `run_config.json`、`history.json` 和 `result.json` 中记录完整灵敏度配置。
- 在每次外层迭代的灵敏度信息中记录模式、求解方式、阈值、谱滤波、伴随正则项和雅可比来源。
- 在分支状态元数据中记录灵敏度配置哈希，便于后续追溯。

### 2.3 灵敏度审计脚本

修改文件：

- `DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py`

新增内容：

- 在 `sensitivity_modes.csv` 和 `finite_difference_audit.csv` 中记录线性求解方式、伴随正则项、奇异值阈值和谱阻尼。

## 三、检查命令

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  inverse_kinematics/ik_multisite_lm_opt.py

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --help | rg "sensitivity|ridge"

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  -m inverse_kinematics.ik_multisite_lm_opt --help | rg "sensitivity|ridge"

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py --help | rg "sensitivity|spectral|svd|ridge|solve"
```

## 四、检查结果

- Python 语法检查通过。
- 正式实验脚本帮助信息中已出现灵敏度参数。
- IK 主脚本帮助信息中已包含对应的下划线参数。
- 审计脚本帮助信息中已包含谱滤波、奇异值阈值、伴随正则项和求解方式参数。
- 不运行求解器的命令构造检查确认，正式脚本会向 IK 主脚本传入：
  - `--x_sensitivity_method implicit`
  - `--sensitivity_mode residual_svd`
  - `--sensitivity_solve_mode svd`
  - `--sensitivity_svd_rcond 1e-09`
  - `--sensitivity_spectral_filter hard`
  - `--sensitivity_spectral_damping 0`
  - `--ridge_adjoint 0`

## 五、结论

本轮完成了正式实验入口与结果追溯的硬门禁修复。后续 A1/A2 正式回归可以显式使用并记录当前可信灵敏度配置。

本轮尚未进行新的数值回归，因此不能据此判断 A1 或 A2 成功率改善。下一步应进入失败通道分支与小步审计。

## 六、下一步

下一轮优先执行以下审计：

- A1-pt13：电流通道 3 和 4。
- A2-pt05：电流通道 1 和 3。
- A2-pt17：电流通道 4。
- A2-pt19：电流通道 5。

建议扰动步长为 `0.0625/0.03125/0.015625 A`，并在必要时加入分段回环，以区分局部线性域过小与分支跳变。
