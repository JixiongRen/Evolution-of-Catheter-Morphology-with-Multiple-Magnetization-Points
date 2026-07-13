# P2 外层优化器稳定化烟测记录

## 一、实验目的

本轮用于验证 P2 外层优化器稳定化代码路径可以在 GPU 环境下实际运行，并且能够保存完整追溯字段。该实验只运行 A1 单个目标点、两轮外层迭代，不评价正式收敛成功率。

## 二、修改内容

### 2.1 IK 主脚本

修改文件：

- `inverse_kinematics/ik_multisite_lm_opt.py`

新增能力：

- 可选启用 P2 信赖域策略。
- 可选启用真实末端误差优先接受判据。
- 可选将外层更新投影到任务空间可靠奇异方向。
- 可设置最多回溯次数和缩步比例。
- 可在推进量变化很小时关闭 `r_Ix` 对电流更新的线性化压制。
- 每个候选评估保存回溯编号、任务空间秩、可靠子空间维数、是否启用 `r_Ix` 线性化等信息。

### 2.2 正式实验脚本

修改文件：

- `DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py`

新增能力：

- 将 P2 外层优化器开关传入 IK 主脚本。
- 在批次元数据、单点汇总和中文报告中记录 P2 配置。

## 三、运行设置

固定条件如下：

- 无重力。
- A1 单点磁化导管。
- 目标点：`pt_01`。
- 外层最大迭代次数：2。
- 前向运动学容差：`1e-5`。
- 真实末端阈值：`5 mm`。
- 灵敏度模式：残差空间奇异值分解。
- 谱滤波方式：硬截断。
- 奇异值阈值：`1e-9`。

P2 设置如下：

- 启用信赖域。
- 启用真实末端误差优先接受。
- 启用可靠子空间。
- 最大回溯次数：3。
- 回溯缩放比例：0.5。
- 当 `|x-x_prev| <= 0.001 m` 时关闭 `r_Ix` 电流雅可比。

## 四、运行结果

输出目录：

- `DELIVER_TO_EXP/formal_experiments/20260713173301_A1_A2_5mm_supervised_IK_no_gravity/`

运行日志显示：

- JAX 设备为 `CudaDevice(id=0)`。
- 外层第 1 轮接受，任务空间秩为 2，可靠子空间维数为 2。
- 外层第 2 轮接受，任务空间秩为 3，可靠子空间维数为 3。
- 两轮均记录 `terminal_improved=True`。
- `result.json`、`run_config.json` 和 `formal_point_summary.json` 均记录了 P2 配置。

两轮后真实末端误差从初始约 `108.852 mm` 降至约 `42.967 mm`。由于本轮只允许两轮外层迭代，该结果不用于评价成功率。

## 五、检查结果

已运行：

```bash
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
```

编译检查通过。命令行帮助中已出现 P2 参数。

## 六、结论

P2 外层优化器稳定化的最小代码路径可以运行，能够使用 GPU、保存状态和记录追溯信息。下一步可以基于该实现运行 A1 十点小回归，验证是否改善真实末端误差收敛质量。
