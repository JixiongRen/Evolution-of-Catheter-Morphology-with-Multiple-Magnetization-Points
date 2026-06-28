# P2c109 n1 刚段-柔性段载荷传播修复预案

## 实验目的

P2c99 和 P2c108 已判定单点磁控 n1 当前不可发布。本轮不重新求解，而是把既有证据转化为可执行的模型修复路线：明确应优先修复力连续方程，而不是继续调整末端权重、边界权重或力矩作用点。

## 证据链

| 来源 | 证据 | 载荷传播比例 | 结论 |
| --- | --- | ---: | --- |
| P2c84 | 近门样本一阶最优性和条件数可诊断，但载荷传播失败 | 2.769e+00 | 不是简单末端拟合失败 |
| P2c87 | 最优力矩模型可使力矩跳变约为零，但力跳变仍超门禁 | 2.769e+00 | 力矩作用点修正不足；最佳力矩模型为 中点等效载荷 |
| P2c108 | 阶段边界要求 n1 转入模型修复 | 2.769e+00 | 刚性磁体段载荷传播失配；模型耦合缺失 |

## 力连续性修复方程

当前阻断项可写为刚段远端柔性段初始内力与刚段传播后内力之间的跳变：

$$\Delta n_i = n_{i+1}^{0} - \left(n_i^{L_i} + F_i\right).$$

若 $\|\Delta n_i\| / \|F_i\| > 1.5$，则该分支不能发布为物理可信灵敏度。P2c87 表明改变力矩作用点可以改善 $\Delta m_i$，但不能改善 $\Delta n_i$，因此下一轮必须直接处理 $\Delta n_i$。

## 修复候选

| 候选 | 方程形式 | 物理含义 | 风险 | 优先级 |
| --- | --- | --- | --- | ---: |
| A 硬力连续 | `n_{i+1}^{0} = n_i^{L_i} + F_i` | 刚段外力必须由相邻柔性段内力链传递 | 可能使当前目标点不可达，导致末端误差或条件数恶化 | 1 |
| B 低维物理载荷传递参数 | `n_{i+1}^{0} = n_i^{L_i} + beta_i F_i, beta_i in [0, 1]` | 只允许沿刚段实际外力方向的可解释载荷传递比例 | 若 beta 只为拟合服务，仍可能退化成松弛变量 | 2 |
| C 接口残差分块重标定 | `r_force = (n_{i+1}^{0} - n_i^{L_i} - F_i) / max(||F_i||, F_floor)` | 使用刚段外力尺度归一化力连续残差 | 单独使用只改变权重，不修复物理链 | 3 |
| D 仅改变刚段力矩作用点 | `m_{i+1}^{0} = m_i^{L_i} + tau_i + r x F_i` | 修正力矩传播臂 | 不能修复力连续性，不应作为 n1 主修复 | 4 |

## 代码定位

| 文件 | 符号 | 作用 |
| --- | --- | --- |
| `experiment_records_ik_optimize/20260605102000_p2c10t_interface_slack_shooting/run_p2c10t_interface_slack_shooting.py` | `forward_interface_slack` | 任意接口力/力矩松弛的基线实现 |
| `experiment_records_ik_optimize/20260605104500_p2c10u_physical_direction_slack/run_p2c10u_physical_direction_slack.py` | `forward_physical_direction_slack` | 沿刚段外载方向的低维松弛候选 |
| `experiment_records_ik_optimize/20260607144044_p2c41_scaled_interface_slack_baseline_gate/run_p2c41_scaled_interface_slack_baseline_gate.py` | `load_scaled_slack_residual` | 按局部刚段载荷归一化接口残差 |
| `experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/run_p2c57_force_continuity_weight_repair.py` | `_make_split_functions` | P2c84-P2c87 近门样本恢复入口 |
| `experiment_records_ik_optimize/20260628140500_p2c87_n1_rigid_endpoint_load_closure/run_p2c87_n1_rigid_endpoint_load_closure.py` | `_audit_interfaces` | 当前载荷传播失败审计逻辑 |

## 推荐路径

第一步应实现 A 与 C 的组合：把硬力连续残差作为主物理约束，并使用刚段外力尺度归一化，避免小外力工况下尺度病态。若 A 直接导致目标不可达，再尝试 B：只允许沿实际刚段外力方向的低维载荷传递参数。D 已被 P2c87 证伪为 n1 主修复方案，只能保留为力矩项修正。

## 验收标准

1. 载荷传播比例必须从 `2.769` 降至 `1.5` 以下。
2. 一阶最优性仍需不高于 `1e-5`。
3. 条件数需低于 `1e7`，否则只做诊断。
4. 末端位置误差不能以牺牲物理闭合为代价人为压低。
5. 过门后才能重启有效子空间扰动灵敏度审计。

## 下一步计划

1. 新建实验实现 A+C 残差形式，仅在实验脚本中验证，不立即替换生产接口。
2. 对 `n1_tip_z_rand15, alpha=0.10` 复测 P2c84 相同门禁。
3. 若 A+C 失败，记录目标不可达或条件数退化，再进入 B 的低维物理载荷传递参数实验。
