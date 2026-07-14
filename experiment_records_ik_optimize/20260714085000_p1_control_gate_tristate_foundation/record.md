# P1 通道门禁三类语义基础实现记录

## 一、实验目的

本轮用于完成 P1 的基础工程改造：将原有“放行/禁用”二值通道门禁扩展为“精修可用、入盆可用、禁用”三类语义。

该改造不改变当前外层求解器的实际更新行为。旧字段 `allowed_mask` 仍表示精修阶段可用通道，因而现有严格门禁实验保持兼容。新增字段 `basin_mask` 和 `channel_roles` 用于后续 P2 两阶段求解器读取。

## 二、代码修改

修改文件：

- `inverse_kinematics/ik_control_gate.py`
- `inverse_kinematics/tests/test_control_gate.py`

主要修改：

- `ControlGate` 新增 `basin_mask`、`channel_roles` 和 `basin_step_abs_max`。
- `as_dict()` 新增 `basin_channels` 和三类角色字段。
- “可生产使用”和“可有限使用”仍进入精修可用。
- “单步可用”进入入盆可用。
- 推进量通道的“弱响应”进入入盆可用，不再被简单硬封禁。
- 电流通道的“弱响应”仍保持禁用，避免把噪声主导通道纳入控制。

## 三、验证结果

已运行语法检查：

- `inverse_kinematics/ik_control_gate.py`
- `inverse_kinematics/tests/test_control_gate.py`
- `inverse_kinematics/ik_multisite_lm_opt.py`

已运行直接调用验证，结果如下：

| 点位 | 精修可用通道 | 入盆可用通道 |
|---|---|---|
| A1-pt01 | I0-I7 | I0-I7, x |
| A1-pt12 | I0, I2-I6 | I0, I2-I7, x |
| A1-pt13 | I0, I2-I5, I7 | I0, I2-I7, x |

## 四、结论

P1 的门禁语义基础层已完成。当前结果证明：原先会破坏 A1-pt01 的推进量通道不再被门禁系统唯一解释为硬禁用，而是作为“入盆可用”通道保留给后续两阶段求解器。

本轮尚未改变外层求解器的实际更新策略，因此还不能声称 A1 收敛率已经改善。下一步应在 P2 中实现入盆阶段读取 `basin_mask`，精修阶段读取 `allowed_mask`。
