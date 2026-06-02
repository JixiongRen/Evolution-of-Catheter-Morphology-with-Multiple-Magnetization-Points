# P2c2 真实末端位姿约束实验记录

## 实验目的

本轮只在 `c3_segment_wrench_18d` 主线上引入真实末端位姿约束。目标是验证分段物理打靶是否能同时约束导管真实末端位置和完整 SO(3) 姿态。

真实末端定义为最后刚段远端，不再使用 FK engine 当前 `tip_p_dim/tip_Q_wxyz`。后者是最后柔性段远端节点，只作为 convention drift 诊断项记录。

## 实验方案

固定配置：

- `M_list=(1,1,1)`
- `tol=1e-5`
- `max_iter=120`
- `I=[0,0,0,0,0,4,0,0]`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

候选：

- `c3_segment_wrench_18d`：旧版分段内力/内矩打靶，对末端姿态只使用轴向代理。
- `c3_pose_terminal_v1`：新版分段内力/内矩打靶，增加真实 terminal position 和 SO(3) residual。

新版残差项：

- terminal position residual：最后刚段远端位置。
- terminal SO(3) residual：`Log(R_target.T @ R_current)`。
- true rigid midpoint residual：由刚段远端反推刚段中点。
- centerline morphology residual：仅作为形态漂移辅助项。
- weak boundary residual：保留物理一致性软约束。

注意：代码审查发现 `forward_kinematics_optimized.fk_engine.query_sites()` 中 `rigid_mid_*` 命名与实际 FK residual 不一致。底层 `equilibrium_solver_nondim.py` 将 `x_rigid` 约束到 `sigma=rigid.length`，即刚段远端。因此本实验按实际残差解释处理：oracle terminal 直接取最后 `x_rigid`，true midpoint 由远端反推。

## 实验过程记录

第一次完整运行包含 FD audit，运行超过 26 分钟无输出后终止。随后分两步运行：

1. `run_p2c2_terminal_pose.py --skip_fd_audit`：完成旧 c3 与新 terminal-pose c3 的基准对比。
2. `run_p2c2_fd_reopt.py`：对新候选做 `eps=0.02`、`audit_nfev=8` 的 re-optimized FD audit。

完整命令见 `commands.sh`。

## 实验结果

FK oracle：

- `base_ok=false`
- `base_normE=8.6407e-4`
- `base_wall_s=83.69s`

旧 `c3_segment_wrench_18d`：

- Stage：B
- terminal position error：`0.344 mm`
- terminal SO(3) error：`7.93 deg`
- true rigid midpoint RMS：`0.251 mm`
- centerline RMS：`13.29 mm`
- boundary residual：`0.266`
- wall time：`579.68s`
- nfev：`16`

新 `c3_pose_terminal_v1`：

- Stage：B
- terminal position error：`0.105 mm`
- terminal SO(3) error：`0.106 deg`
- true rigid midpoint RMS：`0.159 mm`
- centerline RMS：`13.29 mm`
- boundary residual：`0.151`
- wall time：`528.57s`
- nfev：`16`

re-optimized FD audit：

- active current index：`5`
- eps：`0.02`
- audit nfev：`8`
- wall time：`292.02s`
- position relative error：`3.90`
- SO(3) relative error：`5.01`
- pose relative error：`5.01`

## 数据分析

`c3_pose_terminal_v1` 明显改善了真实末端姿态。旧 c3 已能让真实末端位置接近 FK oracle，但姿态误差接近 `8 deg`；新候选将姿态误差压到约 `0.1 deg`，同时位置误差、刚段中点误差和边界残差均更低。

轴线 RMS 仍约 `13 mm`，没有因末端姿态约束显著改善。这一误差应继续作为 morphology drift 记录，不应否定末端位姿收敛。

FD audit 仍不可信。虽然这次使用了 re-optimized shooting，而不是 fixed-state audit，但对电流扰动的 position/SO(3) 响应与 FK oracle 差异很大。因此当前候选可以作为 terminal-pose geometry candidate，但不能直接接入外层 IK 梯度链路。

## 实验结论

真实末端位姿约束已成功引入 `c3` 分段物理打靶主线。`c3_pose_terminal_v1` 达到 Stage B，并且真实末端姿态误差已低到可接受水平。

当前仍不能进入完整 IK 的原因是敏感性链路失败。下一步应优先做 implicit shooting sensitivity 或更充分的 re-optimized sensitivity，而不是继续扩展 `c0-c8` 其他候选。

## 下一步计划

1. 修正 FK engine 观测命名或新增 explicit terminal/rigid_midpoint query，避免 `rigid_mid_*` 命名继续误导 IK。
2. 在 `c3_pose_terminal_v1` 上实现 implicit sensitivity，显式考虑打靶状态 `y(I)` 对电流的响应。
3. 若 implicit sensitivity 通过，再将 residual 定义接入外层 IK。
4. 在 `M=(2,2,2)` 或 `M=(3,3,3)` 上复核 terminal pose constraint 是否保持稳定。
