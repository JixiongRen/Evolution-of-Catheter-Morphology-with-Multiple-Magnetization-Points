# P2c3 Clean Terminal Pose Shooting 实验记录

## 实验目的

本轮验证一个关键问题：在不使用 FK oracle 的整条轴线曲线、不使用刚段中点参考的情况下，分段物理打靶是否还能拟合真实末端位姿。

只保留三类 residual：

- terminal position residual
- terminal SO(3) residual
- physical boundary residual

明确去掉：

- centerline residual
- rigid midpoint residual

因此本轮是 clean terminal-pose shooting，不再使用参考姿态曲线作为训练信号。

## 实验方案

固定配置：

- `M_list=(1,1,1)`
- `tol=1e-5`
- `max_iter=120`
- `I=[0,0,0,0,0,4,0,0]`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

候选：

- `c3_clean_terminal_pose`
- 状态变量：每段一组近端内力/内矩，共 18 维
- 权重：`tip_weight=120.0`，`so3_weight=2.0`，`boundary_weight=0.02`
- 显式设置：`centerline_weight=0.0`，`rigid_weight=0.0`

真实末端仍定义为最后刚段远端。`query_sites()` 中 `rigid_mid_*` 命名与实际 FK residual 不一致，本轮沿用 P2c2 的实际解释：底层 `x_rigid` 是刚段远端。

## 实验过程记录

运行命令见 `commands.sh`。

输出文件：

- `summary.json`
- `summary.csv`
- `M111/clean_terminal_oracle_vs_shooting.png`
- `M111/clean_centerline_pointwise_error.png`
- `M111/clean_residual_history.png`
- `M111/clean_terminal_pose_data.npz`
- `M111/clean_shooting_centerline.npy`
- `M111/clean_shooting_state_y.npy`

## 实验结果

FK oracle：

- `base_ok=false`
- `base_normE=8.6407e-4`
- `base_wall_s=76.92s`

Clean shooting：

- Stage：B
- terminal position error：`0.058 mm`
- terminal SO(3) error：`0.0236 deg`
- boundary residual norm：`0.0900`
- centerline RMS drift：`13.34 mm`
- max centerline drift：`27.23 mm`
- true rigid midpoint RMS drift：`0.495 mm`
- shooting wall time：`371.36s`
- total wall time：`453.57s`
- nfev：`16`
- scipy success：`false`

`success=false` 是因为达到 `max_nfev=16`，不是因为位姿误差不达标。最终 residual cost 已很低，末端位置和姿态都满足 Stage B。

## 数据分析

本轮结果说明：即使完全去掉 centerline 和 rigid midpoint 参考，分段物理打靶仍能把真实末端位姿拟合到很高精度。因此 P2c2 的末端位姿拟合效果并不主要依赖整条参考轴线曲线泄露。

但形态漂移仍然明显：centerline RMS drift 约 `13.34 mm`，max drift 约 `27.23 mm`。这意味着仅约束末端位姿时，导管中间形态并不唯一，打靶解可能走到另一条形态分支。

这对 IK 的含义：

- 如果工程只关心末端位姿，clean shooting 已经很有价值。
- 如果还关心整条导管安全路径或避障，必须增加路径约束、形态正则或多点观测。
- 当前仍未证明 sensitivity 可用，因此还不能直接接入外层 IK 梯度。

## 实验结论

干净实验通过。`c3_clean_terminal_pose` 在没有参考姿态曲线泄露的条件下，仍能实现真实末端位置和姿态收敛。

核心卡点从“是否依赖数据泄露”转移为：

1. terminal pose clean shooting 速度仍慢，约 `371s`。
2. centerline 形态存在较大漂移。
3. 电流敏感性仍需 implicit shooting sensitivity 验证。

## 下一步计划

1. 在 clean residual 上实现 implicit shooting sensitivity。
2. 用相同 clean residual 做 `M=(2,2,2)` 或 `M=(3,3,3)` 验证。
3. 若需要形态可控，加入非 oracle 的正则项，例如弯曲能量、状态范数、分段内力/内矩平滑，而不是 FK centerline target。
4. 优化 shooting 速度，避免每组 6 分钟以上。
