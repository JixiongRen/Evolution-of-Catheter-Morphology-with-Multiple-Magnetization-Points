# P2c.1 物理打靶收敛候选实验记录

## 实验目的

本轮实验用于推进 Richter 风格物理打靶求解器的基础收敛能力。核心问题不是直接进入生产级 IK，而是验证在 `M=(1,1,1)` 的低离散基准上，是否存在一类低维物理状态参数，使打靶积分结果能够接近 FK oracle。

本轮新增评价口径：优先判断末端位置误差与刚段位置误差；轴线整体形态误差作为次级形态漂移记录。原因是当前 IK 工程目标首先需要末端位置和末端姿态可控，整条轴线的姿态漂移严重性低于末端收敛失败。

## 实验方案

固定 FK oracle 配置：

- `M_list=(1,1,1)`
- `tol=1e-5`
- `max_iter=120`
- `I=[0,0,0,0,0,4,0,0]`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`

候选求解模式：

- `proximal`：单组近端内力/内矩作为 6 维状态。
- `segment_wrench`：每个柔性段一组近端内力/内矩，共 18 维状态。
- `segment_twist`：每个柔性段一组常体坐标应变/曲率，共 18 维状态；该模式是几何校准辅助，不是生产候选物理模型。

评价指标：

- `tip_error`：末端点位置误差，主指标。
- `rigid_midpoint_rms_error`：刚段中点位置 RMS 误差，主指标。
- `axis_proxy_rms_error`：刚段轴向代理误差，次级姿态指标。
- `rms_centerline_error`：轴线 RMS 误差，次级形态漂移指标。
- `boundary_residual_norm`：末端内力/内矩边界残差，物理一致性风险指标。

阶段判据采用 endpoint-first 策略：

- Stage C：末端与刚段位置误差均小于导管总长 2%，且轴向代理误差小于 0.15。
- Stage B：末端与刚段位置误差均小于导管总长 5%，且轴向代理误差小于 0.30。
- Stage A：末端与刚段位置误差均小于导管总长 10%，且轴向代理误差小于 0.75。

## 实验过程记录

第一轮运行 `c0-c6` 物理内力/内矩候选，`candidate_max_nfev=12`。第二轮单独运行 `c7-c8` 几何 twist 候选，`candidate_max_nfev=16`。由于第二轮会覆盖汇总文件，已保留 `summary_first7.json/csv`，并在末尾合并为最终 `summary.json/csv`。

完整命令记录见 `commands.sh`。原始 stdout/stderr 见：

- `run_stdout.log`
- `run_stderr.log`
- `run_twist_stdout.log`
- `run_twist_stderr.log`

## 实验结果

FK oracle 本身未达到严格 `base_ok`，但残差量级较低：

- `base_ok=false`
- `base_normE=8.6407e-4`
- `base_wall_s=71.77s`

按 endpoint-first 评分，最佳候选为 `c3_segment_wrench_18d`：

- Stage：B
- `tip_error=3.04mm`
- `rigid_midpoint_rms_error=1.50mm`
- `axis_proxy_rms_error=0.0099`
- `rms_centerline_error=13.30mm`
- `boundary_residual_norm=0.2967`
- `wall_s=274.51s`

其他关键候选：

- `c1_tip_rigid_6d`：末端误差最低，为 `0.78mm`，但边界残差 `5.14`，物理一致性风险较高。
- `c2_centerline_6d`：末端与刚段误差均较低，但边界残差 `7.00`，不适合作为物理打靶主线。
- `c7_segment_twist_centerline`：轴线 RMS 较低，为 `9.46mm`，但刚段误差 `22.74mm` 且轴向代理误差 `0.682`，说明它更像几何拟合，不应作为物理打靶候选。

图像输出：

- `candidate_comparison.png`
- `M111/*_oracle_vs_shooting.png`
- `M111/*_residual_history.png`

## 数据分析

本轮结论需要区分两类误差：

1. 末端与刚段位置误差：`segment_wrench` 候选已经达到 Stage B，说明低维分段内力/内矩打靶可以让末端位置和刚段位置接近 FK oracle。
2. 轴线整体形态误差：`segment_wrench` 仍有约 `13mm` RMS 轴线偏差，但在当前 IK 目标下不应直接判定为失败。它代表形态漂移风险，需要在后续阶段通过更一致的段连接条件、collocation-compatible shooting 或末端姿态约束进一步降低。

`segment_twist` 候选证明几何自由度可以降低轴线 RMS，但它绕开了物理内力/内矩平衡，且末端姿态代理较差，因此只能作为调试参考，不能作为生产路线。

## 实验结论

在修正评价口径后，本轮达到了 P2c.1 的阶段性目标：存在物理意义更强的 `segment_wrench` 打靶候选，可以在 `M=(1,1,1)` 基准上实现末端位置和刚段位置的 Stage B 级接近。

但这不是生产级求解器，原因如下：

- FK oracle 本身 `base_ok=false`，虽然残差不大，但应记录为基准不完美。
- 当前候选依赖对 FK oracle 的初值估计，不是从任意初值稳定收敛。
- 轴线整体形态仍有明显漂移。
- 还没有完成针对最佳物理候选的电流位置/姿态敏感性审计。

## 下一步计划

下一轮应围绕 `c3_segment_wrench_18d` 开展：

1. 对最佳候选补做电流扰动下的末端位置和姿态敏感性审计。
2. 引入更接近 FK collocation 结构的段连接约束，降低轴线 RMS 漂移。
3. 将自由端边界残差作为软约束而非完全忽略，避免 `c1/c2` 这类低末端误差但物理一致性差的候选被误采信。
4. 在 `M=(2,2,2)` 或 `M=(3,3,3)` 上复核 endpoint-first 结论是否保持。
