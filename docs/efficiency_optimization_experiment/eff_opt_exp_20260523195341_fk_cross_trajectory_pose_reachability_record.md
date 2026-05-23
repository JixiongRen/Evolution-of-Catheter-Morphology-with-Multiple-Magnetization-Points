# eff_opt_exp_20260523195341_fk_cross_trajectory_pose_reachability 实验记录

## 实验目的

验证旧 FK 与新 FK 的迭代轨迹是否只是收敛速度或停止点不同。具体检查：

- 新 FK 的迭代过程中，是否曾达到旧 FK 收敛终态附近的姿态。
- 旧 FK 的迭代过程中，是否曾达到新 FK 收敛终态附近的姿态。

## 实验方案

- 复用同参数 FK 对比实验中的终态代理：
  - 旧 FK 终态代理：`old_fk_tol1e-5/callback_iter_004500.npz`
  - 新 FK 终态代理：`new_fk_tol1e-5/callback_iter_002500.npz`
- 复用手动探针中保留的完整 callback 轨迹：
  - 旧 FK：`manual_fk_probe_20260523_user_case/callback_snaps/callback_iter_*.npz`
  - 新 FK：`manual_fk_probe_20260523_user_case/new_fk_default_tol/callback_snaps/callback_iter_*.npz`
- 将每个 checkpoint 提取为相同采样规则的 centerline。
- 计算每个轨迹 checkpoint 到目标终态代理的逐点误差：
  - 最大逐点误差
  - RMS 逐点误差
  - 末端误差
- 以 `10% * L_protrude = 11 mm` 作为“相近”的硬阈值：最大逐点误差不超过 11 mm 才视为达到相近姿态。

## 实验过程记录

- 2026-05-23 19:53：创建实验记录和日志目录。
- 2026-05-23 19:55：复用已有 callback 轨迹提取 centerline，不重跑 FK 求解。
- 2026-05-23 19:58：完成双向轨迹反查和误差曲线绘制。

## 运行命令

本轮使用一次性 Python 分析脚本，核心输入如下：

```text
old_final_proxy = docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/old_fk_tol1e-5/centerline_last_callback.npy
new_final_proxy = docs/efficiency_optimization_experiment/eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data/new_fk_tol1e-5/centerline_last_callback.npy
old_trajectory = docs/efficiency_optimization_experiment/manual_fk_probe_20260523_user_case/callback_snaps/callback_iter_*.npz
new_trajectory = docs/efficiency_optimization_experiment/manual_fk_probe_20260523_user_case/new_fk_default_tol/callback_snaps/callback_iter_*.npz
```

输出：

```text
docs/efficiency_optimization_experiment/eff_opt_exp_20260523195341_fk_cross_trajectory_pose_reachability_logs_data/
```

## 实验结果

10% 阈值：

- `L_protrude=0.11 m`
- `10% * L_protrude = 11 mm`
- 判定方式：最大逐点 centerline 误差小于等于 `11 mm` 才认为达到相近姿态。

### 新 FK 轨迹反查旧 FK 终态

目标：旧 FK 终态代理 `iter=4500`。

| 最优类型 | 迭代 | `normE` | 最大逐点误差 | RMS 误差 | 末端误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 按最大逐点误差最小 | `900` | `2.701e-4` | `14.82 mm` | `12.06 mm` | `14.73 mm` |
| 按 RMS 误差最小 | `1100` | `1.908e-4` | `16.79 mm` | `10.36 mm` | `16.79 mm` |

结论：新 FK 轨迹没有进入旧 FK 终态的 `11 mm` 最大逐点误差阈值。

### 旧 FK 轨迹反查新 FK 终态

目标：新 FK 终态代理 `iter=2500`。

| 最优类型 | 迭代 | `normE` | 最大逐点误差 | RMS 误差 | 末端误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 按最大逐点误差最小 | `6100` | `5.137e-6` | `27.52 mm` | `13.04 mm` | `27.52 mm` |
| 按 RMS 误差最小 | `5900` | `5.261e-6` | `27.57 mm` | `13.01 mm` | `27.57 mm` |

结论：旧 FK 轨迹没有进入新 FK 终态的 `11 mm` 最大逐点误差阈值。

结构化输出：

- `new_to_old_final_proxy.csv`
- `old_to_new_final_proxy.csv`
- `cross_trajectory_summary.json`
- `cross_trajectory_errors.png`

## 数据分析

如果新旧 FK 只是“收敛速度不同”或“在同一收敛谷内停止点不同”，至少应观察到某一条轨迹在某个迭代阶段接近另一条轨迹的终态。结果并不支持这一解释：

- 新 FK 最接近旧 FK 终态时，最大逐点误差仍为 `14.82 mm`，超过 `11 mm` 阈值。
- 旧 FK 最接近新 FK 终态时，最大逐点误差仍为 `27.52 mm`，超过 `11 mm` 阈值很多。
- 旧 FK 反查新 FK 终态时，最接近点发生在 `iter=5900..6100`，此时旧 FK 的 `normE` 已经小于新 FK 终态，但几何仍然不接近。这说明差异不是单纯由“旧 FK 收敛得更充分”造成。
- 新 FK 反查旧 FK 终态时，最接近点出现在较早的 `iter=900..1100`，随后逐渐远离旧 FK 终态。这说明新 FK 轨迹走向与旧 FK 并非同一路径上的早停关系。

因此，新旧 FK 更可能是在不同的数值目标或不同的残差加权下收敛，而不是同一物理求解过程的不同停止时刻。

## 实验结论

双向轨迹反查均未发现“某一轮迭代达到另一版本终态姿态”的证据。新 FK 的结果不是旧 FK 的早停态，旧 FK 也没有经过新 FK 的终态姿态。

结合上一轮 scale 对比，新 FK 的 nondim scale 修改仍是最可疑原因。当前新 FK 不应被视为旧 FK 的等价加速版本。

## 下一步计划

- 优先恢复旧 FK 的 nondim scale 定义，重新运行同参数对比。
- 在 scale 修复前，暂停基于新 FK 的 IK 精度结论。
- 后续所有 FK 优化必须使用逐点 centerline 轨迹门禁，而不能仅使用无量纲 `normE`。
