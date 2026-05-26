# FK 分支保持提速实验记录

## 实验目的

在固定 FK case 和 `tol=1e-5` 下，验证 cached/no-output、正定线性求解器和 continuation 是否能在不明显偏离当前 golden 姿态的前提下实现 3-4x 提速。

## 实验方案

- golden baseline：`experiment_records_fk_optimize/20260525094651_fk_new_tol_sweep/tol_1e-05/`。
- 固定 `tol=1e-5`、`max_iter=100000`、`lm_damping=1e-1`、`jac_method=fwd`。
- 每个候选保存最终中心线，并计算相对 golden 的逐点误差。
- `max_point_error <= 0.011m` 为目标 10% 门禁；同时报告 0.5mm、1mm、2mm 更严格参考。

## 实验过程记录

- 2026-05-26 11:43:32：完成候选矩阵运行和汇总。

## 实验结果

| case | class | wall(s) | speedup | iter | normE | max point err(mm) | end err(mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A1_cached_direct_solve` | branch-preserving result | 97.31 | 1.80x | 2255 | 9.992e-06 | 0.002 | 0.002 |
| `B1_cached_direct_pos` | branch-preserving result | 94.06 | 1.86x | 2255 | 9.992e-06 | 0.002 | 0.002 |
| `B2_cached_direct_cholesky` | branch-preserving result | 89.39 | 1.96x | 2255 | 9.992e-06 | 0.002 | 0.002 |
| `F1_fused_cholesky_direct` | production-candidate result | 40.69 | 4.31x | 2254 | 9.992e-06 | 0.002 | 0.002 |
| `F2_engine_fused_cholesky_direct` | production-candidate result | 40.88 | 4.29x | 2254 | 9.992e-06 | 0.002 | 0.002 |
| `E1_direct_cholesky_scale125` | branch-preserving result | 180.03 | 0.97x | 4717 | 9.969e-06 | 0.002 | 0.002 |
| `D1_short_continuation_solve` | speed-only result | 59.07 | 2.97x | 1373 | 9.855e-06 | 33.171 | 33.171 |
| `D2_short_continuation_pos` | speed-only result | 55.91 | 3.13x | 1373 | 9.855e-06 | 33.171 | 33.171 |

## 数据分析

最快可信候选为 `F2_engine_fused_cholesky_direct`，耗时 `40.88s`，相对 golden 提速 `4.29x`，最大逐点误差 `0.002mm`。该候选使用生产入口 `linear_solver=dense_fused_cholesky`，不是实验脚本内的特例路径。

`F1_fused_cholesky_direct` 是同一 fused JIT 思路的脚本内原型，耗时 `40.69s`，与 F2 几何结果一致。`F2` 证明该方法已接入 `ForwardKinematicsEngine` 和 cached solver。

`A1/B1/B2` 表明单纯 no-output cached path 和正定求解器只能达到约 `1.8-2.0x`。`D1/D2` 虽然达到约 `3x`，但相对 golden 的最大逐点误差约 `33.17mm`，超过目标 10% 门禁，因此只能标记为 speed-only。`E1` 的简单过松弛步长会增加迭代数，不是可用方向。

收敛过程图已补充：

- `F2_engine_fused_cholesky_direct/convergence_centerline_every_50.png`
- `F2_engine_fused_cholesky_direct/convergence_residual_every_50.png`

该 trace 使用带 callback 的 direct LM 路径做诊断采样，只用于展示同一 golden 分支的形态演进；速度结论以 `F2_engine_fused_cholesky_direct/summary.json` 为准。

## 实验结论

目标达成。`F2_engine_fused_cholesky_direct` 在固定 `tol=1e-5`、`lm_damping=1e-1`、`jac_method=fwd` 和同一物理配置下，将 steady solve time 从 golden `175.21s` 降到 `40.88s`，提速 `4.29x`；最终 `normE=9.992e-06`，最大逐点中心线误差 `0.002mm`，远低于 10% 门禁。

主要收益来自 fused JIT LM：将 dense Cholesky LM 循环放入单个 XLA while loop，减少每轮 Python/GPU 同步开销；迭代数基本不变，说明它是运行时结构优化，不是通过改变收敛目标或跳转姿态分支获得提速。

## 下一步计划

- 对 `linear_solver=dense_fused_cholesky` 做多工况 smoke test，确认不同磁体数量、电流和 M_list 下的稳定性。
- 若需要进一步压缩 cold start，总结 XLA 编译缓存或常驻进程复用方案；当前 first-call compile 约 `360.91s`，steady solve 已达标。
