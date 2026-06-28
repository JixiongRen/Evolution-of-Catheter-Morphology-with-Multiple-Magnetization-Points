# P2c98 n2 eps=0.0125 随机电流方向批量复核记录

## 实验目的

P2c95 和 P2c96 已证明 n2 受信任分支在固定任务模态方向和固定随机方向集合上通过电流扰动重求解一致性。本轮使用 20 个固定随机 8 维电流方向，检验 `eps=0.0125` 的发布级线性化证据是否依赖原方向集合。

## 实验设置

- 分支：n2 受信任分支。
- 任务：末端三维位置。
- 电流扰动尺度：`0.0125`。
- 随机种子：`2026062821`。
- 随机方向数：`20`。
- 发布门禁：线性化相对误差 `< 0.050`，正负扰动重求解均成功。

## 基线门禁

- 基线是否通过：`True`
- 末端位置误差：`1.402e-06 m`
- 一阶最优性：`2.232e-07`
- 载荷传播比例：`7.409e-01`
- 任务条件数：`3.228e+02`

## 扰动重求解结果

| 方向 | 状态 | 相对误差 | 方向余弦 | 有限差分范数 | 预测范数 | 正向成功 | 负向成功 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| rand_current_00 | publishable | 4.021e-05 | 1.000e+00 | 4.896e-07 | 4.896e-07 | True | True |
| rand_current_01 | publishable | 4.240e-05 | 1.000e+00 | 9.327e-07 | 9.327e-07 | True | True |
| rand_current_02 | publishable | 4.649e-05 | 1.000e+00 | 7.528e-07 | 7.527e-07 | True | True |
| rand_current_03 | publishable | 4.117e-05 | 1.000e+00 | 7.012e-07 | 7.013e-07 | True | True |
| rand_current_04 | publishable | 9.270e-05 | 1.000e+00 | 3.329e-07 | 3.329e-07 | True | True |
| rand_current_05 | publishable | 7.685e-05 | 1.000e+00 | 4.214e-07 | 4.214e-07 | True | True |
| rand_current_06 | publishable | 6.860e-05 | 1.000e+00 | 7.059e-07 | 7.058e-07 | True | True |
| rand_current_07 | publishable | 7.683e-05 | 1.000e+00 | 4.809e-07 | 4.809e-07 | True | True |
| rand_current_08 | publishable | 7.606e-05 | 1.000e+00 | 2.193e-07 | 2.193e-07 | True | True |
| rand_current_09 | publishable | 8.719e-05 | 1.000e+00 | 3.857e-07 | 3.856e-07 | True | True |
| rand_current_10 | publishable | 7.236e-04 | 3.765e-01 | 6.134e-08 | 6.137e-08 | True | True |
| rand_current_11 | publishable | 8.147e-05 | 1.000e+00 | 3.446e-07 | 3.446e-07 | True | True |
| rand_current_12 | publishable | 9.453e-05 | 1.000e+00 | 5.189e-07 | 5.188e-07 | True | True |
| rand_current_13 | publishable | 4.342e-05 | 1.000e+00 | 1.013e-06 | 1.013e-06 | True | True |
| rand_current_14 | publishable | 1.081e-04 | 1.000e+00 | 1.524e-07 | 1.524e-07 | True | True |
| rand_current_15 | publishable | 1.683e-04 | 1.000e+00 | 2.817e-07 | 2.817e-07 | True | True |
| rand_current_16 | publishable | 1.524e-05 | 1.000e+00 | 9.617e-07 | 9.617e-07 | True | True |
| rand_current_17 | publishable | 4.188e-05 | 1.000e+00 | 8.012e-07 | 8.011e-07 | True | True |
| rand_current_18 | publishable | 4.077e-05 | 1.000e+00 | 8.829e-07 | 8.828e-07 | True | True |
| rand_current_19 | publishable | 2.300e-05 | 1.000e+00 | 1.405e-06 | 1.405e-06 | True | True |

## 结论

共有 `20` / `20` 个随机电流方向通过发布级门禁，最大线性化相对误差为 `7.236e-04`，最小方向余弦为 `3.765e-01`。
若全部通过，则说明 P2c96 的 `eps=0.0125` 结论不明显依赖固定方向集合；若部分失败，则失败方向应作为电流空间边界候选。

该结果仍是受信任分支条件结论，不能解除初始化敏感和多平衡分支风险。

## 下一步计划

1. 若全通过，汇总 n2 分支条件 position-only 灵敏度为阶段性候选，并转入效率优化或 n1 归因。
2. 若失败，沿失败方向做扰动尺度二分。
3. 暂不推进完整末端位姿灵敏度。

- 本轮耗时：`530.65 s`
