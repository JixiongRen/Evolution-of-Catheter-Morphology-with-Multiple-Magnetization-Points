# eff_opt_exp_20260523182339_fk_pointwise_regression_check 实验记录

## 实验目的

排查当前可视化中 IK 解精度过低的问题是否由 FK 修改前后姿态漂移导致。重点验证同一配置下，修改前后 dense FK 收敛后导管完整 centerline 上每个点的位置是否近似一致。

## 实验方案

- 使用 `2c7ab87` 作为修改前 dense FK 对照版本。
- 使用当前 `dev` HEAD 作为修改后版本。
- 在两版代码上运行相同 dense FK case，并保存完整 centerline。
- 对每个 centerline 点计算绝对误差、相对全局位置误差和相对 `L_ref` 误差。
- 目标点在 FK 对照中由同一 case 配置定义，因此 target drift 应为零。
- 判据采用用户指定的 10% 标准：修改前后每个点的位置相对误差不超过 10%，并同时报告相对 `L_ref` 的误差。

## 实验过程记录

- 2026-05-23 18:23：创建 FK 逐点回归排查实验。
- 2026-05-23 18:24：新增 `fk_pointwise_regression_check.py`，用于在指定仓库版本下运行 FK 并比较完整 centerline。
- 2026-05-23 18:25：在 `2c7ab87` 临时 worktree 上运行修改前 dense FK。
- 2026-05-23 18:31：在当前 `dev` HEAD 上运行修改后 dense FK。
- 2026-05-23 18:35：生成逐点误差表。
- 2026-05-23 18:37：生成修改前后 FK centerline overlay 图。

## 运行命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib conda run -n evolution_of_mcrs python experiments/efficiency_optimization/fk_pointwise_regression_check.py run \
  --repo_root docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/before_2c7ab87_worktree \
  --out_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/before_2c7ab87 \
  --max_iter 80 \
  --tol 1e-5 \
  --case_ids straight_nomag_nogravity,gravity_only_mid,mag_zero_mid,coil5_pos,mixed_balanced
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib conda run -n evolution_of_mcrs python experiments/efficiency_optimization/fk_pointwise_regression_check.py run \
  --repo_root . \
  --out_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/current_head \
  --max_iter 80 \
  --tol 1e-5 \
  --case_ids straight_nomag_nogravity,gravity_only_mid,mag_zero_mid,coil5_pos,mixed_balanced
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiments/efficiency_optimization/fk_pointwise_regression_check.py compare \
  --before_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/before_2c7ab87 \
  --after_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/current_head \
  --out_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/compare \
  --threshold 0.10
```

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n evolution_of_mcrs python experiments/efficiency_optimization/fk_pointwise_regression_check.py plot \
  --before_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/before_2c7ab87 \
  --after_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/current_head \
  --compare_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/compare \
  --out_dir docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/plots
```

## 实验结果

对比版本：

- 修改前：`2c7ab878ca1b901a7f75ac20247ee978e9e268b5`
- 修改后：`2fe0a1da77c01c88f2ea16b0948821f6f1303299`
- 后端：GPU，JAX x64，`XLA_PYTHON_CLIENT_PREALLOCATE=false`
- FK inner：`max_iter=80`，`tol=1e-5`

运行时间：

- 修改前 5 个 case 合计 FK wall time：`282.28 s`
- 修改后 5 个 case 合计 FK wall time：`283.01 s`

逐点 centerline 对比结果：

| case | 点数 | 最大点误差 | 最大相对全局位置误差 | 最大相对 L_ref 误差 | 末端误差 | 10% 判据 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `straight_nomag_nogravity` | 19 | `2.13e-18 mm` | `4.26e-20` | `1.18e-20` | `0 mm` | 通过 |
| `gravity_only_mid` | 19 | `1.12e-7 mm` | `1.00e-9` | `6.20e-10` | `1.12e-7 mm` | 通过 |
| `mag_zero_mid` | 19 | `2.13e-18 mm` | `4.26e-20` | `1.18e-20` | `0 mm` | 通过 |
| `coil5_pos` | 19 | `8.26e-12 mm` | `7.77e-14` | `4.59e-14` | `8.26e-12 mm` | 通过 |
| `mixed_balanced` | 19 | `3.87e-11 mm` | `3.52e-13` | `2.15e-13` | `3.87e-11 mm` | 通过 |

结构化结果：

- `docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/compare/fk_pointwise_compare.csv`
- `docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/compare/fk_pointwise_compare.json`

可视化结果：

- `docs/efficiency_optimization_experiment/eff_opt_exp_20260523182339_fk_pointwise_regression_check_logs_data/plots/fk_before_after_pointwise_overlay.png`

## 数据分析

本轮 5 个代表性 dense FK case 在修改前后完整 centerline 逐点结果几乎完全重合，最大误差远低于 10% 阈值。即使使用更严格的相对全局位置误差，最大值也只有 `1.00e-9`，主要来自 `gravity_only_mid` 的数值级差异。

因此，当前 benchmark 图片中目标点偏差很大的现象不能归因于 FK 修改导致的姿态漂移。该现象应回到 IK 流程本身排查。

结合 `eff_opt_exp_20260523173326_ik_full_max_iter_benchmark` 的 history：

- forced full-cap run 的 `outer_len=241`。
- `accept_count=1`，仅初始化状态被记录为 accepted。
- `trial_accept_count=0`，后续 240 次 trial 没有任何一步被接受。
- 全部记录的 `cls` 都是 `REJECT`。
- 初始化误差为 `99.55 mm`，`normE=0.336`，且 `cls=REJECT`。
- 最后一轮 trial 误差为 `103.98 mm`，`rho=-1.5867e14`，`lam=1e12`，仍为 `REJECT`。

这说明图片中的绿色曲线不是成功收敛后的 IK 解，而是最后 accepted state；在该 forced run 中它实质上仍是初始化状态。

## 实验结论

FK 修改前后在本轮 5 个代表性 case 上通过逐点 10% 一致性检查。当前图片展示的低精度不是 FK 回归漂移导致，而是 IK 初始化和 globalization 失败：候选 trial 全部被拒绝，最终结果没有真正更新到更优解。

## 下一步计划

- 不应把本轮 full max-iter benchmark 作为 IK 求解质量结论，只能作为耗时上界参考。
- 下一步若继续修正 IK 精度，应优先排查 attempt 初值、目标可达性、trial 接受准则、LM 阻尼更新和 reject 状态下的 fallback，而不是回退 FK 优化。
- 后续绘图脚本需要显式标记 `accepted_count`、`success`、`cls` 和 final accepted 是否只是初始化，避免误读为成功收敛解。
