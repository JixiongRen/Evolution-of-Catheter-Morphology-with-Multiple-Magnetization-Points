# 20260523204022_fk_under_60s 实验记录

## 实验目的

在无人值守窗口内，围绕固定 FK case 大幅压缩单组求解耗时。当前已知新 FK 同参同 `tol=1e-5` 约 `353.07 s / 2555 iter`，本轮目标为在保持几何结果可解释、可分级的前提下，尽量压缩到 `60 s` 以内。

## 固定 Case

原始配置中的 `/abs/path/calibration.json` 在本机不存在，因此本轮按要求替换为仓库内标定文件：

```text
calib/mpem_calibration_file_sp=40_order=1.yaml
```

同时使用仓库内 actuation table：

```text
supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl
```

固定参数：

```text
flex_lengths=0.03,0.03
rigid_lengths=0.003,0.003,0.003
L_protrude=0.11
M_list=5,5,5
enable_gravity=true
enable_magnetics=true
coil_currents=0,0,0,0,0,50,0,0
m_body_list=0,0,-0.005301; 0,0,0.005301; 0.005301,0,0
p0=0,0,0
Q0=0.70710678,0,0.70710678,0
axis_body=0,0,1
```

## 实验方案

1. 建立生产路径基线：cached engine、无 callback、无 plot、无每轮保存。
2. 对比 scale 配置：保留当前 optimized scale，同时增加 legacy scale 兼容路径。
3. 测试 warm-up 后 steady solve。
4. 测试 current continuation：从低电流逐段加载到目标电流，最终精修。
5. 测试 coarse-to-fine：粗网格求 basin，再 lift 到 `M_list=5,5,5` 精修。
6. 测试 LM globalization 低风险变体：阻尼初值、step clipping 或线搜索候选。
7. 所有候选均保存 summary、centerline、日志、耗时和几何对比。

## 评价分级

- `speed-only result`：耗时满足或接近目标，但几何偏移较大。
- `physically comparable result`：相对选定 baseline 的 centerline 最大误差、RMS 和末端误差在可解释范围内。
- `production-candidate result`：同时满足 `<=60 s`、几何可接受、残差和实现路径可长期复用。

## 执行记录

- 2026-05-23 20:40：创建实验目录。
- 2026-05-23 20:45：新增固定 case 生产 benchmark driver，输出 `summary.json`、`stages.csv`、`centerline.npy`、`solution.npz`、`centerline_overlay.png` 和 `/usr/bin/time -v` 统计。
- 2026-05-23 20:55：确认 no-callback cached engine 的首次 JAX 编译约 `172 s`，warm-up 后 10 个 LM step 约 `0.85 s`。
- 2026-05-23 21:05：完成多段 current continuation 探针，steady solve 首次达到 `58.75 s`，但只能归类为 speed-only。
- 2026-05-23 21:25：完成 damping、legacy scale、coarse-to-fine、`diagH` damping/backtracking 和 `jac_method=rev` 探针。
- 2026-05-23 21:55：确定短 continuation 加目标电流长精修是本轮最优方向。
- 2026-05-23 22:05：得到最终候选 `fk_under_60s_final_candidate_tol15e4`。

## 运行命令

完整命令保存在：

```text
experiment_records_fk_optimize/20260523204022_fk_under_60s/commands.sh
```

## 实验结果

汇总表保存在：

```text
experiment_records_fk_optimize/20260523204022_fk_under_60s/summary.csv
experiment_records_fk_optimize/20260523204022_fk_under_60s/summary.json
```

关键结果如下：

| 候选 | 分类 | steady wall time | ok | final normE | 相对旧 FK max error | 相对旧 FK RMS | 相对新 FK max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fk_under_60s_final_candidate_tol15e4` | production-candidate | `48.21 s` | true | `1.50e-4` | `8.18 mm` | `6.45 mm` | `29.43 mm` |
| `fk_short_continuation_final1150` | physically comparable | `49.23 s` | false | `1.36e-4` | `8.13 mm` | `6.62 mm` | `30.08 mm` |
| `fk_short_continuation_final1250` | physically comparable | `52.93 s` | false | `1.00e-4` | `9.38 mm` | `7.68 mm` | `31.60 mm` |
| `fk_current_continuation_damp_03_final1000` | speed-only | `59.38 s` | false | `1.28e-4` | `22.70 mm` | `12.77 mm` | `20.48 mm` |
| `fk_current_continuation_probe_a` | speed-only | `58.75 s` | false | `1.77e-4` | `17.57 mm` | `10.50 mm` | `27.50 mm` |

最终候选末端点：

```text
[0.0211692930, -0.0652211888, 0.0487450873] m
```

旧 FK baseline 末端点：

```text
[0.0253055412, -0.0643168345, 0.0462209269] m
```

最终候选的阶段统计：

| 阶段 | 迭代 | 时间 | stop | normE |
|---|---:|---:|---|---:|
| zero-current continuation | 8 | `0.61 s` | converged | `7.03e-4` |
| target-current coarse gate | 60 | `2.40 s` | max_iter | `1.14e-2` |
| target-current final refine | 1103 | `45.19 s` | converged | `1.50e-4` |

## 数据分析

1. 首次 JAX 编译仍然很贵。固定 fine mesh 的 warm-up 中，residual/Jacobian/one-step 编译合计约 `150-180 s`。因此本轮 `<=60 s` 结论只适用于 persistent/cached engine 的 steady solve，不包含新进程 cold compile。
2. no-callback/no-plot/no-save 的生产路径显著降低单步耗时。warm-up 后 fine mesh LM step 约 `0.04-0.08 s/iter`，相比原 CLI 路径 `353 s / 2555 iter` 的综合耗时有明显改善。
3. 多段 continuation 可把 wall time 压到 `~50-60 s`，但几何不一定更接近旧 FK。最有效方案不是细分电流，而是 `0 -> target` 的极短 continuation，把预算留给目标电流 final refine。
4. `legacy` scale 在本轮候选中更快，但几何明显偏离两个已有 baseline，因此只记录为 speed-only，不作为可信候选。
5. coarse-to-fine 进入错误 basin，`diagH + backtracking` 使每步变慢，`jac_method=rev` 与 `fwd` 基本等价，均不作为下一步默认路径。
6. 最终候选相对旧 FK baseline 满足 10% L_ref 几何门禁：`max point error = 8.18 mm < 11 mm`。但它相对当前 optimized CLI baseline 偏差约 `29.43 mm`，说明旧/新 FK baseline 本身存在 scale 或 basin 差异，后续报告必须明确选择的物理对照对象。

## 结论

本轮达到了目标意义上的 steady solve `<=60 s`：最终候选 `fk_under_60s_final_candidate_tol15e4` 用 persistent cached engine、no callback/no plot、显式 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、current continuation 和宽松 residual gate，完成固定 case 求解：

```text
steady solve wall time = 48.21 s
first-call compile time = 151.79 s
iteration count = 1103 final refine iterations
final normE = 1.50e-4
max_abs_E = 6.45e-5
old-baseline max pointwise centerline error = 8.18 mm
```

该结果可作为 `production-candidate result` 的前提是：生产环境复用 persistent engine，正确性门禁以旧 FK baseline 的 centerline 几何为主，并接受 `tol=1.5e-4` 的宽松 residual gate。若必须严格 `tol=1e-5` 或必须贴近当前 optimized CLI baseline，则本轮尚未达成 production-candidate，只能视为 speed/geometry trade-off 候选。

## 下一步计划

1. 把 persistent engine 的 warm-up 生命周期纳入 FK/IK 上层调用，避免每次新进程重新支付 `150-180 s` 编译成本。
2. 明确 FK 正确性 baseline：建议优先审查 optimized CLI 与 engine 的 scale/basin 差异，决定后续以旧 FK、optimized CLI 还是实验观测作为物理基准。
3. 如果必须严格 `tol=1e-5` 且 `<=60 s`，下一轮不应继续微调 continuation；应改造 LM 内核或残差结构，重点降低 final refine 的后期慢收敛。
4. 将短 continuation 参数接入 IK FK replay 路径，测试 IK 内多次 FK 调用时的真实端到端收益。
