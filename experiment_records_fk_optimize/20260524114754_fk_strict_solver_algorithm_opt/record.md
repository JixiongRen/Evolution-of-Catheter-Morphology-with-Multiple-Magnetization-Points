# 20260524114754_fk_strict_solver_algorithm_opt 实验记录

## 实验目的

在固定 FK case、严格 `tol=1e-5`、物理配置不变的条件下，从 solver algorithm（求解算法）层面压缩 steady solve（稳定热路径求解）耗时。上一轮依赖 `tol=1.5e-4` 的结果不计入严格达标。

## 固定 Case

`/abs/path/calibration.json` 在本机若不存在，使用仓库内：

```text
calib/mpem_calibration_file_sp=40_order=1.yaml
```

固定配置：

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
tol=1e-5
```

## 实验方案

1. 复核严格 dense LM（稠密阻尼最小二乘）生产路径基线。
2. 拆分 profiling（性能剖析）：residual、Jacobian、`J.T @ J`、dense solve、trial residual、lambda update。
3. 审查 residual/Jacobian block structure（残差/雅可比块结构）。
4. 实现显式开关的新结构化求解路径，并保留 dense LM fallback（回退路径）。
5. 测试 sparse/block-structured Newton（稀疏/块结构牛顿）、Schur/block elimination（舒尔补/块消元）、preconditioned Newton-Krylov（带预条件的牛顿-克雷洛夫）和 trust-region LM（信赖域阻尼最小二乘）候选。

## 判定规则

- `tol=1e-5` 未满足，不标记 production-candidate（生产候选）。
- 几何误差超过 `10% L_ref`，不标记 strict-comparable（严格几何可比）。
- 新算法改变 convergence basin（收敛盆）时，只保留为实验路径。

## 执行记录

- 2026-05-24 11:47：创建实验目录。
- 2026-05-24 12:10：接入 trust-region LM（信赖域阻尼最小二乘）显式开关，参数包括 `trust_region_eta`、`max_damping_attempts`、`reject_lam_gain`。
- 2026-05-24 12:22：只读复核指出 trust-region 内层重复构造 `J.T @ J`，已改为每个外迭代只构造一次 dense normal system（稠密正规方程系统），内层仅用不同阻尼重新求解。
- 2026-05-24 12:45：接入 `dense_solve_method`，新增 `pos`（正定求解器）和 `cholesky`（乔列斯基分解）路径，默认仍保留原 `solve`（通用线性求解）fallback。
- 2026-05-24 13:10：完成参数扫描，发现 `lm_damping=0.1 + dense_solve_method=pos + 短 continuation` 可以严格收敛并进入 60s 内。
- 2026-05-24 13:35：补跑当前 driver 严格 baseline（基线），用于同一 centerline convention（中心线约定）下的几何比较。
- 2026-05-24 13:50：重复最终候选，作为 `production-candidate result（生产候选结果）` 记录。

## 运行命令

见同目录 `commands.sh`。

## 实验结果

关键结果如下：

| Case | 分类 | Strict ok | steady solve | final refine | iter | final normE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fk_driver_lm03_solve_strict_baseline` | strict baseline | True | 60.384s | 59.027s | 1456 | 9.882e-6 |
| `fk_pos_lm01_shortstage_final1450` | speed-only result | True | 58.745s | 57.284s | 1373 | 9.855e-6 |
| `fk_pos_lm01_shortstage_final1450_production` | production-candidate result | True | 55.586s | 54.248s | 1373 | 9.855e-6 |
| `fk_pos_lm01_final1450` | speed-only result | True | 61.087s | 58.059s | 1424 | 9.926e-6 |
| `fk_cholesky_no_trust_final1400` | speed-only result | False | 61.038s | 57.982s | 1400 | 4.864e-5 |
| `fk_trust_region_eta01_final1400` | speed-only result | False | 72.114s | 68.147s | 1400 | 2.419e-5 |

最终候选：

```text
case: fk_pos_lm01_shortstage_final1450_production
steady solve: 55.586s
first-call compile: 157.977s
wall including warmup: 213.563s
final refine: 54.248s
iteration count: 1373
final normE: 9.855144658932908e-6
max_abs_E: 3.2116018483348218e-6
centerline end: [0.012816936039722612, -0.07526103559213498, 0.041115689978247834]
```

几何比较使用当前 driver 严格 baseline，而不是旧 callback 输出。最终候选相对 `fk_driver_lm03_solve_strict_baseline`：

```text
max pointwise centerline error: 0.0001154054 m
RMS error: 0.0001013074 m
end error: 0.0001116775 m
max_rel_to_L_ref: 0.001049
```

该误差约为 `0.10% L_ref`，低于 `10% L_ref` 门限。

## 数据分析

1. `trust-region LM（信赖域阻尼最小二乘）` 可以在相同迭代数下降低残差，但会增加 trial residual（试探残差）和模型评估开销；本轮没有成为最快 strict 路径。
2. `cholesky（乔列斯基分解）` 比原 `solve（通用线性求解）` 有小幅单步收益，但单独不足以 strict 达标。
3. `pos（正定求解器）` 在该 dense normal equation（稠密正规方程）上效果最好，保持严格收敛同时降低迭代数和单步开销。
4. 将 continuation（延续）阶段从 60 次缩短到 20 次没有破坏最终 strict 收敛，反而减少总耗时；最终收敛由 final refine 保证。
5. 旧 callback centerline（回调中心线）与当前 driver 输出存在明显差异，可能混合了中心线采样约定或收敛路径差异。本轮严格几何判定采用同一 driver、同一 extraction convention（提取约定）的严格 baseline。

## 结论

本轮达到目标：固定 FK case 在物理配置不变、`tol=1e-5` 不放宽的条件下，获得 `55.586s` 的 production-candidate result（生产候选结果）。相对当前 driver 严格 baseline，几何误差为 `0.10% L_ref`，满足 strict-comparable（严格几何可比）要求。

本轮代码保留原 dense LM（稠密阻尼最小二乘）路径作为 fallback（回退路径），新增的 `dense_solve_method=pos` 是显式开关，不会强制改变旧调用方默认行为。

## 下一步计划

1. 将 `dense_solve_method=pos` 纳入后续 FK/IK production benchmark（生产基准），验证更多极端 case。
2. 继续评估 sparse/block-structured Newton（稀疏/块结构牛顿），目标是把 final refine 进一步压到 30s 级别。
3. 若多 case 稳定，再考虑 warm-start snapshot（热启动快照）落盘，作为工程层面的二次加速。
