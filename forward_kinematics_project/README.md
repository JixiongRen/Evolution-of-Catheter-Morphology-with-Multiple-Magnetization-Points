# FK 打包项目说明

该目录是从主工程中整理出的 FK 相关代码包，用于后续归档、迁移和独立验证。整理原则是保留当前生产路径行为，不改变收敛阈值、离散化网格、物理参数或求解器逻辑。

## 目录结构

- `forward_kinematics_optimized/`：当前生产使用的新 FK 求解器，包含 `ForwardKinematicsEngine`、无量纲求解器、磁力模型适配和 CLI。
- `forward_kinematics_legacy/`：旧 FK CLI 和旧版无量纲实现，用于历史对照与回归检查。
- `experiments/efficiency_optimization/fk_under_60s_driver.py`：当前推荐的生产 benchmark/运行驱动，支持 `dense_fused_cholesky`。
- `supiee_auto_diff/`：FK 磁场插值所需的最小依赖，包括 `actuation_interpolator_jax.py` 和 `actuation_table.pkl`。
- `calib/`：磁场标定文件。
- `scripts/`：可直接运行的入口脚本。
- `tests/`：轻量导入与资源检查脚本。

## 推荐环境

建议使用主工程已验证的 conda 环境：

```bash
conda activate evolution_of_mcrs
```

运行 GPU 实验时建议显式设置：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## 生产路径入口

运行当前生产候选 FK 固定 case：

```bash
bash scripts/run_production_fixed_case.sh
```

该脚本默认使用：

- `M_list=5,5,5`
- `tol=1e-5`
- `linear_solver=dense_fused_cholesky`
- `dense_solve_method=cholesky`
- `scale_mode=optimized`
- `scale_include_external=true`

输出会写入 `outputs/production_fixed_case/`。

## 旧 FK 对照入口

运行旧 FK CLI 对照 case：

```bash
bash scripts/run_legacy_fixed_case.sh
```

输出会写入 `outputs/legacy_fixed_case/`。

## 轻量检查

执行导入和资源检查：

```bash
python tests/smoke_imports.py
```

该检查不会启动完整 FK 求解，只验证核心模块、校准文件和磁场插值表是否可访问。

## 注意事项

- `wall_s` 通常指 JAX 编译完成后的净求解时间。
- `first_call_compile_s` 是 JAX/XLA 首次编译和 warm-up 时间。
- 批量扫参时如果每组都重新构建 engine，会重复产生较大的 warm-up 成本。
- `forward_kinematics_optimized/` 中的注释和 docstring 维持英文；本说明文档使用中文。
