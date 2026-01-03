# Evolution of Catheter Morphology with Multiple Magnetization Points

多磁化点导管形态演化（多段柔-刚耦合 Cosserat 模型）求解与可视化工程。

本项目最初采用 mag_manip 库进行磁场/梯度计算，配合 NumPy 有限差分近似雅可比；现已演进为：
- 基于离线采样构建的三线性插值磁场-驱动矩阵（JAX 实现，处于自动微分计算图内）。
- 求解器全程使用 JAX 数组与自动微分生成雅可比（可切换前向/反向模式）。

通过该升级，非线性最小二乘（Levenberg–Marquardt, LM）求解的稳定性、收敛速度与最终解质量显著提升，避免了有限差分带来的数值噪声与性能瓶颈。

---

## 1. 项目概览

目标：在多段柔性（Cosserat rod）+ 多段刚性（带磁体）串联的导管结构下，给定外部磁场线圈电流、重力等条件，求解导管在静力学平衡下的三维姿态（位置、姿态、内力、内矩）与可视化。

关键特性：
- 多段柔-刚耦合：每段“柔性段”后接一段“刚性段”（含磁体参数），段与段之间满足连续性边界条件。
- 无量纲化建模：通过参考长度/力/力矩进行统一的无量纲处理，提升数值条件数与可移植性。
- 外载建模：
  - 柔性段分布式载荷（世界坐标系下的线密度力/力矩，例如重力）。
  - 刚性段集中载荷：重力 + 磁场对磁体产生的合力/合力矩，均关于刚段近端点计算。
- 磁场与梯度：采用离线插值的 JAX 驱动矩阵 A(P)（8×8），在线 y8 = A(P) · i（i 为 8 路线圈电流），其中前三行为 B，后五行为约化梯度 G5；由此构造磁力/磁矩，并将整条链路置于 JAX 计算图中，可自动微分。
- 数值求解：JAX 实现的 LM（支持 fwd/rev 模式的雅可比生成），提供多类稳健停止准则与调试诊断。
- 可视化与调试：迭代回调实时绘制当前姿态；块级残差报告、刚段末端平衡审计等工具，便于定位收敛障碍。

---

## 2. 技术路线与模块架构

目录主干：
- `pose_modules_nondim_jax/`
  - `equilibrium_solver_nondim_jax.py`：核心无量纲 JAX 求解器与残差装配、LM 实现、诊断工具。
  - `segments_nondim_jax.py`：柔段 Cosserat 方程 RHS、刚段解析推进（给定总外载）等。
  - `external_wrench_nondim_jax.py`：重力与磁力接口；提供 `MagneticModel` 协议与合力/力矩计算。
  - `mas_nondim_jax.py`：JAX 磁力系统封装（`MagneticActuationSystem`），基于插值驱动矩阵在线计算 B 与 G5，并生成磁力/磁矩（完全可微）。
  - `nondim_jax.py`：无量纲量与 SI 量的互转、RHS 量纲变换等。
  - 其余：网格、工具函数等。
- `supiee_auto_diff/`
  - `actuation_interpolator_jax.py`：JAX 版三线性插值与批量/自动微分封装（`interpolate_A_jax/apply_jax`）。
  - `build_actuation_table.py`：将离线的单位电流 B 栅格数据构造成 A(P) 表（含 G5），并持久化为 `actuation_table.pkl`。
  - `offline_interpolation_data/`：示例离线数据与输出位置（需按实际环境准备）。
- 顶层脚本：
  - `multi_mag_mas_driven_nondim_jax.py`：JAX 全流程演示（构造参数、求解、可视化、诊断）。
  - 历史/对照版本：`pose_modules_nondim_main/`（NumPy 版本，便于比对迁移前后差异）。

数值求解流程（JAX 版）：
1) 构造无量纲求解参数 `SolverParams`（柔/刚段参数、网格、外载、磁体参数、线圈电流等）。
2) 构造初值 `z0_bar`（多段节点状态、每段刚体末端状态等）。
3) 残差 `E(z)` 装配：
   - 柔段：Gauss–Lobatto 6 点离散的区间残差（状态更新一致性、积分一致性、段间边界约束）。
   - 刚段：解析推进给出远端预测，与变量远端状态做差，并施加 |q|=1 与末端自由的边界约束（最后一段）。
4) 通过 JAX 自动微分获得雅可比 J（可选 fwd/rev），进入 LM 迭代，带有多种稳健停止准则与回调机制。
5) 收敛后反解出各段坐标（SI 单位）并可视化。

---

## 3. 从 mag_manip+有限差分 到 JAX 插值+自动微分

历史起点：
- 早期实现直接调用 `mag_manip` 的 `ForwardModelMPEM` 在线计算 B 与梯度；求解器雅可比通过 NumPy 有限差分近似。
- 在复杂、多段、多约束耦合问题中，有限差分噪声与代价均较高，且磁场调用作为黑盒不在自动微分图内，导致优化对磁力项的“感知”不足。

当前方案：
- 采用离线采样/预处理：对工作空间网格和单位电流响应进行采样，构建 `A_table(x,y,z)`（8×8）。前三行为 B，后五行为约化梯度 G5（`[dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]`，满足 Maxwell 约束后足以构造磁力）。
- 在线阶段：`y8 = A(P) @ i`，用 `_wrench_from_BG5(B,G5,m)` 映射到磁力/力矩；该链路完全由 JAX 实现（三线性插值+矩阵乘），天然可微，避免 FD。
- 结果：更稳健的 LM 收敛（步数更少、失败率更低）、更平滑的残差与更一致的数值尺度；整体求解质量显著提高。

注：若需与历史实现进行定量对比，可使用 `supiee_auto_diff/expriments` 下的对比与验证脚本（详见“实验与验证”）。

---

## 4. 目前已实现的功能

- 多段柔-刚耦合（每段：柔段 M 个网格间隔 + 刚段 1 个末端状态）。
- 无量纲化尺度与 SI 互转（位置、力、力矩通道分别缩放）。
- 柔段分布式载荷（世界系下常量线密度力/力矩，默认提供 PDMS 材料示例重力线密度）。
- 刚段集中载荷（重力 + 磁力），磁力来自 JAX 插值的 `MagneticActuationSystem`。
- 非线性 LM 求解（JAX AD 自动雅可比），可选 fwd/rev；稳健停止准则：
  - 相对/绝对代价阈值；
  - 代价改善停滞的耐心机制；
  - 拒绝次数与阻尼饱和守护；
  - 一阶最优性（||J^T E||）与步长阈值（xtol）。
- 可视化与诊断：迭代过程 3D 姿态预览、块级残差排序报告、刚段末端受力恒等式审计、关键变量打印等。

---

## 5. 快速开始

1) 安装依赖（Python ≥ 3.10）
- JAX（CPU 或 GPU 版本，参考官方说明），NumPy，Matplotlib。
- 本仓库不强制第三方 `mag_manip`，因为在线阶段改为 JAX 插值；离线数据准备阶段可选用你已有的工具链。

2) 准备插值表 `actuation_table.pkl`
- 方式 A：直接使用 `supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl`（若已提供）。
- 方式 B：从单位电流 B 栅格数据构建：
  - 将数据（`unit_current_impact.pkl`）放到 `offline_interpolation_data/unit_current_b_data/`。
  - 运行 `python supiee_auto_diff/build_actuation_table.py` 生成 `actuation_table.pkl`。

3) 运行 JAX 演示脚本
- 打开 `multi_mag_mas_driven_nondim_jax.py`，根据你的实验：
  - 设置 `calib_file`（标定 JSON）；
  - 设置 `coil_currents`（8 路电流，单位 A）；
  - 可调整段数、每段长度与网格数 `M_list`；
  - 可选择是否启用重力/磁力。
- 运行：
  
  ```bash
  python multi_mag_mas_driven_nondim_jax.py
  ```
  
- 终端将打印 JAX 设备信息、收敛状态与载荷检查；若开启绘图，会显示 3D 姿态。

注意：插值查询不做外推，需确保位置落在表格范围内（`apply_checked` 会进行边界检查）。

---

## 6. 关键 API 速览（JAX 版）

- `pose_modules_nondim_jax.equilibrium_solver_nondim_jax.SolverParams`
  - 汇集柔/刚段、网格、外载、磁体、线圈电流、残差布局等。
- `pose_modules_nondim_jax.equilibrium_solver_nondim_jax.MultiSegmentEquilibriumSolverNondimJAX`
  - `solve_lm(z0_bar, max_iter, tol, lm_damping, jac_method, callback, ...)`：LM 求解主入口；`jac_method` 可选 `"fwd"` 或 `"rev"`。
  - `residual_block_report/print_top_blocks`：诊断残差主导块。
- `pose_modules_nondim_jax.mas_nondim_jax.MagneticActuationSystem`
  - `y8(position, currents)` → `[B(3), G5(5)]`；
  - `magnetic_wrench(pose_list, magnetic_moment, currents)` → 每点 6×1 扳手（F,T）。
- `supiee_auto_diff.actuation_interpolator_jax`
  - `interpolate_A_jax`, `apply_jax`, `apply_vmap`：三线性插值核心，完全可微。

---

## 7. 实验与验证脚本

- `supiee_auto_diff/expriments/compare_actuation_npz.py`
  - 对比 mag_manip 生成的驱动 npz 与 JAX 插值生成的结果差异。
- `supiee_auto_diff/expriments/validate_actuation_interpolator_accuarcy.py`
  - 在一组采样点上对比 `A(P)`、`B`、`G5` 的误差，验证插值精度与一致性。
- `supiee_auto_diff/expriments/plot_plane_fields_mag_vs_jax.py`
  - 可视化平面上的场/梯度差异。

---

## 8. 单位与无量纲说明

- 参考量：
  - 长度 `L_ref`（典型取导管总长或代表性长度）。
  - 力 `F_ref`（典型取柔段轴向刚度 `E*A` 的量级）。
  - 力矩 `M_ref = F_ref * L_ref`。
- 状态量缩放：`p(0:3)` 用 `L_ref`，`f(7:10)` 用 `F_ref`，`tau(10:13)` 用 `M_ref`。RHS 与残差装配遵循相同规则。

---

## 9. 局限与待办

- 插值表范围外不支持外推；实际应用中应覆盖操作工作空间。
- 当前示例的 G5 为 5 个派生梯度分量，满足 Maxwell 约束并足以构造磁力；如需更一般的场模型或更多物理效应，可在离线阶段扩展表结构与在线映射。
- 进一步完善性能（如更优的线性解器/预条件）与更丰富的边界条件/约束类型。
- 增加系统化基准与自动化测试。

---

## 10. 参考与致谢

- JAX 与 jax.numpy
- Cosserat rod 文献与数值积分/配点方法
- 历史实现使用的 mag_manip 思想与数据来源

---

## 11. 变更记录（摘要）

- 初版：mag_manip 在线磁场，NumPy 有限差分雅可比。
- 现版：JAX 三线性插值 A(P) + 自动微分雅可比；LM 稳健性与收敛质量显著提升。

提示：如需将更详细的 git 提交历史自动汇总进本节，请允许我读取仓库提交记录后补充时间线与里程碑。
