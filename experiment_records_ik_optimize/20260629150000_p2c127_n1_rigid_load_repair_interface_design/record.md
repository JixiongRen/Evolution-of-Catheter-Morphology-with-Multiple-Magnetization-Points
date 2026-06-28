# P2c127 n1 显式刚性磁体段载荷传播修复接口设计

## 实验目的

P2c126 已判定单点磁控当前模型不可发布，主失败为刚性磁体段载荷传播失配与模型耦合缺失。本轮不直接修改求解器，而是定位代码接口并设计最小修复路线，保证下一轮实现只改变物理方程，不继续依赖权重调参或自由比例变量。

## 代码定位

- FK 刚段传播参考：`pose_modules_nondim/equilibrium_solver_nondim.py` / `EquilibriumSolverNondim.residual`。当前作用：对每个刚段计算自动外载，并调用 rigid.propagate 得到刚段远端状态。 修复用途：作为 IK 显式刚段载荷传播方程的物理参考。
- IK 位置与连续残差：`experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/run_p2c57_force_continuity_weight_repair.py` / `_make_split_functions`。当前作用：残差由末端位置、边界、interface_wrench_residual 和正则组成。 修复用途：新增刚段载荷传播残差，不再只靠 force_scale 或 beta 比例吸收载荷链缺口。
- 硬力连续反事实：`experiment_records_ik_optimize/20260629030000_p2c110_n1_hard_force_scaled_continuity/run_p2c110_n1_hard_force_scaled_continuity.py` / `FORCE_SCALES`。当前作用：扫描 1、4、16 倍力连续权重。 修复用途：作为修复模型的负对照，证明调权重不能替代物理传播。
- beta 载荷传递反事实：`experiment_records_ik_optimize/20260629033000_p2c111_n1_beta_load_transfer/summary.json` / `beta_reg_scales`。当前作用：用自由比例参数近似载荷传递。 修复用途：作为自由松弛变量不可发布的负对照。

## 方程设计

当前实验层残差可以写为：

\[
r = [r_p, r_b, w_F r_c^F, w_M r_c^M, r_{reg}],
\]

其中 \(w_F\) 和 \(w_M\) 是权重。历史实验说明，仅改变权重会把失配转移到一阶最优性、条件数或接口残差。修复版应显式引入刚性磁体段载荷传播：

- 刚段平衡传播：r_x = x_rigid_end - Phi_rigid(x_flex_end, F_m, tau_m)。含义：刚段远端状态必须由柔性段末端状态和磁体段合外载传播得到。 验收：几何残差进入基线残差；不允许只由正则项吸收。
- 力连续：r_F = n_flex_distal + F_m。含义：柔性段远端内力应平衡刚性磁体段对接口的等效外力。 验收：||r_F|| / max(||F_m||, floor) <= 1.5。
- 矩连续：r_M = m_flex_distal + tau_m + r_cm x F_m。含义：柔性段远端内矩应平衡外力矩和力臂引起的附加力矩。 验收：与力连续同时满足；不能只满足矩而破坏力。
- 任务残差：r_p = p_tip - p_target。含义：末端位置拟合仍保留，但不能覆盖物理闭合失败。 验收：任务维度不得超过有效秩；基线物理门禁不过时不扫描扰动。

## 实施阶段

- P2c128：实现只读载荷传播观测器。工作：在不改求解残差的情况下输出 F_m、tau_m、r_F、r_M、力臂项和现有 interface_wrench_residual 的差异。 验收：复现实验 P2c99/P2c120 的主失败，并定位力连续与矩连续各自贡献。
- P2c129：显式传播残差最小原型。工作：在实验级 _make_split_functions 分支中加入 r_F/r_M/r_x，不替换生产 IK。 验收：n1 基线同时接近 stationarity <=1e-5、condition <1e7、force ratio <=1.5。
- P2c130：任务降维对照。工作：若三维位置仍失败，测试一维或二维有效子空间任务，不跑完整位姿。 验收：有效秩与任务维度一致，且扰动重求解误差 <0.05。
- P2c131：负样本复核。工作：若连续三轮仍同类失败，固定当前 n1 为不可发布边界，不再扫参。 验收：输出模型修复失败证明和工作空间建议。

## 不变量

- 不在基线门禁前做扰动灵敏度扫描。
- 不用自由比例变量替代真实力链传播。
- 不扩大到完整末端位姿任务。
- 不替换生产 IK；先在实验级 runner 中验证。
- 每个候选必须同时记录一阶最优性、条件数、载荷传播、接口残差、中心线分支漂移和局部线性化误差。

## 当前结论

本轮给出的是模型修复接口设计，不声明 n1 已可发布。下一轮应先实现只读载荷传播观测器，验证刚段外载、力臂矩和接口内力链之间的数值差异；只有该观测器复现 P2c126 失败机制后，才进入显式传播残差最小原型。

## 下一步计划

执行 P2c128：实现只读载荷传播观测器。输出 \(F_m\)、\(	au_m\)、\(r_F\)、\(r_M\)、力臂项、现有接口残差分块，并与 P2c120 的不可发布结论对齐。

## 运行信息

- 状态：n1_rigid_load_repair_interface_designed。
- 墙钟耗时：0.0000 s。
