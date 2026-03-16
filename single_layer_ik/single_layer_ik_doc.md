 # 单层联合逆运动学（Single-layer Joint IK）方案文档
 
 ## 1. 背景与目标
 
 当前仓库的求解结构是典型的“双层”(bi-level) 结构：
 
 - 外层 IK：在控制变量（电流 `I` 与插入深度 `x`）上做 LM/梯度类优化，使末端位置接近期望 `p_des`。
 - 内层 FK：对每个外层给定的 `(I, x)`，调用 `ForwardKinematicsEngine.solve_with_stats(...)`，其内部通过 LM 求解平衡形状变量 `z`，使平衡残差 `E(z, I, x) ≈ 0`。
 
 你希望实现“真正意义”的单层联合求解：在同一个 LM/非线性最小二乘迭代中，同时更新
 
 - 平衡形状未知量 `z`（FK 内层的 decision vector）
 - 逆问题控制未知量 `(I, x)`（或其无约束参数化 `u`）
 
 从而不再出现“外层每迭代一次就完整求一次内层平衡收敛”的硬嵌套。
 
 ---
 
 ## 2. 现有代码结构梳理（必须理解的现状）
 
 ### 2.1 FK 平衡方程与残差
 
 - 平衡残差函数：
   - 文件：`forward_kinematics_optimized/equilibrium_solver_nondim.py`
   - 函数：`residual_bar(z_bar, params) -> E`
 
 该函数将多段柔性 Cosserat rod + 刚段 lumped model 的离散残差按固定顺序堆叠为向量 `E`。
 
 - `z_bar`：nondim（bar）单位下的 packed decision vector。
 - `params`：`SolverParams`，包含材料参数、外载（重力/磁力模型）、离散网格、尺度 `scales`、以及 `coil_currents`、`L1_dim` 等。
 
 关键点：插入深度 `x` 并不是直接进入 `residual_bar`，而是通过 advancer 改变第一段柔性长度 `L1_dim`，再由 `with_L1_dim(params, L1_dim)` 注入到 `params` 中。
 
 ### 2.2 FK 求解器（LM）
 
 - 文件：`forward_kinematics_optimized/equilibrium_solver_nondim.py`
 - 类：`MultiSegmentEquilibriumSolverNondimJAXCached`
 - 方法：`solve_lm(z0_bar, params, ...)`
 
 其目标等价于最小化
 
 $$
 \min_{z}\ \frac{1}{2}\|E(z;\,\text{params})\|^2
 $$
 
 并通过 LM 更新 `z`。
 
 返回的 `LMStats`（`ok_strict, normE, lam, J` 等）目前被 IK 用于敏感度计算。
 
 ### 2.3 FK 引擎封装（给 IK 调用的接口）
 
 - 文件：`forward_kinematics_optimized/fk_engine.py`
 - 类：`ForwardKinematicsEngine`
 - 方法：
   - `solve_with_stats(coil_currents, L_protrude, ...) -> (z_star_bar, params, meshes, ok, lm_stats)`
   - `query_sites(z_bar, scales) -> dict`（含 `tip_p_dim`）
 
 关键流程：
 
 - 通过 `apply_advancer_protrude_length` 使插入/回撤只改变第一段柔性长度 `L1_dim`。
 - 用 `with_coil_currents(params_base, I)` 更新 `params.coil_currents`。
 - 用 `with_L1_dim(params, L1_dim)` 更新 `params.L1_dim`。
 - 然后调用 `solve_lm` 对 `z` 求平衡。
 
 ### 2.4 现有 IK（外层 LM）为何会“每轮 3 次 FK”
 
 - 文件：`inverse_kinematics/ik_multisite_lm_opt.py`
 - 在每轮外层迭代中：
   - `dp/dI`：通过 `compute_dp_dI_via_lm_adjoint(...)`（利用内层 `dE/dz` 与 `dE/dI`）
   - `dp/dx`：用有限差分 `x±h`，需要 2 次 FK
   - `trial` 评价：对候选 `(I_trial, x_trial)` 再做 1 次 FK
 
 这也是单层方案想要从结构上消除的冗余来源。
 
 ---
 
 ## 3. 单层联合求解：数学建模（推荐可落地版本）
 
 单层联合求解可用 KKT/SQP、增广拉格朗日、或纯 penalty 等形式。结合现有代码最容易复用、实现风险最低、且与 JAX 自动微分相容，第一版建议采用：
 
 - “加权残差的统一最小二乘（penalty）”形式
 
 直观理解：把“平衡方程”和“末端目标”一起写成一个大残差向量，LM 在联合变量上一步到位更新。
 
 ### 3.1 联合未知量定义
 
 - `z_bar ∈ R^{nZ}`：与 FK 相同的 nondim packed equilibrium state。
 - `u ∈ R^{nU}`：控制变量的无约束参数（推荐 `nU=9`）。
 
 控制量的约束参数化（建议复用 `inverse_kinematics/ik_multisite_lm_opt.py` 的映射）：
 
 - 电流限幅（逐 coil）：
 
 $$
 I_i(u) = I_{\max}\tanh(u_{I,i}),\quad i=1..8
 $$
 
 - 插入深度区间约束：
 
 $$
 x(u)=x_{\min}+(x_{\max}-x_{\min})\,\sigma(u_x),\quad \sigma(t)=\frac{1}{1+e^{-t}}
 $$
 
 - 第一段柔性长度（advancer 模型）：
 
 $$
 L_1(x)=x - L_{\text{fixed}},\quad L_{\text{fixed}}=\sum_{k\ge 2} L^{\text{flex}}_k + \sum_{k} L^{\text{rigid}}_k
 $$
 
 最终联合变量：
 
 $$
 y=\begin{bmatrix}z_{\text{bar}}\\u\end{bmatrix}
 $$
 
 ### 3.2 联合残差向量构造
 
 定义统一残差向量 `R(y)`，求解
 
 $$
 \min_y\ \frac{1}{2}\|R(y)\|^2
 $$
 
 推荐堆叠如下（其中 1) 与 2) 为核心，3) 及以后为稳定性与可解性增强项）：
 
 1) 平衡残差（强约束倾向）
 
 $$
 r_E(y)=w_E\,E\big(z_{\text{bar}},\ \text{params}(I(u), L_1(x(u)))\big)
 $$
 
 其中 `E` 对应 `forward_kinematics_optimized.equilibrium_solver_nondim.residual_bar(z_bar, params)`。
 
 2) 末端位置残差（IK 目标）
 
 tip 位置从 `z_bar` 解包得到（与 `fk_engine._extract_tip_pose_dim` 逻辑一致）：
 
 $$
 r_p(y)=\frac{p_{\text{tip}}(z_{\text{bar}})-p_{\text{des}}}{\sigma_p}
 $$
 
 3) 控制正则（可选，但推荐保留以提升可解性/唯一性）
 
 $$
 r_I(y)=w_I\,I(u)
 $$
 
 $$
 r_x(y)=w_x\,(x(u)-x_{\text{ref}})
 $$
 
 4) 平滑/连续性项（可选，若你希望解随时间/迭代平滑）
 
 $$
 r_{dI}(y)=w_{dI}\,(I(u)-I_{\text{prev}}),\quad r_{dx}(y)=w_{dx}\,(x(u)-x_{\text{prev}})
 $$
 
 以及你已有的耦合项（可选）：
 
 $$
 r_{Ix}(y)=w_{Ix}\,\frac{I(u)-I_{\text{prev}}}{\max(|x(u)-x_{\text{prev}}|,\ dx_{\text{floor}})}
 $$
 
 最终
 
 $$
 R(y)=\begin{bmatrix}
 r_E\\r_p\\r_I\\r_x\\r_{dI}\\r_{dx}\\r_{Ix}
 \end{bmatrix}
 $$
 
 ### 3.3 为什么这是“单层”
 
 - 在每次 LM 迭代中，`Δz_bar` 与 `Δu` 由同一个正规方程联合求解得到。
 - 不再需要“固定 `(I,x)` 然后内层把 `z` 求到收敛”，`z` 会在联合残差里被约束与推动。
 - 当 `w_E` 足够大时，平衡会被强制满足，行为会逐渐接近双层方法，但更新仍是单层完成。
 
 ---
 
 ## 4. 单层 LM 求解器：算法细节
 
 ### 4.1 LM 线性化与正规方程
 
 对残差做一阶近似：
 
 $$
 R(y+\Delta y)\approx R(y)+J\Delta y,\quad J=\frac{\partial R}{\partial y}
 $$
 
 LM 步：
 
 $$
 (J^T J + \lambda D)\Delta y=-J^T R
 $$
 
 其中 `D` 推荐取 `diag(J^T J)`（与 `inverse_kinematics/ik_multisite_lm_opt.py` 的外层策略一致），或简化为 `I`。
 
 ### 4.2 Accept/Reject 与阻尼更新
 
 预测下降（可用 `R_lin = R + JΔy`）：
 
 $$
 \text{pred}=\frac12\left(\|R\|^2-\|R_{\text{lin}}\|^2\right)
 $$
 
 实际下降：
 
 $$
 \text{act}=\frac12\left(\|R\|^2-\|R(y+\Delta y)\|^2\right)
 $$
 
 比例：
 
 $$
 \rho=\frac{\text{act}}{\max(\text{pred},\epsilon)}
 $$
 
 可直接复用 `forward_kinematics_optimized/equilibrium_solver_nondim.py` 中现有 LM 的 damping policy（`rho>0.75` 降、`rho<=0` 升等）。
 
 ### 4.3 Jacobian 计算策略（推荐顺序）
 
 - 第一版（推荐）：直接用 JAX 对 `R(y)` 做 `jacfwd/jacrev` 得到 `J`。
   - 优点：实现快、正确性更稳、便于与数值差分对照。
   - 代价：一次迭代构造较大 Jacobian。
 
 - 第二版（性能优化，可选）：利用块结构拼装：
   - `dE/dz`：已有 `lm_stats.J` 或 `solver_cached.jacobian_jit(z, params)`
   - `dE/dI`、`dE/dL1`：对 `residual_bar` 对参数做求导（要求 `params` 结构稳定）
   - `dp/dz`：与 `inverse_kinematics/ik_diff.py` 中 `_DP_DZ_REV` 同理
   - 再通过链式法则得到 `dR/du`
 
 ---
 
 ## 5. 与现有代码的映射关系（工程落地要点）
 
 ### 5.1 必须复用/调用的现有函数
 
 - 平衡残差：`forward_kinematics_optimized.equilibrium_solver_nondim.residual_bar`
 - 参数更新：
   - `forward_kinematics_optimized.equilibrium_solver_nondim.with_coil_currents`
   - `forward_kinematics_optimized.equilibrium_solver_nondim.with_L1_dim`
 - advancer 语义：`forward_kinematics_optimized.advancer_nondim.apply_advancer_protrude_length`
 
 tip 提取建议“复制同逻辑”，避免依赖 engine 内部状态：
 
 - `fk_engine._extract_tip_pose_dim` 的实现是：
   - 解包 `z_bar`，取最后一段柔性最后一个 node 的 `x_tip_bar`
   - 用 `x_bar_to_dim` 得到 `x_tip_dim`
   - `p_tip = x_tip_dim[0:3]`
 
 ### 5.2 `x` 如何进入平衡残差（必须写清楚）
 
 在现有 FK 模型里：插入深度 `x=L_protrude` 并不是直接出现在平衡方程中，而是通过
 
 $$
 L_1=x-L_{\text{fixed}}
 $$
 
 改变第一段柔性长度，从而改变 `params.L1_dim`，同时 FK 引擎会更新第一段 mesh：`build_uniform_mesh(L1_dim, M_list[0])`。
 
 单层联合求解第一版建议：
 
 - 固定离散结构（`M_list` 不变）
 - 允许 `L1_dim` 连续变化
 - 注意 JAX retrace：`params` PyTree 结构必须稳定（leaf shape/dtype 不要切换）
 
 ---
 
 ## 6. 建议的代码落地位置：`single_layer_ik/` 目录结构
 
 建议新增：
 
 - `single_layer_ik/joint_problem.py`
   - `u_to_Ix(u, I_max, x_min, x_max) -> (I, x, dI_du, dx_du)`（可直接复用/复制 `inverse_kinematics/ik_multisite_lm_opt.py` 的版本）
   - `build_params_from_base(params_base, I, L1_dim)`：`with_coil_currents` + `with_L1_dim`
   - `extract_tip_p_dim(z_bar, scales, M_list)`：复刻 `fk_engine` 的 tip 提取
   - `joint_residual(y, params_base, ...) -> R`
 
 - `single_layer_ik/joint_lm_solver.py`
   - 实现 `JointLMSolverCached`：
     - `residual_jit = jax.jit(joint_residual)`
     - `jacobian_jit = jax.jit(jax.jacfwd(joint_residual, argnums=0))`（或 `jacrev`）
     - accept/reject 与 damping policy 参考 `MultiSegmentEquilibriumSolverNondimJAXCached.solve_lm`
 
 - `single_layer_ik/run_joint_ik.py`
   - CLI 参数尽量对齐 `inverse_kinematics/ik_multisite_lm_opt.py`
   - 输出 `ik_out/run_xxx/{run_config.json, history.json, result.json}` 便于对照与复用现有可视化
 
 ---
 
 ## 7. 对 `forward_kinematics_optimized` 的最小改造建议（强烈建议做）
 
 这些改造不会破坏现有 FK/IK，但能显著降低单层联合求解的风险。
 
 ### 7.1 固定 `params.coil_currents` 的 shape（避免 None）
 
 目前 `forward_kinematics_optimized/fk.py::build_solver_params` 在 `enable_magnetics=False` 时会令 `coil_currents_out = None`，导致：
 
 - `SolverParams` PyTree leaf 结构不稳定（None/array 切换）
 - 对 `I` 求导可能出现结构不依赖/不可导/触发 retrace
 
 建议：无论是否 enable magnetics，都令 `coil_currents` 为 `jnp.zeros((8,), dtype=jnp.float64)`，仅当 `magnetic_model is not None` 时磁力项才贡献非零。
 
 ### 7.2（可选）增加显式 residual 接口便于对参数求导
 
 可以新增辅助函数（不影响原逻辑）：
 
 - `residual_bar_zIL1(z_bar, coil_currents, L1_dim, params_static_base) -> E`
 
 让 `(I, L1)` 作为显式自变量，避免在 `joint_residual` 内部构造复杂 params 时引入 PyTree 静态字段变化。
 
 ---
 
 ## 8. 超参数与预期效果
 
 ### 8.1 关键超参数：`w_E`
 
 `w_E` 决定“平衡优先级”：
 
 - `w_E` 太小：算法可能为了贴近 `p_des` 而牺牲平衡，得到不物理的 `z`。
 - `w_E` 足够大：平衡会被强制满足，算法行为趋近双层，但更新仍是单层完成。
 
 建议初始化：
 
 $$
 w_E \approx \frac{1}{\text{tol}_E}\ \text{或}\ 10\times\frac{1}{\text{tol}_E}
 $$
 
 并可做 continuation：逐步增大 `w_E`。
 
 ### 8.2 预期效果
 
 - 结构上消除“外层每步完整 FK 收敛”的硬嵌套。
 - 第一版实现可不依赖 `compute_dp_dI_via_lm_adjoint` 与 `dp/dx` 有限差分。
 - 为后续做“严格约束/增广拉格朗日”提供稳定基线。
 
 ### 8.3 主要风险与缓解
 
 - 平衡与目标拉扯：通过增大 `w_E`、continuation、或增广拉格朗日（后续版本）缓解。
 - 收敛域变窄：初始化可先用一次 FK 给 `z0`；LM 阻尼/信赖域要稳健。
 - JAX retrace：确保 `params` PyTree 结构稳定（尤其 `coil_currents` 不为 None）。
 
 ---
 
 ## 9. 验证计划（必须做的对照实验）
 
 - 对照 1：与旧 IK 输出对齐
   - 同一个 `p_des`，比较 `(I,x)`、末端误差 `||p_tip-p_des||`、以及平衡残差 `||E||`。
 
 - 对照 2：Jacobian sanity check
   - 抽取 `u` 的某一维做数值差分，检查 `J` 中对应列（至少在小问题/较小 `M_list` 上）。
 
 - 对照 3：稳定性
   - 多随机初值/不同 `p_des` 统计成功率、迭代次数、NaN 发生率。
 
 ---
 
 ## 10. 建议的实现顺序（降低风险）
 
 - Step 0：确保 `params.coil_currents` 总是 shape=(8,) 的 array。
 - Step 1：实现 `joint_residual(y)`，并能在固定 `y` 下输出合理的 `R`。
 - Step 2：实现 `JointLMSolverCached`，先在小模型（较小 `M_list`）上跑通。
 - Step 3：加入 `ik_out` 输出与可视化复用（可复用 `inverse_kinematics` 的 history 保存逻辑）。
 - Step 4：性能优化（若需要）：块 Jacobian/伴随、预条件等。
 
 ---
 
 ## 11. 附：符号与变量对照表
 
 - `z_bar`：FK 的 packed equilibrium state（nondim）。对应 `fk_engine.solve_with_stats` 的返回 `z_star_bar`。
 - `params`：`SolverParams`（见 `equilibrium_solver_nondim.py`），由 `fk.build_solver_params` 构造并用 `with_coil_currents/with_L1_dim` 更新。
 - `E(z, params)`：平衡残差向量，对应 `residual_bar`。
 - `p_tip(z)`：末端位置，从 `z` 解包并尺度还原得到，对应 `query_sites()['tip_p_dim']`。
 - `I`：线圈电流向量（8维）。建议用 `I=I_max*tanh(u_I)`。
 - `x`：插入深度（`L_protrude`），建议用 sigmoid 参数化。
 - `L1_dim`：第一段柔性长度，由 advancer 语义推导 `L1=x-L_fixed`。
