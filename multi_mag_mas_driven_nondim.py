# multi_mag_mas_driven_nondim.py
from __future__ import annotations

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Nondim modules ----------------------
from pose_modules_nondim.equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondim
from pose_modules_nondim.segments_nondim import RigidSegment, FlexibleSegment
from pose_modules_nondim.rod_mesh_nondim import RodMesh
from pose_modules_nondim.nondim import compute_default_scales, NondimScales, x_bar_to_dim

# external_wrench / mas：保持 SI 物理逻辑不变
try:
    from pose_modules_nondim.external_wrench_nondim import (
        GravityRigid,
        make_external_wrench_density_flexible,
        compute_external_wrench_total_rigid,
    )
except Exception:
    # 如果你暂时没做 external_wrench_nondim 的 re-export 壳，则回退到原版
    from pose_modules.external_wrench import (
        GravityRigid,
        make_external_wrench_density_flexible,
        compute_external_wrench_total_rigid,
    )

try:
    from pose_modules_nondim.mas_nondim import MagneticActuationSystem, SupieeMagneticModel
except Exception:
    from pose_modules.mas import MagneticActuationSystem, SupieeMagneticModel

# utils：这里会用到 PDMS/磁体的材料-几何构造 helper
# 强烈建议你在 utils_nondim.py 里 re-export 这些 helper；此处做了兼容回退
try:
    from pose_modules_nondim.utils_nondim import (
        build_k_matrices_for_pdms,
        build_gravity_line_density_for_pdms,
        build_gravity_rigid_for_ndfeb,
        make_initial_guess_multi_bar,
    )
except Exception:
    from pose_modules.utils import (
        build_k_matrices_for_pdms,
        build_gravity_line_density_for_pdms,
        build_gravity_rigid_for_ndfeb,
    )
    from pose_modules_nondim.utils_nondim import make_initial_guess_multi_bar


def extract_catheter_points_bar(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list_bar: List[np.ndarray],
        x_rigid_list_bar: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        scales: NondimScales,
        n_samples_rigid: int = 10,
) -> np.ndarray:
    """
    从无量纲解 (bar) 提取导管三维点坐标（单位：m），用于绘图。

    返回：
    - P: (N_points, 3) 导管上所有采样点的坐标（SI, meters）
    """
    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i_bar = x_nodes_list_bar[i]
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        # 柔性段：p_bar -> p_dim (m)
        p_flex_i = x_nodes_i_bar[:, :3] * scales.L_ref
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        # 刚性段：需要在 SI 下计算外载并采样
        x_flex_end_bar = x_nodes_i_bar[-1]
        x_flex_end_dim = x_bar_to_dim(x_flex_end_bar, scales)

        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_dim,
            rigid_length=rigid_i.length,
            gravity=gravity_rigid_i,
            magnetic_model=magnetic_model,
            magnet_params=magnet_params_i,
            coil_currents=coil_currents,
        )

        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)
        p_rigid_i = []
        for s in sigmas_rigid[1:]:
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_dim,
                sigma=float(s),
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    return np.vstack(P_all)


# ---------------------- Iteration callback (uses globals) ----------------------
solver: Optional[MultiSegmentEquilibriumSolverNondim] = None
meshes: Optional[List[RodMesh]] = None
rigid_segs: Optional[List[RigidSegment]] = None
gravity_rigid_list: Optional[List[GravityRigid]] = None
supiee_model: Optional[SupieeMagneticModel] = None
magnet_params_list: Optional[List[dict]] = None
coil_currents: Optional[np.ndarray] = None
iteration_history = []
ax = None
scales: Optional[NondimScales] = None


def iteration_callback(iter_num, z, normE):
    """每次迭代的回调函数：每10步绘制一次导管姿态（以 SI 米制绘图）。"""
    global solver, meshes, rigid_segs, gravity_rigid_list, supiee_model, magnet_params_list, coil_currents, ax, scales
    if iter_num % 10 != 0:
        return

    x_nodes_list_bar, _, x_rigid_list_bar = solver.unpack_z(z)

    P = extract_catheter_points_bar(
        meshes=meshes,
        rigid_segs=rigid_segs,
        x_nodes_list_bar=x_nodes_list_bar,
        x_rigid_list_bar=x_rigid_list_bar,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=supiee_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        scales=scales,
        n_samples_rigid=10,
    )

    iteration_history.append({'iter': iter_num, 'P': P.copy(), 'normE': normE})

    # 透明度映射（保持你原逻辑）
    if normE > 1e-2:
        alpha = 0.9
    elif normE > 1e-3:
        alpha = 0.6
    elif normE > 1e-4:
        alpha = 0.3
    else:
        alpha = 0.1

    color_intensity = 1.0 - alpha
    color = (
        0.2 * (1 - color_intensity),
        0.4 * (1 - color_intensity),
        0.8 + 0.2 * color_intensity
    )

    ax.plot(P[:, 0], P[:, 1], P[:, 2],
            color=color, alpha=alpha, linewidth=1.5,
            label=f'iter={iter_num}, ||E||={normE:.2e}')

    ax.set_title(f'convergence history (iter={iter_num}, ||E||={normE:.3e})')
    plt.draw()
    plt.pause(0.01)


def plot_catheter_3d_multiseg_with_magnetics_bar(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list_star_bar: List[np.ndarray],
        x_rigid_list_star_bar: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        scales: NondimScales,
        n_samples_rigid: int = 10,
):
    """
    使用最终无量纲解 (bar)，在 SI 单位下重新计算每个刚性段总外载并采样，绘制最终姿态。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        rigid_i = rigid_segs[i]
        x_nodes_i_bar = x_nodes_list_star_bar[i]
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        # 柔性段点：bar -> meters
        p_flex_i = x_nodes_i_bar[:, :3] * scales.L_ref
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        # 刚性段近端：flex 末端（SI）
        x_flex_end_dim = x_bar_to_dim(x_nodes_i_bar[-1], scales)

        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_dim,
            rigid_length=rigid_i.length,
            gravity=gravity_rigid_i,
            magnetic_model=magnetic_model,
            magnet_params=magnet_params_i,
            coil_currents=coil_currents,
        )

        Lr = rigid_i.length
        sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)
        p_rigid_i = []
        for s in sigmas_rigid[1:]:
            x_s = rigid_i.state_along(
                x_proximal=x_flex_end_dim,
                sigma=float(s),
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)

    fig = plt.figure()
    ax_ = fig.add_subplot(111, projection='3d')
    ax_.plot(P[:, 0], P[:, 1], P[:, 2], marker='o', markersize=3, linewidth=1.5)

    ax_.set_xlabel('X [m]')
    ax_.set_ylabel('Y [m]')
    ax_.set_zlabel('Z [m]')
    ax_.set_title('Multi-segment catheter (gravity + magnetic) configuration (nondim solve)')
    ax_.view_init(elev=30, azim=-60)
    ax_.set_box_aspect([1, 1, 1])

    ax_.set_xlim([-0.1, 0.1])
    ax_.set_ylim([-0.1, 0.1])
    ax_.set_zlim([-0.05, 0.05])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ------------------- 1) Geometry & material -------------------
    d_outer = 1.5e-3  # m

    # Flexible lengths
    Lf1 = 0.03
    Lf2 = 0.03
    Lf3 = 0.03

    # Rigid NdFeB lengths
    Lr1 = Lr2 = Lr3 = 0.003

    # Stiffness matrices (SI)
    K_se_pdms, K_bt_pdms = build_k_matrices_for_pdms(d_outer)
    print(f"K_se_pdms:\n{K_se_pdms}; K_bt_pdms:\n{K_bt_pdms}")

    flex1 = FlexibleSegment(length=Lf1, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex2 = FlexibleSegment(length=Lf2, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex3 = FlexibleSegment(length=Lf3, K_se=K_se_pdms, K_bt=K_bt_pdms)

    # Flexible gravity line density (SI)
    grav_line = build_gravity_line_density_for_pdms(d_outer)
    fext1, tauext1 = make_external_wrench_density_flexible(gravity=grav_line)
    fext2, tauext2 = make_external_wrench_density_flexible(gravity=grav_line)
    fext3, tauext3 = make_external_wrench_density_flexible(gravity=grav_line)

    flex1.fext_density, flex1.tauext_density = fext1, tauext1
    flex2.fext_density, flex2.tauext_density = fext1, tauext2
    flex3.fext_density, flex3.tauext_density = fext1, tauext3

    # Meshes (IMPORTANT: use corresponding flex segment)
    mesh1 = RodMesh(flex_seg=flex1, n_intervals=5)
    mesh2 = RodMesh(flex_seg=flex2, n_intervals=5)
    mesh3 = RodMesh(flex_seg=flex3, n_intervals=5)

    rigid1 = RigidSegment(length=Lr1, v_star=np.array([0.0, 0.0, 1.0]))
    rigid2 = RigidSegment(length=Lr2, v_star=np.array([0.0, 0.0, 1.0]))
    rigid3 = RigidSegment(length=Lr3, v_star=np.array([0.0, 0.0, 1.0]))

    flex_segs = [flex1, flex2, flex3]
    meshes = [mesh1, mesh2, mesh3]
    rigid_segs = [rigid1, rigid2, rigid3]

    # ------------------- 2) Rigid gravity (SI) -------------------
    gravity_rigid_list = [
        build_gravity_rigid_for_ndfeb(d_outer, Lr1),
        build_gravity_rigid_for_ndfeb(d_outer, Lr2),
        build_gravity_rigid_for_ndfeb(d_outer, Lr3),
    ]

    # ------------------- 3) Magnetic actuation (SI) -------------------
    calib_file = "calib/mpem_calibration_file_sp=40_order=1.yaml"
    mas = MagneticActuationSystem(calib_file=calib_file)
    supiee_model = SupieeMagneticModel(mas=mas)

    # coil_currents = np.array([21.6536,21.6208,-2.6466,-4.0574,17.0103,-9.3642,14.9133,-18.8761], dtype=np.float64)
    coil_currents = np.array([3.3832,24.4370,-30.7323,-2.5516,20.1070,44.8393,23.1883,35.1559], dtype=np.float64)

    m_mag = 0.005301  # A·m^2
    magnet_params_list = [
        {"m_body": np.array([m_mag, 0.0, 0.0])},  # R1
        {"m_body": np.array([0.0, 0.0, -m_mag])},  # R2
        {"m_body": np.array([0.0, 0.0, m_mag])},  # R3
    ]

    # Manual extra loads (SI)
    N_pairs = 3
    f_ext_list = [np.zeros(3) for _ in range(N_pairs)]
    tau_ext_list = [np.zeros(3) for _ in range(N_pairs)]

    # ------------------- 4) Boundary conditions (SI input; solver will scale) -------------------
    p0_target = np.array([0.0, 0.0, -50.0]) * 1e-3
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    # ------------------- 5) Scales + initial guess in BAR -------------------
    scales = compute_default_scales(flex_segs, rigid_segs)

    z0_bar, _, _, _ = make_initial_guess_multi_bar(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        scales=scales,
    )

    # ------------------- 6) Nondim solver -------------------
    solver = MultiSegmentEquilibriumSolverNondim(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=f_ext_list,
        tau_ext_list=tau_ext_list,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=supiee_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        scales=scales,
        max_iter=20000,
        tol=1e-6,
        lm_damping=1e-4,
    )

    # ------------------- 7) Visualization during iteration -------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Convergence visualization (every 10 iters) [nondim solve]')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.05, 0.05])

    plt.ion()
    plt.show(block=False)

    iteration_history = []

    z_star_bar, success = solver.solve(z0_bar, callback=iteration_callback)
    print("Multi-segment LM success:", success)

    x_nodes_list_star_bar, _, x_rigid_list_star_bar = solver.unpack_z(z_star_bar)

    # Print endpoints in SI meters (convert from bar)
    for i in range(N_pairs):
        p_flex_end_m = x_nodes_list_star_bar[i][-1, :3] * scales.L_ref
        p_rigid_end_m = x_rigid_list_star_bar[i][:3] * scales.L_ref
        print(f"Segment pair {i + 1}:")
        print("  F end p [m] =", p_flex_end_m)
        print("  R end p [m] =", p_rigid_end_m)

    plt.ioff()

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 5:
        selected_handles = handles[:3] + handles[-2:]
        selected_labels = labels[:3] + labels[-2:]
        ax.legend(selected_handles, selected_labels, loc='upper right', fontsize=8)
    else:
        ax.legend(loc='upper right', fontsize=8)

    print(f"\nIteration visualization complete: {len(iteration_history)} snapshots stored.")
    plt.show()

    # Optional: final pose plot
    if success:
        plot_catheter_3d_multiseg_with_magnetics_bar(
            meshes=meshes,
            rigid_segs=rigid_segs,
            x_nodes_list_star_bar=x_nodes_list_star_bar,
            x_rigid_list_star_bar=x_rigid_list_star_bar,
            gravity_rigid_list=gravity_rigid_list,
            magnetic_model=supiee_model,
            magnet_params_list=magnet_params_list,
            coil_currents=coil_currents,
            scales=scales,
            n_samples_rigid=20,
        )
