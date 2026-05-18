from __future__ import annotations
from typing import List
import matplotlib.pyplot as plt
from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver
from pose_modules.segments import RigidSegment, FlexibleSegment
from pose_modules.rod_mesh import RodMesh
from pose_modules.utils import *
from pose_modules.external_wrench import (
    GravityLineDensity,
    GravityRigid,
    make_external_wrench_density_flexible,
    compute_external_wrench_total_rigid,
)
from pose_modules.mas import (
    MagneticActuationSystem,
    SupieeMagneticModel,
)


def extract_catheter_points(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list: List[np.ndarray],
        x_rigid_list: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        n_samples_rigid: int = 10,
) -> np.ndarray:
    """Load extract catheter points ."""
    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list[i]
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        p_flex_i = x_nodes_i[:, :3]
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        x_flex_end_i = x_nodes_i[-1]
        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_i,
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
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)
    return P


def iteration_callback(iter_num, z, normE):
    """Render the current centerline every few LM iterations."""
    if iter_num % 10 != 0:
        return

    x_nodes_list, k_array_list, x_rigid_list = solver.unpack_z(z)

    P = extract_catheter_points(
        meshes=meshes,
        rigid_segs=rigid_segs,
        x_nodes_list=x_nodes_list,
        x_rigid_list=x_rigid_list,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=supiee_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        n_samples_rigid=10,
    )

    iteration_history.append({
        'iter': iter_num,
        'P': P.copy(),
        'normE': normE
    })

    if normE > 1e-2:
        alpha = 0.9
    elif normE > 1e-3:
        alpha = 0.6
    elif normE > 1e-4:
        alpha = 0.3
    else:
        alpha = 0.1

    color_intensity = 1.0 - alpha
    color = (0.2 * (1 - color_intensity), 0.4 * (1 - color_intensity), 0.8 + 0.2 * color_intensity)

    ax.plot(P[:, 0], P[:, 1], P[:, 2],
            color=color, alpha=alpha, linewidth=1.5,
            label=f'iter={iter_num}, ||E||={normE:.2e}')

    ax.set_title(f'convergence history (iter={iter_num}, ||E||={normE:.3e})')

    plt.draw()
    plt.pause(0.01)


def plot_catheter_3d_multiseg_with_magnetics(
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        x_nodes_list_star: List[np.ndarray],
        x_rigid_list_star: List[np.ndarray],
        gravity_rigid_list: List[GravityRigid],
        magnetic_model: SupieeMagneticModel,
        magnet_params_list: List[dict],
        coil_currents: np.ndarray,
        n_samples_rigid: int = 10,
):
    """Plot the final multi-segment catheter shape with magnetic loading."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    N = len(rigid_segs)
    P_all = []

    for i in range(N):
        mesh_i = meshes[i]
        rigid_i = rigid_segs[i]
        x_nodes_i = x_nodes_list_star[i]  # (Mi+1, 13)
        gravity_rigid_i = gravity_rigid_list[i]
        magnet_params_i = magnet_params_list[i]

        p_flex_i = x_nodes_i[:, :3]
        if i > 0:
            p_flex_i = p_flex_i[1:, :]
        P_all.append(p_flex_i)

        x_flex_end_i = x_nodes_i[-1]

        f_ext_total_i, tau_ext_total_i = compute_external_wrench_total_rigid(
            x_proximal=x_flex_end_i,
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
                x_proximal=x_flex_end_i,
                sigma=s,
                f_ext_total=f_ext_total_i,
                tau_ext_total=tau_ext_total_i,
            )
            p_rigid_i.append(x_s[:3])
        p_rigid_i = np.array(p_rigid_i)
        P_all.append(p_rigid_i)

    P = np.vstack(P_all)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o', markersize=3, linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-segment catheter (gravity + magnetic) configuration')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.15])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    d_outer = 1.5e-3  # m

    Lf1 = 0.05
    Lf2 = 0.03
    Lf3 = 0.01

    Lr1 = Lr2 = Lr3 = 0.03

    K_se_pdms, K_bt_pdms = build_k_matrices_for_pdms(d_outer)
    print(f"K_se_pdms:\n{K_se_pdms}; K_bt_pdms:\n{K_bt_pdms}")

    flex1 = FlexibleSegment(length=Lf1, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex2 = FlexibleSegment(length=Lf2, K_se=K_se_pdms, K_bt=K_bt_pdms)
    flex3 = FlexibleSegment(length=Lf3, K_se=K_se_pdms, K_bt=K_bt_pdms)

    grav_line = build_gravity_line_density_for_pdms(d_outer)
    fext1, tauext1 = make_external_wrench_density_flexible(gravity=grav_line)
    fext2, tauext2 = make_external_wrench_density_flexible(gravity=grav_line)
    fext3, tauext3 = make_external_wrench_density_flexible(gravity=grav_line)

    flex1.fext_density = fext1
    flex1.tauext_density = tauext1
    flex2.fext_density = fext2
    flex2.tauext_density = tauext2
    flex3.fext_density = fext3
    flex3.tauext_density = tauext3

    mesh1 = RodMesh(flex_seg=flex1, n_intervals=3)
    mesh2 = RodMesh(flex_seg=flex1, n_intervals=3)
    mesh3 = RodMesh(flex_seg=flex1, n_intervals=3)

    rigid1 = RigidSegment(length=Lr1, v_star=np.array([0.0, 0.0, 1.0]))
    rigid2 = RigidSegment(length=Lr2, v_star=np.array([0.0, 0.0, 1.0]))
    rigid3 = RigidSegment(length=Lr3, v_star=np.array([0.0, 0.0, 1.0]))

    flex_segs = [flex1, flex2, flex3]
    meshes = [mesh1, mesh2, mesh3]
    rigid_segs = [rigid1, rigid2, rigid3]

    gravity_rigid_list = [
        build_gravity_rigid_for_ndfeb(d_outer, Lr1),
        build_gravity_rigid_for_ndfeb(d_outer, Lr2),
        build_gravity_rigid_for_ndfeb(d_outer, Lr3),
    ]

    calib_file = "../../calib/mpem_calibration_file_sp=40_order=1.yaml"
    mas = MagneticActuationSystem(calib_file=calib_file)
    supiee_model = SupieeMagneticModel(mas=mas)

    coil_currents = np.random.rand(8) * 50.0

    m_mag = 0.005301
    magnet_params_list = [
        {"m_body": np.array([0.0, 0.0, m_mag])},   # R1
        {"m_body": np.array([0.0, 0.0, m_mag])},   # R2
        {"m_body": np.array([0.0, 0.0, m_mag])},  # R3
    ]

    N_pairs = 3
    f_ext_list = [np.zeros(3) for _ in range(N_pairs)]
    tau_ext_list = [np.zeros(3) for _ in range(N_pairs)]


    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    z0, x_node_list0, k_array_list0, x_rigid_list0 = make_initial_guess_multi(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
    )

    solver = MultiSegmentEquilibriumSolver(
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
        max_iter=20000,
        tol=1e-6,
        lm_damping=1e-4,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Catheter convergence visualization')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0, 0.15])

    plt.ion()
    plt.show(block=False)

    iteration_history = []

    z_star, success = solver.solve(z0, callback=iteration_callback)
    print("Multi-segment LM success:", success)

    x_nodes_list_star, k_array_list_star, x_rigid_list_star = solver.unpack_z(z_star)

    for i in range(N_pairs):
        p_flex_end = x_nodes_list_star[i][-1, :3]
        p_rigid_end = x_rigid_list_star[i][:3]
        print(f"Segment pair {i+1}:")
        print("  F end p =", p_flex_end)
        print("  R end p =", p_rigid_end)

    plt.ioff()

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 5:
        selected_handles = handles[:3] + handles[-2:]
        selected_labels = labels[:3] + labels[-2:]
        ax.legend(selected_handles, selected_labels, loc='upper right', fontsize=8)
    else:
        ax.legend(loc='upper right', fontsize=8)

    print(f"\nConvergence visualization complete. Rendered {len(iteration_history)} iteration states.")
    plt.show()

    if success:
        plot_catheter_3d_multiseg_with_magnetics(
            meshes=meshes,
            rigid_segs=rigid_segs,
            x_nodes_list_star=x_nodes_list_star,
            x_rigid_list_star=x_rigid_list_star,
            gravity_rigid_list=gravity_rigid_list,
            magnetic_model=supiee_model,
            magnet_params_list=magnet_params_list,
            coil_currents=coil_currents,
            n_samples_rigid=20,
        )





