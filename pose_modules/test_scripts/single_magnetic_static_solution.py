"""Solve and visualize a single flexible-rigid magnetic catheter example."""

import numpy as np
import matplotlib.pyplot as plt

from pose_modules.segments import FlexibleSegment, RigidSegment
from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver
from pose_modules.rod_mesh import RodMesh


def compute_catheter_points(
        mesh: RodMesh,
        rigid: RigidSegment,
        x_nodes_star: np.ndarray,
        f_ext_rigid: np.ndarray,
        tau_ext_rigid: np.ndarray,
        n_samples_rigid: int = 20,
) -> np.ndarray:
    """Assemble flexible and rigid samples into one centerline."""
    p_flex = x_nodes_star[:, :3]

    x_flex_end = x_nodes_star[-1]
    Lr = rigid.length
    sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

    p_rigid_list = []
    for s in sigmas_rigid[1:]:
        x_s = rigid.state_along(
            x_proximal=x_flex_end,
            sigma=s,
            f_ext_total=f_ext_rigid,
            tau_ext_total=tau_ext_rigid,
        )
        p_rigid_list.append(x_s[:3])

    p_rigid = np.array(p_rigid_list)
    P = np.vstack([p_flex, p_rigid])
    return P


def demo_one_flex_one_rigid():
    flex = FlexibleSegment(
        length=0.1,
        K_se=np.eye(3),
        K_bt=np.eye(3),
    )

    rigid = RigidSegment(
        length=0.02,
        v_star=np.array([0.0, 0.0, 1.0]),
    )

    mesh = RodMesh(flex_seg=flex, n_intervals=5)
    # mesh = RodMesh(flex_seg=flex, sigma_nodes=np.array([0.0, 0.02, 0.05, 0.1]))

    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    f_ext_rigid = np.array([0., 0., 0.])
    tau_ext_rigid = np.array([30., 0., 0.])

    solver = MultiSegmentEquilibriumSolver(
        meshes=[mesh],
        rigid_segs=[rigid],
        flex_segs=[flex],
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=[f_ext_rigid],
        tau_ext_list=[tau_ext_rigid],
        max_iter=20000,
        tol=1e-5,
        lm_damping=1e-3,
    )

    M = mesh.M
    sigmas = mesh.sigma_nodes
    x_nodes = np.zeros((M + 1, 13))
    for n in range(M + 1):
        p = np.array([0.0, 0.0, sigmas[n]])
        Q = Q0_target
        f = np.zeros(3)
        tau = np.zeros(3)
        x_nodes[n] = np.concatenate([p, Q, f, tau])

    p_flex_end = x_nodes[-1][:3]
    p_rigid_end = p_flex_end + np.array([0.0, 0.0, rigid.length])
    x_rigid = np.concatenate([p_rigid_end, Q0_target, np.zeros(3), np.zeros(3)])

    # k_array
    k_array = np.zeros((M, 3, 13))

    z0 = solver.pack_z(x_nodes, k_array, x_rigid)
    
    n_dirs = 16
    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    tau_mag = 15.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.hsv(np.linspace(0, 1, n_dirs, endpoint=False))

    end_points = []
    z_init = z0.copy()

    for i, theta in enumerate(angles):
        tau_dir = np.array([np.cos(theta), np.sin(theta), 0.0])
        tau_i = tau_mag * tau_dir
        solver.tau_ext_rigid = tau_i
        solver.f_ext_rigid = f_ext_rigid

        z_star, success = solver.solve(z_init)
        print(f"LM success (i={i}, theta={theta:.3f}):", success)
        x_nodes_star, k_array_star, x_rigid_star = solver.unpack_z(z_star)

        P = compute_catheter_points(
            mesh=mesh,
            rigid=rigid,
            x_nodes_star=x_nodes_star[0],
            f_ext_rigid=f_ext_rigid,
            tau_ext_rigid=tau_i,
            n_samples_rigid=20,
        )
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colors[i], marker='o', markersize=3, linewidth=1.5)

        end_points.append(P[-1])

        z_init = z_star

    end_points = np.array(end_points)
    end_points_closed = np.vstack([end_points, end_points[0]])
    ax.plot(end_points_closed[:, 0], end_points_closed[:, 1], end_points_closed[:, 2], 'k--', linewidth=1.5, label='end-tip trajectory')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('16 catheter poses with rotating torque in xOy (|tau|=15)')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_one_flex_one_rigid()
