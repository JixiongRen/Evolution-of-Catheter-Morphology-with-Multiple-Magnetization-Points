"""Visualize LM convergence for a single-torque benchmark."""

import numpy as np
import matplotlib.pyplot as plt

from pose_modules.segments import FlexibleSegment, RigidSegment
from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver
from pose_modules.rod_mesh import RodMesh


def compute_catheter_points(
    mesh: RodMesh,
    rigid: RigidSegment,
    x_nodes: np.ndarray,
    f_ext_rigid: np.ndarray,
    tau_ext_rigid: np.ndarray,
    n_samples_rigid: int = 20,
) -> np.ndarray:
    """Assemble flexible and rigid samples into one centerline."""
    p_flex = x_nodes[:, :3]

    x_flex_end = x_nodes[-1]
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


def lm_convergence_demo():
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

    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    f_ext_rigid = np.array([0.0, 0.0, 0.0])
    tau_vec = np.array([0., 15., 0.0])

    solver = MultiSegmentEquilibriumSolver(
        flex_segs=[flex],
        meshes=[mesh],
        rigid_segs=[rigid],
        p0_target=p0_target,
        Q0_target=Q0_target,
        f_ext_list=[f_ext_rigid],
        tau_ext_list=[tau_vec],
        max_iter=20000,
        tol=1e-5,
        lm_damping=1e-3,
    )

    M = mesh.M
    sigmas = mesh.sigma_nodes

    x_nodes0 = np.zeros((M + 1, 13))
    for n in range(M + 1):
        p = np.array([0.0, 0.0, sigmas[n]])
        Q = Q0_target
        f = np.zeros(3)
        tau = np.zeros(3)
        x_nodes0[n] = np.concatenate([p, Q, f, tau])

    p_flex_end = x_nodes0[-1][:3]
    p_rigid_end = p_flex_end + np.array([0.0, 0.0, rigid.length])
    x_rigid0 = np.concatenate([p_rigid_end, Q0_target, np.zeros(3), np.zeros(3)])

    k_array0 = np.zeros((M, 3, 13))
    z = solver.pack_z(x_nodes0, k_array0, x_rigid0)

    max_iter = 200
    tol = 1e-5
    lm_damping = 1e-3

    z_history = []
    E_history = []

    for it in range(max_iter):
        E = solver.residual(z)
        normE = np.linalg.norm(E)
        z_history.append(z.copy())
        E_history.append(normE)
        print(f"[LM-visual] iter={it}, ||E||={normE:.3e}")
        if normE < tol:
            break

        J = solver.jacobian_fd(z, E)
        A = J.T @ J + lm_damping * np.eye(J.shape[1])
        g = J.T @ E
        delta = np.linalg.solve(A, g)
        z = z - delta

    n_snapshots = 12
    idx_all = np.arange(len(z_history))
    if len(idx_all) <= n_snapshots:
        idx_pick = idx_all
    else:
        idx_pick = np.linspace(0, len(z_history) - 1, n_snapshots, dtype=int)

    alphas = np.linspace(0.15, 1.0, len(idx_pick))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for a, idx in zip(alphas, idx_pick):
        x_nodes_i, k_array_i, x_rigid_i = solver.unpack_z(z_history[idx])
        P = compute_catheter_points(
            mesh=mesh,
            rigid=rigid,
            x_nodes=x_nodes_i[0],
            f_ext_rigid=f_ext_rigid,
            tau_ext_rigid=tau_vec,
            n_samples_rigid=20,
        )
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=(0.2, 0.4, 0.8, a), linewidth=2)

    x_nodes_first, _, _ = solver.unpack_z(z_history[idx_pick[0]])
    x_nodes_last, _, _ = solver.unpack_z(z_history[idx_pick[-1]])
    P_first = compute_catheter_points(mesh, rigid, x_nodes_first[0], f_ext_rigid, tau_vec, 20)
    P_last = compute_catheter_points(mesh, rigid, x_nodes_last[0], f_ext_rigid, tau_vec, 20)
    ax.scatter(P_first[-1, 0], P_first[-1, 1], P_first[-1, 2], c='red', s=40, label='start tip')
    ax.scatter(P_last[-1, 0], P_last[-1, 1], P_last[-1, 2], c='green', s=40, label='final tip')

    ax.set_title('LM convergence under a single torque (|tau|=15, dir=x)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([0, 0.15])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lm_convergence_demo()
