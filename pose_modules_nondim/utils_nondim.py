# utils_nondim.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List
from .rod_mesh_nondim import RodMesh
from .segments_nondim import FlexibleSegment, RigidSegment
from .external_wrench_nondim import GravityLineDensity, GravityRigid
from .nondim import NondimScales, x_dim_to_bar, x_bar_to_dim


def make_C_S_flexible(env_constraint: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    def C_S_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        Qn = x_n[3:7]
        Qnp1 = x_np1[3:7]
        quat_res_n = np.array([np.dot(Qn, Qn) - 1.])
        quat_res_np1 = np.array([np.dot(Qnp1, Qnp1) - 1.])
        res_list = [quat_res_n, quat_res_np1]
        if env_constraint is not None:
            res_list.append(env_constraint(x_n))
            res_list.append(env_constraint(x_np1))
        return np.concatenate(res_list)
    return C_S_fun


def make_C_BV_proximal_pose(p0_target: np.ndarray, Q0_target: np.ndarray):
    def C_BV_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        p = x_n[:3]
        Q = x_n[3:7]
        res_p = p - p0_target
        res_Q = Q - Q0_target
        return np.concatenate([res_p, res_Q])
    return C_BV_fun


def make_C_BV_distal_free_tip():
    def C_BV_fun(x_n: np.ndarray, x_np1: np.ndarray) -> np.ndarray:
        f = x_np1[7:10]
        tau = x_np1[10:13]
        return np.concatenate([f, tau])
    return C_BV_fun


def make_initial_guess_multi_bar(
        flex_segs: List[FlexibleSegment],
        meshes: List[RodMesh],
        rigid_segs: List[RigidSegment],
        scales: NondimScales,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Build nondimensional initial guess z0_bar.
    Strategy: build SI straight catheter then scale to bar.
    k_array initialized to zeros (already dimensionless in our GL6 setup).
    """
    from .equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondim

    N = len(flex_segs)
    assert N == len(meshes) == len(rigid_segs)

    x_nodes_list_bar: List[np.ndarray] = []
    k_array_list_bar: List[np.ndarray] = []
    x_rigid_list_bar: List[np.ndarray] = []

    z_base = 0.0
    Q0 = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(N):
        flex = flex_segs[i]
        mesh = meshes[i]
        rigid = rigid_segs[i]

        M = mesh.M
        sigmas = mesh.sigma_nodes

        # build SI nodes
        x_nodes_dim = np.zeros((M + 1, 13))
        for n in range(M + 1):
            p = np.array([0.0, 0.0, z_base + float(sigmas[n])])
            f = np.zeros(3)
            tau = np.zeros(3)
            x_nodes_dim[n] = np.concatenate([p, Q0, f, tau])

        x_nodes_bar = np.vstack([x_dim_to_bar(x_nodes_dim[n], scales) for n in range(M + 1)])
        x_nodes_list_bar.append(x_nodes_bar)

        z_base += float(flex.length)

        # rigid end in SI then scale
        p_rigid_end = np.array([0.0, 0.0, z_base + float(rigid.length)])
        xR_dim = np.concatenate([p_rigid_end, Q0, np.zeros(3), np.zeros(3)])
        x_rigid_list_bar.append(x_dim_to_bar(xR_dim, scales))

        z_base += float(rigid.length)

        k_array_list_bar.append(np.zeros((M, 3, 13)))

    dummy = MultiSegmentEquilibriumSolverNondim(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=np.array([0.0, 0.0, 0.0]),
        Q0_target=Q0,
        scales=scales,
    )
    z0_bar = dummy.pack_z(x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar)
    return z0_bar, x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar


def plot_catheter_3d_bar(
        mesh: RodMesh,
        rigid: RigidSegment,
        x_nodes_bar: np.ndarray,      # (M+1,13) in bar
        x_rigid_end_bar: np.ndarray,  # (13,) in bar
        f_ext_rigid_dim: np.ndarray,
        tau_ext_rigid_dim: np.ndarray,
        scales: NondimScales,
        n_samples_rigid: int = 10,
):
    """
    Plot configuration given nondimensional solution.
    We convert p_bar back to meters for plotting and for rigid sampling.
    """
    # flexible positions in meters
    p_flex = x_nodes_bar[:, :3] * scales.L_ref

    # rigid sampling uses SI state
    x_flex_end_dim = x_bar_to_dim(x_nodes_bar[-1], scales)
    Lr = rigid.length
    sigmas_rigid = np.linspace(0.0, Lr, n_samples_rigid + 1)

    p_rigid_list = []
    for s in sigmas_rigid[1:]:
        x_s = rigid.state_along(
            x_proximal=x_flex_end_dim,
            sigma=float(s),
            f_ext_total=f_ext_rigid_dim,
            tau_ext_total=tau_ext_rigid_dim,
        )
        p_rigid_list.append(x_s[:3])

    p_rigid = np.array(p_rigid_list) if p_rigid_list else np.zeros((0, 3))
    P = np.vstack([p_flex, p_rigid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker='o')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Catheter configuration (nondim solution restored)')
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
