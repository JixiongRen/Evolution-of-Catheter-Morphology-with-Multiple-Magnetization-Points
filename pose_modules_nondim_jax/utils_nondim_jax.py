# utils_nondim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Sequence, Any
from .external_wrench_nondim_jax import GravityLineDensity, GravityRigid
import jax.numpy as jnp

Array = jnp.ndarray

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


# ---------------------------------------------------------------------
# Minimal scales (match nondim_jax.NondimScales fields)
# If you already have nondim_jax.NondimScales, you can import it instead.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class NondimScalesJAX:
    L_ref: float
    F_ref: float
    M_ref: float


def x_dim_to_bar_jax(x_dim: Array, s: NondimScalesJAX) -> Array:
    """(13,) SI -> nondim bar."""
    x_dim = jnp.asarray(x_dim, dtype=jnp.float64).reshape(13,)
    p = x_dim[0:3] / s.L_ref
    Q = x_dim[3:7]
    f = x_dim[7:10] / s.F_ref
    tau = x_dim[10:13] / s.M_ref
    return jnp.concatenate([p, Q, f, tau], axis=0)


def x_bar_to_dim_jax(x_bar: Array, s: NondimScalesJAX) -> Array:
    """(13,) bar -> SI."""
    x_bar = jnp.asarray(x_bar, dtype=jnp.float64).reshape(13,)
    p = x_bar[0:3] * s.L_ref
    Q = x_bar[3:7]
    f = x_bar[7:10] * s.F_ref
    tau = x_bar[10:13] * s.M_ref
    return jnp.concatenate([p, Q, f, tau], axis=0)


# ---------------------------------------------------------------------
# Constraints builders (JAX-traceable)
# ---------------------------------------------------------------------
def make_C_S_flexible_jax(
    env_constraint: Optional[Callable[[Array], Array]] = None
) -> Callable[[Array, Array], Array]:
    """
    Same semantics as numpy version:
    - returns residuals enforcing quaternion norm at both nodes,
      plus optional env constraint at both nodes.
    """
    def C_S_fun(x_n: Array, x_np1: Array) -> Array:
        x_n = jnp.asarray(x_n, dtype=jnp.float64).reshape(13,)
        x_np1 = jnp.asarray(x_np1, dtype=jnp.float64).reshape(13,)
        Qn = x_n[3:7]
        Qnp1 = x_np1[3:7]
        quat_res_n = jnp.array([jnp.dot(Qn, Qn) - 1.0], dtype=jnp.float64)
        quat_res_np1 = jnp.array([jnp.dot(Qnp1, Qnp1) - 1.0], dtype=jnp.float64)
        res_list = [quat_res_n, quat_res_np1]
        if env_constraint is not None:
            res_list.append(jnp.asarray(env_constraint(x_n), dtype=jnp.float64).reshape(-1,))
            res_list.append(jnp.asarray(env_constraint(x_np1), dtype=jnp.float64).reshape(-1,))
        return jnp.concatenate(res_list, axis=0)
    return C_S_fun


def make_C_BV_proximal_pose_jax(
    p0_target_bar: Array,
    Q0_target: Array,
) -> Callable[[Array, Array], Array]:
    """C_BV at segment proximal: enforce x_n pose equals target pose (bar)."""
    p0_target_bar = jnp.asarray(p0_target_bar, dtype=jnp.float64).reshape(3,)
    Q0_target = jnp.asarray(Q0_target, dtype=jnp.float64).reshape(4,)

    def C_BV_fun(x_n: Array, x_np1: Array) -> Array:
        x_n = jnp.asarray(x_n, dtype=jnp.float64).reshape(13,)
        p = x_n[:3]
        Q = x_n[3:7]
        return jnp.concatenate([p - p0_target_bar, Q - Q0_target], axis=0)

    return C_BV_fun


def make_C_BV_distal_free_tip_jax() -> Callable[[Array, Array], Array]:
    """C_BV at distal tip: enforce free tip (f=tau=0) using x_np1 in bar."""
    def C_BV_fun(x_n: Array, x_np1: Array) -> Array:
        x_np1 = jnp.asarray(x_np1, dtype=jnp.float64).reshape(13,)
        f = x_np1[7:10]
        tau = x_np1[10:13]
        return jnp.concatenate([f, tau], axis=0)
    return C_BV_fun


# ---------------------------------------------------------------------
# Packing / unpacking (replace dependence on numpy MultiSegmentEquilibriumSolverNondim)
# Layout MUST match your current solver:
#   For each segment i:
#     nodes: (M+1)*13
#     k:     M*39   (flattened from (M,3,13) row-major)
#     rigid: 13
# ---------------------------------------------------------------------
def pack_z_bar_jax(
    x_nodes_list_bar: Sequence[Array],
    k_array_list_bar: Sequence[Array],
    x_rigid_list_bar: Sequence[Array],
) -> Array:
    parts: List[Array] = []
    N = len(x_nodes_list_bar)
    assert N == len(k_array_list_bar) == len(x_rigid_list_bar)

    for i in range(N):
        x_nodes = jnp.asarray(x_nodes_list_bar[i], dtype=jnp.float64)  # (M+1,13)
        k_arr = jnp.asarray(k_array_list_bar[i], dtype=jnp.float64)    # (M,3,13) or (M,39)
        xR = jnp.asarray(x_rigid_list_bar[i], dtype=jnp.float64).reshape(13,)

        if k_arr.ndim == 3:
            # (M,3,13) -> (M,39)
            k_flat = k_arr.reshape((k_arr.shape[0], 39))
        elif k_arr.ndim == 2 and k_arr.shape[1] == 39:
            k_flat = k_arr
        else:
            raise ValueError(f"k_array_list_bar[{i}] must be (M,3,13) or (M,39), got {k_arr.shape}")

        parts.append(x_nodes.reshape(-1))
        parts.append(k_flat.reshape(-1))
        parts.append(xR.reshape(-1))

    return jnp.concatenate(parts, axis=0)


def unpack_z_bar_jax(
    z_bar: Array,
    M_list: Sequence[int],
) -> Tuple[List[Array], List[Array], List[Array]]:
    """
    Returns:
      x_nodes_list_bar: list of (M+1,13)
      k_array_list_bar: list of (M,39)
      x_rigid_list_bar: list of (13,)
    """
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64).reshape(-1,)
    x_nodes_list_bar: List[Array] = []
    k_array_list_bar: List[Array] = []
    x_rigid_list_bar: List[Array] = []

    idx = 0
    for M in M_list:
        n_nodes = (M + 1) * 13
        x_nodes = z_bar[idx:idx + n_nodes].reshape((M + 1, 13))
        idx += n_nodes

        n_k = M * 39
        k_flat = z_bar[idx:idx + n_k].reshape((M, 39))
        idx += n_k

        xR = z_bar[idx:idx + 13].reshape((13,))
        idx += 13

        x_nodes_list_bar.append(x_nodes)
        k_array_list_bar.append(k_flat)
        x_rigid_list_bar.append(xR)

    if idx != z_bar.shape[0]:
        raise ValueError(f"Unpack consumed {idx} entries, but z_bar has {z_bar.shape[0]} entries.")
    return x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar


# ---------------------------------------------------------------------
# Initial guess (JAX, no numpy solver import)
# Accepts minimal fields: mesh.M and mesh.sigma_nodes; seg.length; rigid.length
# ---------------------------------------------------------------------
def make_initial_guess_multi_bar_jax(
    flex_segs: Sequence[Any],
    meshes: Sequence[Any],
    rigid_segs: Sequence[Any],
    scales: NondimScalesJAX,
) -> Tuple[Array, List[Array], List[Array], List[Array]]:
    """
    Build nondimensional initial guess z0_bar (JAX).
    Strategy: straight catheter in SI, then scale to bar.
    k_array initialized to zeros (dimensionless GL6 setup).
    """
    N = len(flex_segs)
    assert N == len(meshes) == len(rigid_segs)

    x_nodes_list_bar: List[Array] = []
    k_array_list_bar: List[Array] = []
    x_rigid_list_bar: List[Array] = []

    z_base = 0.0
    Q0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    for i in range(N):
        flex = flex_segs[i]
        mesh = meshes[i]
        rigid = rigid_segs[i]

        M = int(mesh.M)
        sigmas = jnp.asarray(mesh.sigma_nodes, dtype=jnp.float64).reshape((M + 1,))

        # SI nodes (M+1, 13)
        def node_dim(n):
            p = jnp.array([0.0, 0.0, z_base + float(sigmas[n])], dtype=jnp.float64)
            f = jnp.zeros((3,), dtype=jnp.float64)
            tau = jnp.zeros((3,), dtype=jnp.float64)
            return jnp.concatenate([p, Q0, f, tau], axis=0)

        x_nodes_dim = jnp.stack([node_dim(n) for n in range(M + 1)], axis=0)  # (M+1,13)
        x_nodes_bar = jnp.stack([x_dim_to_bar_jax(x_nodes_dim[n], scales) for n in range(M + 1)], axis=0)
        x_nodes_list_bar.append(x_nodes_bar)

        z_base += float(flex.length)

        # rigid distal end in SI then scale
        p_rigid_end = jnp.array([0.0, 0.0, z_base + float(rigid.length)], dtype=jnp.float64)
        xR_dim = jnp.concatenate([p_rigid_end, Q0, jnp.zeros((3,), dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64)], axis=0)
        x_rigid_list_bar.append(x_dim_to_bar_jax(xR_dim, scales))

        z_base += float(rigid.length)

        # k: (M,3,13) zeros -> packer will reshape to (M,39)
        k_array_list_bar.append(jnp.zeros((M, 3, 13), dtype=jnp.float64))

    z0_bar = pack_z_bar_jax(x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar)
    return z0_bar, x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar


# ---------------------------------------------------------------------
# Plot helper (kept similar to numpy version; uses numpy+matplotlib)
# This still relies on rigid.state_along(...) being available (numpy rigid segment).
# If you go full JAX-only visualization later, we can add a JAX rigid sampler.
# ---------------------------------------------------------------------
def plot_catheter_3d_bar_jax(
    mesh: Any,
    rigid: Any,
    x_nodes_bar: Array,      # (M+1,13) bar
    x_rigid_end_bar: Array,  # (13,) bar (not used here but kept for signature compatibility)
    f_ext_rigid_dim: np.ndarray,
    tau_ext_rigid_dim: np.ndarray,
    scales: NondimScalesJAX,
    n_samples_rigid: int = 10,
):
    """
    Plot configuration given nondimensional solution.
    Convert p_bar back to meters for plotting and for rigid sampling.

    Note: rigid sampling uses rigid.state_along in SI (numpy implementation).
    """
    x_nodes_bar = jnp.asarray(x_nodes_bar, dtype=jnp.float64)

    # flexible positions in meters
    p_flex = np.asarray(x_nodes_bar[:, :3] * scales.L_ref)

    # rigid sampling uses SI state
    x_flex_end_dim = np.asarray(x_bar_to_dim_jax(x_nodes_bar[-1], scales))
    Lr = float(rigid.length)
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
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], marker="o")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Catheter configuration (nondim solution restored) [JAX utils]")
    ax.view_init(elev=30, azim=-60)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()



def build_k_matrices_for_pdms_jax(d_outer: float) -> tuple[Array, Array]:
    """
    PDMS flexible segment stiffness matrices (Cosserat rod), JAX version.

    Parameters
    ----------
    d_outer : float
        Outer diameter [m]

    Assumptions
    -----------
    E = 1.8 MPa
    G = 0.6 MPa
    Circular cross-section
    """
    E = 1.8e6  # Pa
    G = 0.6e6  # Pa

    d_outer = jnp.asarray(d_outer, dtype=jnp.float64)
    r = d_outer / 2.0
    A = jnp.pi * r**2

    Ixx = Iyy = jnp.pi * r**4 / 4.0
    J = jnp.pi * r**4 / 2.0

    K_se = jnp.diag(jnp.array([G * A, G * A, E * A], dtype=jnp.float64))
    K_bt = jnp.diag(jnp.array([E * Ixx, E * Iyy, G * J], dtype=jnp.float64))
    return K_se, K_bt


def build_gravity_line_density_for_pdms_jax(d_outer: float) -> GravityLineDensity:
    """
    PDMS flexible segment gravity line density (WORLD frame), JAX version.

    Assumptions
    -----------
    rho = 970 kg/m^3
    g_world = [0, 0, -9.81] m/s^2
    force line density = rho * A * g_world  [N/m]
    """
    rho = 970.0  # kg/m^3
    g_world = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64)

    d_outer = jnp.asarray(d_outer, dtype=jnp.float64)
    r = d_outer / 2.0
    A = jnp.pi * r**2               # m^2
    line_mass = rho * A             # kg/m
    f_line_world = line_mass * g_world  # N/m (WORLD)

    return GravityLineDensity(f_line_world=f_line_world)


def build_gravity_rigid_for_ndfeb_jax(d_outer: float, length: float) -> GravityRigid:
    """
    NdFeB rigid segment gravity (lumped), JAX version.

    Assumptions
    -----------
    rho = 7500 kg/m^3
    g_world = [0, 0, -9.81] m/s^2
    r_cm_body = [0, 0, L/2] (about proximal, in BODY frame)
    """
    rho = 7500.0  # kg/m^3
    g_world = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64)

    d_outer = jnp.asarray(d_outer, dtype=jnp.float64)
    length = jnp.asarray(length, dtype=jnp.float64)

    r = d_outer / 2.0
    A = jnp.pi * r**2
    V = A * length
    mass = rho * V

    r_cm_body = jnp.array([0.0, 0.0, 0.5], dtype=jnp.float64) * length

    # GravityRigid.mass 在我给你的 JAX 外载实现里是 float；这里用 float(mass) 固化为 Python float
    # 这样更利于作为“静态参数”参与 JIT（避免 mass 变成 traced array）。
    return GravityRigid(mass=float(mass), g_world=g_world, r_cm_body=r_cm_body)

