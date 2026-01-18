# utils_nondim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Sequence, Any

from nondim import NondimScales
from basics_nondim import quat_normalize, quat_to_rotmat
from external_wrench_nondim import GravityLineDensity, GravityRigid

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray



# ---------------------------------------------------------------------
# Nondimensional transforms (match nondim_jax.NondimScales)
# ---------------------------------------------------------------------
def x_dim_to_bar_jax(x_dim: Array, s: NondimScales) -> Array:
    """(13,) SI -> nondim bar."""
    x_dim = jnp.asarray(x_dim, dtype=jnp.float64).reshape(13,)
    p = x_dim[0:3] / float(s.L_ref)
    Q = x_dim[3:7]
    f = x_dim[7:10] / float(s.F_ref)
    tau = x_dim[10:13] / float(s.M_ref)
    return jnp.concatenate([p, Q, f, tau], axis=0)


def x_bar_to_dim_jax(x_bar: Array, s: NondimScales) -> Array:
    """(13,) bar -> SI."""
    x_bar = jnp.asarray(x_bar, dtype=jnp.float64).reshape(13,)
    p = x_bar[0:3] * float(s.L_ref)
    Q = x_bar[3:7]
    f = x_bar[7:10] * float(s.F_ref)
    tau = x_bar[10:13] * float(s.M_ref)
    return jnp.concatenate([p, Q, f, tau], axis=0)


def _unit3(v: Array, eps: float = 1e-12) -> Array:
    v = jnp.asarray(v, dtype=jnp.float64).reshape(3,)
    n = jnp.linalg.norm(v)
    return v / jnp.maximum(n, eps)


# ---------------------------------------------------------------------
# Packing / unpacking (layout MUST match equilibrium_solver_nondim_jax.residual_bar)
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
    if not (N == len(k_array_list_bar) == len(x_rigid_list_bar)):
        raise ValueError("pack_z_bar_jax: input lists length mismatch")

    for i in range(N):
        x_nodes = jnp.asarray(x_nodes_list_bar[i], dtype=jnp.float64)  # (M+1,13)
        k_arr = jnp.asarray(k_array_list_bar[i], dtype=jnp.float64)    # (M,3,13) or (M,39)
        xR = jnp.asarray(x_rigid_list_bar[i], dtype=jnp.float64).reshape(13,)

        if k_arr.ndim == 3 and k_arr.shape[1:] == (3, 13):
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
    """Unpack z_bar to (nodes, k_flat, rigid_end) per segment."""
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64).reshape(-1,)
    x_nodes_list_bar: List[Array] = []
    k_array_list_bar: List[Array] = []
    x_rigid_list_bar: List[Array] = []

    idx = 0
    for M in M_list:
        M = int(M)
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
# Initial guess (JAX; supports arbitrary base pose p0/Q0 and axis direction)
# ---------------------------------------------------------------------
def make_initial_guess_multi_bar_jax(
    flex_segs: Sequence[Any],
    meshes: Sequence[Any],
    rigid_segs: Sequence[Any],
    scales: NondimScales,
    *,
    p0_dim: Optional[Array] = None,
    Q0: Optional[Array] = None,
    axis_body: Optional[Array] = None,
) -> Tuple[Array, List[Array], List[Array], List[Array]]:
    """Build nondimensional initial guess z0_bar.

    Strategy
    --------
    - Choose base state (p0_dim, Q0) at the proximal of seg0.
    - For each flexible segment: place nodes along the world direction (R(Q0) @ axis_body) with cumulative arclength.
    - For each rigid segment: place a distal-end state continuing in the same direction.

    Notes
    -----
    - This is a *geometric* initial guess; internal force/moment are initialized to 0.
    - k_array is initialized to 0 (size M×3×13), and will be reshaped to (M,39) by packer.
    """
    N = len(flex_segs)
    if not (N == len(meshes) == len(rigid_segs)):
        raise ValueError("make_initial_guess_multi_bar_jax: segment/mesh length mismatch")

    p0_dim = jnp.zeros((3,), dtype=jnp.float64) if p0_dim is None else jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3,)
    Q0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64) if Q0 is None else quat_normalize(jnp.asarray(Q0, dtype=jnp.float64).reshape(4,))
    axis_body = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64) if axis_body is None else _unit3(axis_body)

    R0 = quat_to_rotmat(Q0)
    t_world = R0 @ axis_body  # catheter centerline direction in WORLD frame

    x_nodes_list_bar: List[Array] = []
    k_array_list_bar: List[Array] = []
    x_rigid_list_bar: List[Array] = []

    s_base = 0.0  # cumulative arclength in SI along catheter axis

    for i in range(N):
        mesh = meshes[i]
        rigid = rigid_segs[i]

        M = int(mesh.M)
        sigmas = jnp.asarray(mesh.sigma_nodes, dtype=jnp.float64).reshape((M + 1,))

        def node_dim(n: int) -> Array:
            p = p0_dim + (s_base + sigmas[n]) * t_world
            f = jnp.zeros((3,), dtype=jnp.float64)
            tau = jnp.zeros((3,), dtype=jnp.float64)
            return jnp.concatenate([p, Q0, f, tau], axis=0)

        x_nodes_dim = jnp.stack([node_dim(n) for n in range(M + 1)], axis=0)  # (M+1,13)
        x_nodes_bar = jnp.stack([x_dim_to_bar_jax(x_nodes_dim[n], scales) for n in range(M + 1)], axis=0)
        x_nodes_list_bar.append(x_nodes_bar)

        # advance by flexible length
        s_base += float(flex_segs[i].length)

        # rigid distal end in SI then scale
        p_rigid_end = p0_dim + (s_base + float(rigid.length)) * t_world
        xR_dim = jnp.concatenate([p_rigid_end, Q0, jnp.zeros((3,), dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64)], axis=0)
        x_rigid_list_bar.append(x_dim_to_bar_jax(xR_dim, scales))

        # advance by rigid length
        s_base += float(rigid.length)

        # k: (M,3,13) zeros
        k_array_list_bar.append(jnp.zeros((M, 3, 13), dtype=jnp.float64))

    z0_bar = pack_z_bar_jax(x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar)
    return z0_bar, x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar


# ---------------------------------------------------------------------
# Generic circular-section stiffness / gravity helpers (SI)
# ---------------------------------------------------------------------
def build_k_matrices_circular_jax(*, d_outer: float, E: float, G: float) -> tuple[Array, Array]:
    """Circular-section K_se, K_bt in SI for Cosserat rod.

    K_se = diag(GA, GA, EA)
    K_bt = diag(EI, EI, GJ)
    """
    d_outer = float(d_outer)
    E = float(E)
    G = float(G)
    r = d_outer / 2.0
    A = jnp.pi * (r ** 2)
    I = jnp.pi * (r ** 4) / 4.0
    J = jnp.pi * (r ** 4) / 2.0
    K_se = jnp.diag(jnp.array([G * A, G * A, E * A], dtype=jnp.float64))
    K_bt = jnp.diag(jnp.array([E * I, E * I, G * J], dtype=jnp.float64))
    return K_se, K_bt


def build_gravity_line_density_jax(*, rho: float, d_outer: float, g_world: Array) -> GravityLineDensity:
    """Gravity line density [N/m] in WORLD frame for a solid circular section."""
    rho = float(rho)
    d_outer = float(d_outer)
    g_world = jnp.asarray(g_world, dtype=jnp.float64).reshape(3,)
    r = d_outer / 2.0
    A = jnp.pi * (r ** 2)
    line_mass = rho * A  # kg/m
    f_line_world = line_mass * g_world
    return GravityLineDensity(f_line_world=f_line_world)


def build_gravity_rigid_jax(
    *,
    rho: float,
    d_outer: float,
    length: float,
    g_world: Array,
    v_star_body: Optional[Array] = None,
) -> GravityRigid:
    """Rigid-segment gravity (lumped) about proximal point.

    COM offset is assumed at half-length along v_star_body (body frame).
    """
    rho = float(rho)
    d_outer = float(d_outer)
    length = float(length)
    g_world = jnp.asarray(g_world, dtype=jnp.float64).reshape(3,)

    v_star_body = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64) if v_star_body is None else _unit3(v_star_body)

    r = d_outer / 2.0
    A = jnp.pi * (r ** 2)
    V = A * length
    mass = rho * V

    r_cm_body = 0.5 * length * v_star_body
    return GravityRigid(mass=float(mass), g_world=g_world, r_cm_body=r_cm_body)


# ---------------------------------------------------------------------
# Backward-compatible convenience wrappers (defaults)
# ---------------------------------------------------------------------
def build_k_matrices_for_pdms_jax(d_outer: float) -> tuple[Array, Array]:
    return build_k_matrices_circular_jax(d_outer=float(d_outer), E=1.8e6, G=0.6e6)


def build_gravity_line_density_for_pdms_jax(d_outer: float) -> GravityLineDensity:
    # PDMS defaults: rho=970, gravity along -Z
    return build_gravity_line_density_jax(rho=970.0, d_outer=float(d_outer), g_world=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64))


def build_gravity_rigid_for_ndfeb_jax(d_outer: float, length: float) -> GravityRigid:
    # NdFeB defaults: rho=7500, gravity along -Z, COM at half-length along +Z body axis.
    return build_gravity_rigid_jax(rho=7500.0, d_outer=float(d_outer), length=float(length), g_world=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float64), v_star_body=jnp.array([0.0,0.0,1.0], dtype=jnp.float64))


# ---------------------------------------------------------------------
# Plot helper (kept for convenience)
# ---------------------------------------------------------------------
def plot_catheter_3d_bar_jax(
    mesh: Any,
    rigid: Any,
    x_nodes_bar: Array,      # (M+1,13) bar
    x_rigid_end_bar: Array,  # (13,) bar (not used here but kept for signature compatibility)
    f_ext_rigid_dim: np.ndarray,
    tau_ext_rigid_dim: np.ndarray,
    scales: NondimScales,
    n_samples_rigid: int = 10,
):
    """Plot configuration given nondimensional solution (meters)."""
    x_nodes_bar = jnp.asarray(x_nodes_bar, dtype=jnp.float64)
    p_flex = np.asarray(x_nodes_bar[:, :3] * float(scales.L_ref))

    # rigid sampling uses SI state (requires rigid.state_along from your numpy objects if used)
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
    P = np.vstack([p_flex, p_rigid]) if p_rigid.size else p_flex

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
