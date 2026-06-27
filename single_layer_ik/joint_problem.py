from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from .jax_config import configure_jax_enable_x64

from .forward_kinematics_optimized_bak.advancer_nondim import apply_advancer_protrude_length
from .forward_kinematics_optimized_bak.equilibrium_solver_nondim import (
    SolverParams,
    residual_bar,
    with_coil_currents,
    with_L1_dim,
)
from .forward_kinematics_optimized_bak.fk import build_solver_params, compute_scales_from_flex
from .forward_kinematics_optimized_bak.nondim import NondimScales
from .forward_kinematics_optimized_bak.utils_nondim import make_initial_guess_multi_bar_jax, unpack_z_bar_jax

configure_jax_enable_x64()
Array = jnp.ndarray


@dataclass(frozen=True)
class StaticFK:
    params_base: SolverParams
    meshes_ref: tuple
    scales: NondimScales
    z_len: int
    L_fixed: float
    flex_lengths_tail: Tuple[float, ...]
    rigid_lengths: Tuple[float, ...]
    M_list: Tuple[int, ...]
    p0_dim: Array
    Q0_wxyz: Array
    axis_body: Array
    L1_min: float


def compute_z_len_from_M_list(M_list: Sequence[int]) -> int:
    """Compute packed z length for the residual layout.

    MUST match `equilibrium_solver_nondim.residual_bar` / `utils_nondim.unpack_z_bar_jax` layout:
      For each segment i with Mi intervals:
        - nodes: (Mi+1) * 13
        - k-array: Mi * 39
        - rigid lumped state: 13
    """
    total = 0
    for m in M_list:
        Mi = int(m)
        total += (Mi + 1) * 13
        total += Mi * 39
        total += 13
    return int(total)


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))


def u_to_Ix(
    u: Array,
    *,
    I_max: float,
    x_min: float,
    x_max: float,
    I_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Map unconstrained u -> (I, x) with box constraints.

    u layout:
      - u[0:8]: currents params
      - u[8]: insertion param

    I_scale > 1 amplifies the Jacobian dI/du near I=0, making the
    LM solver more willing to change coil currents.  Physically the
    mapping is the same; only the curvature of the parameterisation
    changes.
    """
    u = jnp.asarray(u, dtype=jnp.float64).reshape(-1,)

    uI = u[0:8]
    ux = u[8]

    I = jnp.asarray(I_max, dtype=jnp.float64) * jnp.tanh(jnp.asarray(I_scale, dtype=jnp.float64) * uI)
    x = jnp.asarray(x_min, dtype=jnp.float64) + (jnp.asarray(x_max, dtype=jnp.float64) - jnp.asarray(x_min, dtype=jnp.float64)) * _sigmoid(ux)
    return jnp.asarray(I, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64).reshape(())


def Ix_to_u(
    I: Array,
    x: float,
    *,
    I_max: float,
    x_min: float,
    x_max: float,
    I_scale: float = 1.0,
    eps: float = 1e-12,
) -> Array:
    """Inverse mapping for initialization.

    - I: clipped to (-I_max, I_max)
    - x: clipped to (x_min, x_max)
    """
    I = jnp.asarray(I, dtype=jnp.float64).reshape(-1,)
    if I.size != 8:
        raise ValueError(f"I must have size 8, got {int(I.size)}")

    I_clip = jnp.clip(I / float(I_max), -1.0 + eps, 1.0 - eps)
    uI = jnp.arctanh(I_clip) / jnp.asarray(max(I_scale, eps), dtype=jnp.float64)

    x_clip = min(max(float(x), float(x_min) + eps), float(x_max) - eps)
    t = (x_clip - float(x_min)) / max(float(x_max) - float(x_min), eps)
    t = min(max(float(t), eps), 1.0 - eps)
    ux = jnp.log(t / (1.0 - t))

    return jnp.concatenate([uI, jnp.asarray([ux], dtype=jnp.float64)], axis=0)


def build_static_fk(
    *,
    flex_lengths_tail: Sequence[float],
    rigid_lengths: Sequence[float],
    M_list: Sequence[int],
    L_protrude_max: float,
    L1_min: float = 1e-6,
    # materials
    flex_d_outer: Sequence[float] = (0.0015,),
    flex_E: Sequence[float] = (1.8e6,),
    flex_G: Sequence[float] = (0.6e6,),
    flex_rho: Sequence[float] = (970.0,),
    rigid_d_outer: Sequence[float] = (0.0015,),
    rigid_rho: Sequence[float] = (7500.0,),
    # base pose
    p0_dim: Sequence[float] = (0.0, 0.0, -0.05),
    Q0_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    axis_body: Sequence[float] = (0.0, 0.0, 1.0),
    # environment
    enable_gravity: bool = False,
    g_world: Sequence[float] = (0.0, 0.0, -9.81),
    # magnetics
    enable_magnetics: bool = False,
    calib_file: Optional[str] = None,
    actuation_table_pkl: Optional[str] = None,
    m_body_list: Optional[Sequence[Sequence[float]]] = None,
) -> StaticFK:
    """Build a static FK configuration for the single-layer joint solver.

    This mirrors `ForwardKinematicsEngine._ensure_static_initialized()` semantics:
    - Choose a fixed nondimensional scale using L_ref = L_protrude_max.
    - Build a reference params_base at L_protrude_max, then update (I, L1_dim) per-iteration.

    Important:
    - M_list must be fixed across iterations.
    - params PyTree leaf structure should remain stable for JAX.
    """
    rigid_lengths_t = tuple(float(x) for x in rigid_lengths)
    M_list_t = tuple(int(m) for m in M_list)

    N = len(rigid_lengths_t)
    if N <= 0:
        raise ValueError("rigid_lengths is empty")

    flex_lengths_tail_t = tuple(float(x) for x in flex_lengths_tail)
    if len(flex_lengths_tail_t) == N - 1:
        tail = list(flex_lengths_tail_t)
    elif len(flex_lengths_tail_t) == N:
        tail = list(flex_lengths_tail_t[1:])
    else:
        raise ValueError(
            f"flex_lengths_tail must have N-1 or N values where N={N}; got {len(flex_lengths_tail_t)}"
        )

    L_fixed = float(sum(tail) + sum(rigid_lengths_t))
    L1_ref = float(L_protrude_max) - L_fixed
    if L1_ref < float(L1_min):
        raise ValueError(
            f"Infeasible L_protrude_max={float(L_protrude_max):.6g}: inferred L1_ref={L1_ref:.6g} < L1_min={float(L1_min):.6g}"
        )

    flex_lengths_ref = apply_advancer_protrude_length(
        flex_lengths_in=[L1_ref] + tail,
        rigid_lengths=list(rigid_lengths_t),
        L_protrude=float(L_protrude_max),
        L1_min=float(L1_min),
    )

    # Fixed scales (important for compilation stability)
    scales_ref = compute_scales_from_flex(
        L_ref=float(L_protrude_max),
        d_outer=float(flex_d_outer[0]),
        E=float(flex_E[0]),
        G=float(flex_G[0]),
    )

    p0_dim_a = jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3,)
    Q0_a = jnp.asarray(Q0_wxyz, dtype=jnp.float64).reshape(4,)
    axis_a = jnp.asarray(axis_body, dtype=jnp.float64).reshape(3,)

    params_base, meshes_ref = build_solver_params(
        flex_lengths=list(flex_lengths_ref),
        rigid_lengths=list(rigid_lengths_t),
        M_list=list(M_list_t),
        flex_d_outer=list(flex_d_outer),
        flex_E=list(flex_E),
        flex_G=list(flex_G),
        flex_rho=list(flex_rho),
        rigid_d_outer=list(rigid_d_outer),
        rigid_rho=list(rigid_rho),
        scales=scales_ref,
        p0_dim=p0_dim_a,
        Q0=Q0_a,
        axis_body=axis_a,
        enable_gravity=bool(enable_gravity),
        g_world=jnp.asarray(g_world, dtype=jnp.float64).reshape(3,),
        enable_magnetics=bool(enable_magnetics),
        calib_file=calib_file,
        actuation_table_pkl=actuation_table_pkl,
        coil_currents=jnp.zeros((8,), dtype=jnp.float64),
        m_body_list=m_body_list,
    )

    z_len = compute_z_len_from_M_list(params_base.M_list)

    return StaticFK(
        params_base=params_base,
        meshes_ref=tuple(meshes_ref),
        scales=scales_ref,
        z_len=int(z_len),
        L_fixed=float(L_fixed),
        flex_lengths_tail=tuple(float(x) for x in flex_lengths_tail_t),
        rigid_lengths=rigid_lengths_t,
        M_list=M_list_t,
        p0_dim=p0_dim_a,
        Q0_wxyz=Q0_a,
        axis_body=axis_a,
        L1_min=float(L1_min),
    )


def extract_tip_p_dim(*, z_bar: Array, scales: NondimScales, M_list: Sequence[int]) -> Array:
    """Extract tip position (SI meters) from packed `z_bar`.

    This mirrors `ForwardKinematicsEngine._extract_tip_pose_dim`, but does not depend on engine state.
    """
    z_bar = jnp.asarray(z_bar, dtype=jnp.float64).reshape(-1,)
    offset = 0
    tip_idx = 0
    for m in M_list:
        Mi = int(m)
        tip_idx = offset + Mi * 13
        offset += (Mi + 1) * 13
        offset += Mi * 39
        offset += 13
    idx = jnp.asarray([tip_idx + 0, tip_idx + 1, tip_idx + 2], dtype=jnp.int32)
    return jnp.asarray(z_bar[idx] * float(scales.L_ref), dtype=jnp.float64)


def extract_distal_tip_p_dim(*, z_bar: Array, scales: NondimScales, M_list: Sequence[int]) -> Array:
    """Extract the distal catheter tip, including the last rigid/magnet segment."""
    centerline = extract_centerline_p_dim(
        z_bar=z_bar,
        scales=scales,
        M_list=M_list,
        include_rigid=True,
        rigid_sample_count=0,
    )
    return jnp.asarray(centerline[-1, :], dtype=jnp.float64)


def extract_task_tip_p_dim(
    *,
    z_bar: Array,
    scales: NondimScales,
    M_list: Sequence[int],
    tip_mode: str = "flex_tip",
) -> Array:
    if str(tip_mode) in {"flex", "flex_tip", "flexible_tip"}:
        return extract_tip_p_dim(z_bar=z_bar, scales=scales, M_list=M_list)
    if str(tip_mode) in {"distal", "distal_tip", "centerline_end"}:
        return extract_distal_tip_p_dim(z_bar=z_bar, scales=scales, M_list=M_list)
    raise ValueError(f"Unknown tip_mode: {tip_mode}")


def extract_interface_position_gaps_dim(
    *,
    z_bar: Array,
    scales: NondimScales,
    M_list: Sequence[int],
) -> Array:
    x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(jnp.asarray(z_bar, dtype=jnp.float64), M_list=M_list)
    gaps: list[Array] = []
    for seg_idx in range(1, len(x_nodes_list_bar)):
        prev_rigid_p = jnp.asarray(x_rigid_list_bar[seg_idx - 1][0:3] * float(scales.L_ref), dtype=jnp.float64)
        curr_prox_p = jnp.asarray(x_nodes_list_bar[seg_idx][0, 0:3] * float(scales.L_ref), dtype=jnp.float64)
        gaps.append(curr_prox_p - prev_rigid_p)
    if not gaps:
        return jnp.zeros((0, 3), dtype=jnp.float64)
    return jnp.stack(gaps, axis=0)


def _centerline_position_indices(M_list: Sequence[int], *, include_rigid: bool) -> list[int]:
    idxs: list[int] = []
    offset = 0
    for m in M_list:
        Mi = int(m)
        for node in range(Mi + 1):
            base = offset + node * 13
            idxs.extend([base + 0, base + 1, base + 2])
        offset += (Mi + 1) * 13
        offset += Mi * 39
        if include_rigid:
            idxs.extend([offset + 0, offset + 1, offset + 2])
        offset += 13
    return idxs


def extract_centerline_p_dim(
    *,
    z_bar: Array,
    scales: NondimScales,
    M_list: Sequence[int],
    include_rigid: bool = True,
    rigid_sample_count: int = 0,
    drop_duplicate_interfaces: bool = False,
) -> Array:
    """Extract the dimensional centerline points used for plotting and smoothing."""
    x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(jnp.asarray(z_bar, dtype=jnp.float64), M_list=M_list)
    pts: list[Array] = []
    sample_count = max(int(rigid_sample_count), 0)
    for seg_idx, x_nodes_bar in enumerate(x_nodes_list_bar):
        node_pts = jnp.asarray(x_nodes_bar[:, 0:3] * float(scales.L_ref), dtype=jnp.float64)
        if drop_duplicate_interfaces and seg_idx > 0:
            node_pts = node_pts[1:, :]
        if int(node_pts.shape[0]) > 0:
            pts.append(node_pts)
        if include_rigid:
            prox_p = jnp.asarray(x_nodes_bar[-1, 0:3] * float(scales.L_ref), dtype=jnp.float64)
            dist_p = jnp.asarray(x_rigid_list_bar[seg_idx][0:3] * float(scales.L_ref), dtype=jnp.float64)
            alphas = jnp.linspace(0.0, 1.0, sample_count + 2, dtype=jnp.float64)[1:]
            rigid_pts = prox_p[None, :] + alphas[:, None] * (dist_p - prox_p)[None, :]
            pts.append(rigid_pts)
    if not pts:
        return jnp.zeros((0, 3), dtype=jnp.float64)
    return jnp.concatenate(pts, axis=0)


def extract_shape_smooth_residual(
    *,
    z_bar: Array,
    scales: NondimScales,
    M_list: Sequence[int],
    include_rigid: bool = True,
    rigid_sample_count: int = 0,
) -> Array:
    """Second-difference position residual to suppress nonphysical sharp bends."""
    centerline = extract_centerline_p_dim(
        z_bar=z_bar,
        scales=scales,
        M_list=M_list,
        include_rigid=include_rigid,
        rigid_sample_count=rigid_sample_count,
        drop_duplicate_interfaces=True,
    )
    if int(centerline.shape[0]) < 3:
        return jnp.zeros((0,), dtype=jnp.float64)

    edges = centerline[1:, :] - centerline[:-1, :]
    edge_norm = jnp.linalg.norm(edges, axis=1, keepdims=True)
    edge_norm = jnp.clip(edge_norm, a_min=1e-12)
    tangents = edges / edge_norm
    tangent_diff = tangents[1:, :] - tangents[:-1, :]
    return jnp.asarray(tangent_diff.reshape(-1,), dtype=jnp.float64)


def make_initial_z0_bar(static_fk: StaticFK) -> Array:
    """Build a consistent z0_bar initial guess for the given static FK setup."""
    params = static_fk.params_base
    z0_bar, *_ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=list(static_fk.meshes_ref),
        rigid_segs=list(params.rigid),
        scales=params.scales,
        p0_dim=static_fk.p0_dim,
        Q0=static_fk.Q0_wxyz,
        axis_body=static_fk.axis_body,
    )
    return jnp.asarray(z0_bar, dtype=jnp.float64)


def build_params_for_y(
    static_fk: StaticFK,
    *,
    I: Array,
    x: Array,
) -> SolverParams:
    """Build per-iteration params from base params by updating (coil currents, L1_dim)."""
    I = jnp.asarray(I, dtype=jnp.float64).reshape(8,)

    x = jnp.asarray(x, dtype=jnp.float64).reshape(())
    L1_dim = x - jnp.asarray(static_fk.L_fixed, dtype=jnp.float64)
    L1_dim = jnp.maximum(L1_dim, jnp.asarray(static_fk.L1_min, dtype=jnp.float64))

    p = with_coil_currents(static_fk.params_base, I)
    p = with_L1_dim(p, jnp.asarray(L1_dim, dtype=jnp.float64).reshape(()))
    return p


def joint_residual_components(
    y: Array,
    *,
    static_fk: StaticFK,
    p_des: Array,
    I_max: float,
    x_min: float,
    x_max: float,
    w_E: float,
    sigma_p: float,
    w_I: float,
    w_x: float,
    w_smooth: float = 0.0,
    x_ref: float,
    smooth_include_rigid: bool = True,
    w_gap: float = 0.0,
    sigma_gap: float = 1e-3,
    centerline_rigid_samples: int = 0,
    I_scale: float = 1.0,
    I_ref: Optional[Array] = None,
    tip_mode: str = "flex_tip",
) -> dict:
    """Compute decoded controls, raw equilibrium residual, and weighted residual blocks.

    This helper is intended for diagnostics/logging and mirrors the logic used by
    :func:`joint_residual`.
    """
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1,)
    nZ = int(static_fk.z_len)

    z_bar = y[0:nZ]
    u = y[nZ:]

    I, x = u_to_Ix(u, I_max=float(I_max), x_min=float(x_min), x_max=float(x_max), I_scale=float(I_scale))

    x = jnp.asarray(x, dtype=jnp.float64).reshape(())
    L1_raw_dim = x - jnp.asarray(static_fk.L_fixed, dtype=jnp.float64)
    L1_dim = jnp.maximum(L1_raw_dim, jnp.asarray(static_fk.L1_min, dtype=jnp.float64))
    params = build_params_for_y(static_fk, I=I, x=x)

    E = residual_bar(z_bar, params)
    r_E = jnp.asarray(w_E, dtype=jnp.float64) * E

    p_des = jnp.asarray(p_des, dtype=jnp.float64).reshape(3,)
    p_tip = extract_task_tip_p_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        tip_mode=str(tip_mode),
    )
    r_p = (p_tip - p_des) / jnp.asarray(sigma_p, dtype=jnp.float64)

    _I_ref = jnp.zeros(8, dtype=jnp.float64) if I_ref is None else jnp.asarray(I_ref, dtype=jnp.float64).reshape(8,)
    r_I = jnp.asarray(w_I, dtype=jnp.float64) * (I - _I_ref)
    r_x = (jnp.asarray(w_x, dtype=jnp.float64) * (x - jnp.asarray(x_ref, dtype=jnp.float64))).reshape((1,))
    interface_gap_p = extract_interface_position_gaps_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
    )
    r_gap = jnp.asarray(w_gap, dtype=jnp.float64) * (
        interface_gap_p.reshape(-1,) / jnp.asarray(sigma_gap, dtype=jnp.float64)
    )
    smooth_raw = extract_shape_smooth_residual(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=bool(smooth_include_rigid),
        rigid_sample_count=int(centerline_rigid_samples),
    )
    r_smooth = jnp.asarray(w_smooth, dtype=jnp.float64) * smooth_raw
    centerline_p = extract_centerline_p_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=bool(smooth_include_rigid),
        rigid_sample_count=int(centerline_rigid_samples),
    )
    smooth_metric = jnp.linalg.norm(smooth_raw) if smooth_raw.size else jnp.asarray(0.0, dtype=jnp.float64)
    interface_gap_norms = jnp.linalg.norm(interface_gap_p, axis=1) if interface_gap_p.size else jnp.zeros((0,), dtype=jnp.float64)
    max_interface_gap = jnp.max(interface_gap_norms) if interface_gap_norms.size else jnp.asarray(0.0, dtype=jnp.float64)

    return {
        "z_bar": z_bar,
        "u": u,
        "I": I,
        "x": x,
        "params": params,
        "E": E,
        "p_tip": p_tip,
        "tip_mode": str(tip_mode),
        "r_E": r_E,
        "r_p": r_p,
        "r_I": r_I,
        "r_x": r_x,
        "r_gap": r_gap,
        "r_smooth": r_smooth,
        "interface_gap_p": interface_gap_p,
        "interface_gap_norms": interface_gap_norms,
        "max_interface_gap": max_interface_gap,
        "smooth_raw": smooth_raw,
        "smooth_metric": smooth_metric,
        "centerline_p": centerline_p,
        "L1_raw_dim": L1_raw_dim,
        "L1_dim": L1_dim,
    }


def joint_residual(
    y: Array,
    *,
    static_fk: StaticFK,
    p_des: Array,
    I_max: float,
    x_min: float,
    x_max: float,
    w_E: float,
    sigma_p: float,
    w_I: float,
    w_x: float,
    w_smooth: float = 0.0,
    x_ref: float,
    smooth_include_rigid: bool = True,
    w_gap: float = 0.0,
    sigma_gap: float = 1e-3,
    centerline_rigid_samples: int = 0,
    I_scale: float = 1.0,
    I_ref: Optional[Array] = None,
    tip_mode: str = "flex_tip",
) -> Array:
    """Unified residual R(y) for single-layer joint IK.

    y = [z_bar, u]
    Residual order:
      - r_E:  w_E * E(z, params(I,x))
      - r_p:  (p_tip(z) - p_des)/sigma_p
      - r_I:  w_I * I
      - r_x:  w_x * (x - x_ref)

    Notes:
      - This first version keeps residual length fixed and avoids optional terms.
      - Box constraints for I and x are enforced by parameterization u->(I,x).
    """
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1,)

    nZ = int(static_fk.z_len)

    z_bar = y[0:nZ]
    u = y[nZ:]

    I, x = u_to_Ix(u, I_max=float(I_max), x_min=float(x_min), x_max=float(x_max), I_scale=float(I_scale))
    params = build_params_for_y(static_fk, I=I, x=x)

    E = residual_bar(z_bar, params)
    r_E = jnp.asarray(w_E, dtype=jnp.float64) * E

    p_des = jnp.asarray(p_des, dtype=jnp.float64).reshape(3,)
    p_tip = extract_task_tip_p_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        tip_mode=str(tip_mode),
    )
    r_p = (p_tip - p_des) / jnp.asarray(sigma_p, dtype=jnp.float64)

    _I_ref = jnp.zeros(8, dtype=jnp.float64) if I_ref is None else jnp.asarray(I_ref, dtype=jnp.float64).reshape(8,)
    r_I = jnp.asarray(w_I, dtype=jnp.float64) * (I - _I_ref)
    r_x = (jnp.asarray(w_x, dtype=jnp.float64) * (x - jnp.asarray(x_ref, dtype=jnp.float64))).reshape((1,))
    
    interface_gap_p = extract_interface_position_gaps_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
    )
    r_gap = jnp.asarray(w_gap, dtype=jnp.float64) * (
        interface_gap_p.reshape(-1,) / jnp.asarray(sigma_gap, dtype=jnp.float64)
    )
    
    r_smooth = jnp.asarray(w_smooth, dtype=jnp.float64) * extract_shape_smooth_residual(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=bool(smooth_include_rigid),
        rigid_sample_count=int(centerline_rigid_samples),
    )

    return jnp.concatenate([r_E, r_p, r_I, r_x, r_gap, r_smooth], axis=0)
