from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


from .forward_kinematics_optimized_bak.advancer_nondim import apply_advancer_protrude_length
from .forward_kinematics_optimized_bak.equilibrium_solver_nondim import (
    SolverParams,
    residual_bar,
    with_coil_currents,
    with_L1_dim,
)
from .forward_kinematics_optimized_bak.fk import build_solver_params, compute_scales_from_flex
from .forward_kinematics_optimized_bak.nondim import NondimScales, x_bar_to_dim
from .forward_kinematics_optimized_bak.utils_nondim import make_initial_guess_multi_bar_jax, unpack_z_bar_jax


jax.config.update("jax_enable_x64", True)
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
) -> tuple[Array, Array]:
    """Map unconstrained u -> (I, x) with box constraints.

    u layout:
      - u[0:8]: currents params
      - u[8]: insertion param
    """
    u = jnp.asarray(u, dtype=jnp.float64).reshape(-1,)

    uI = u[0:8]
    ux = u[8]

    I = jnp.asarray(I_max, dtype=jnp.float64) * jnp.tanh(uI)
    x = jnp.asarray(x_min, dtype=jnp.float64) + (jnp.asarray(x_max, dtype=jnp.float64) - jnp.asarray(x_min, dtype=jnp.float64)) * _sigmoid(ux)
    return jnp.asarray(I, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64).reshape(())


def Ix_to_u(
    I: Array,
    x: float,
    *,
    I_max: float,
    x_min: float,
    x_max: float,
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
    uI = jnp.arctanh(I_clip)

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
    x_nodes_list_bar, _, _ = unpack_z_bar_jax(jnp.asarray(z_bar, dtype=jnp.float64), M_list=tuple(int(m) for m in M_list))
    x_tip_bar = x_nodes_list_bar[-1][-1]
    x_tip_dim = x_bar_to_dim(x_tip_bar, scales)
    return jnp.asarray(x_tip_dim[0:3], dtype=jnp.float64)


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
    x_ref: float,
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

    I, x = u_to_Ix(u, I_max=float(I_max), x_min=float(x_min), x_max=float(x_max))
    params = build_params_for_y(static_fk, I=I, x=x)

    E = residual_bar(z_bar, params)
    r_E = jnp.asarray(w_E, dtype=jnp.float64) * E

    p_des = jnp.asarray(p_des, dtype=jnp.float64).reshape(3,)
    p_tip = extract_tip_p_dim(z_bar=z_bar, scales=params.scales, M_list=params.M_list)
    r_p = (p_tip - p_des) / jnp.asarray(sigma_p, dtype=jnp.float64)

    r_I = jnp.asarray(w_I, dtype=jnp.float64) * I
    r_x = (jnp.asarray(w_x, dtype=jnp.float64) * (x - jnp.asarray(x_ref, dtype=jnp.float64))).reshape((1,))

    return jnp.concatenate([r_E, r_p, r_I, r_x], axis=0)
