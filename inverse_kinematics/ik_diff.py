"""Implicit differentiation utilities for magnetic-catheter IK (position-only).

This module provides a practical, *engineering-oriented* way to estimate the
Jacobian of the FK tip position w.r.t. coil currents:

    Jp = dp_tip / dI   (shape: 3 x n_coils)

The implementation follows the "reuse LM Jacobian" idea:
  - Inner equilibrium is solved by LM on a nonlinear least-squares residual E(z, I).
  - At the solution (z*, I), LM forms Jz = dE/dz.
  - We approximate the implicit sensitivity via the Gauss-Newton/LM linearization:

        (Jz^T Jz + lam I) U = (dp/dz)^T
        Jp \approx - U^T (Jz^T JI)

where JI = dE/dI.

This is *not* an exact implicit function theorem derivative (because E is
overdetermined). Instead, it is the standard adjoint sensitivity of the
least-squares KKT system used by LM/GN, and is typically what you want for
gradient-based IK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from forward_kinematics_optimized.nondim import x_bar_to_dim
from forward_kinematics_optimized.utils_nondim import unpack_z_bar_jax
from forward_kinematics_optimized.equilibrium_solver_nondim import residual_bar_zI, residual_bar_zIx, SolverParams, LMStats
from inverse_kinematics.ik_observations import ObservationSpec, compute_observation_residual

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


# ---------------------------------------------------------------------
# JAX compilation cache
# ---------------------------------------------------------------------
#
# For IK, we repeatedly compute Jacobians with identical tensor shapes.
# If we define the transformed functions (jacfwd/jacrev) inside a loop,
# JAX will often retrace/recompile. Keeping them at module scope ensures
# the XLA executables are cached and reused.

_JAC_E_WRT_I_FWD = jax.jit(jax.jacfwd(residual_bar_zI, argnums=1))
_JAC_E_WRT_I_REV = jax.jit(jax.jacrev(residual_bar_zI, argnums=1))
_JAC_E_WRT_Z_IX_FWD = jax.jit(jax.jacfwd(residual_bar_zIx, argnums=0))
_JAC_E_WRT_IX_FWD = jax.jit(jax.jacfwd(residual_bar_zIx, argnums=(1, 2)))


def _p_of_z_params(z_bar: Array, params: SolverParams) -> Array:
    """Helper for differentiating tip position wrt z (params carries scales/M_list)."""
    return extract_tip_p_dim(z_bar, M_list=params.M_list, scales=params.scales)


_DP_DZ_REV = jax.jit(jax.jacrev(_p_of_z_params, argnums=0))


def extract_tip_p_dim(z_bar: Array, *, M_list: Sequence[int], scales) -> Array:
    """Extract the current FK task point in SI units from packed ``z_bar``.

    The canonical task point for the current FK/IK stack is the distal node of
    the last flexible segment. This matches ``ForwardKinematicsEngine`` and the
    single-layer joint residual, so adjoint sensitivities and residuals refer to
    the same point.
    """
    x_nodes_list_bar, _k_list_bar, _x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=M_list)
    x_tip_bar = x_nodes_list_bar[-1][-1]
    x_tip_dim = x_bar_to_dim(x_tip_bar, scales)
    return x_tip_dim[0:3]


@dataclass
class TipJacobianResult:
    """Return bundle for dp/dI computation."""

    p_tip_dim: Array          # (3,)
    J_p_I: Array              # (3, nI)
    J_E_z: Array              # (nE, nZ)
    J_E_I: Array              # (nE, nI)
    lam: float


@dataclass
class TipSensitivityResult:
    """Return bundle for implicit tip sensitivity in control variables."""

    p_tip_dim: Array
    J_p_I: Array
    J_p_x: Array
    J_E_z: Optional[Array]
    J_E_I: Optional[Array]
    J_E_x: Optional[Array]
    lam: float
    stationarity_norm: float
    condition_number: Optional[float]
    status: str
    method: str


@dataclass
class TaskSensitivityResult:
    """Return bundle for diagnostic multi-observation sensitivity."""

    residual: Array
    J_r_I: Array
    J_r_x: Array
    J_E_z: Optional[Array]
    J_E_I: Optional[Array]
    J_E_x: Optional[Array]
    lam: float
    stationarity_norm: float
    condition_number: Optional[float]
    status: str
    method: str


def _classify_conditioning(
    *,
    normE: float,
    stationarity_norm: float,
    condition_number: Optional[float],
    weak_normE: float,
    cond_warning: float,
    cond_bad: float,
) -> str:
    if normE > weak_normE:
        return "weak_equilibrium"
    if condition_number is not None:
        if (not jnp.isfinite(jnp.asarray(condition_number))) or float(condition_number) >= float(cond_bad):
            return "ill_conditioned"
        if float(condition_number) >= float(cond_warning):
            return "weak_conditioning"
    if stationarity_norm > 1.0:
        return "weak_stationarity"
    return "ok"


def compute_tip_sensitivities_via_lm_adjoint(
    *,
    z_star_bar: Array,
    params: SolverParams,
    lm_stats: LMStats,
    coil_currents: Array,
    ridge: Optional[float] = None,
    J_E_z: Optional[Array] = None,
    materialize_jacobian: bool = True,
    compute_conditioning: bool = False,
    weak_normE: float = 1e-1,
    cond_warning: float = 1e8,
    cond_bad: float = 1e10,
) -> TipSensitivityResult:
    """Compute implicit tip sensitivity with respect to currents and protrusion.

    The residual is overdetermined, so this uses the LM/Gauss-Newton
    stationarity approximation rather than a square implicit-function solve.
    The ``x`` derivative is taken through ``L1_dim``; for the current advancer
    convention, ``dL1_dim / dL_protrude = 1`` away from clipping bounds.

    Args:
        z_star_bar: Equilibrium state in nondimensional units.
        params: Solver parameters used by the FK solve.
        lm_stats: LM diagnostics. ``lm_stats.J`` is reused if present, but is
            not required.
        coil_currents: Coil current vector in ampere.
        ridge: Optional ridge added to the LM damping.
        J_E_z: Optional precomputed ``dE/dz``. This can come from the FK
            solver's cached ``jacobian_jit`` without storing it in ``lm_stats``.
        materialize_jacobian: Must currently be true. The Jacobian may be
            materialized inside this routine, but it is not stored back into
            ``lm_stats``.
        compute_conditioning: Whether to compute ``cond(H)`` for diagnostics.
        weak_normE: Residual norm threshold above which sensitivity is weak.
        cond_warning: Condition number threshold for weak conditioning.
        cond_bad: Condition number threshold for ill conditioning.

    Returns:
        TipSensitivityResult with ``dp/dI`` and ``dp/dx``.
    """
    if not materialize_jacobian:
        raise NotImplementedError("matrix-free implicit sensitivity is not enabled as the default path")
    if lm_stats is None:
        raise ValueError("lm_stats is required for damping and residual diagnostics")

    I0 = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,)
    z0 = jnp.asarray(z_star_bar, dtype=jnp.float64).reshape(-1,)
    L1 = jnp.asarray(params.L1_dim, dtype=jnp.float64).reshape(())

    if J_E_z is not None:
        Jz = jnp.asarray(J_E_z, dtype=jnp.float64)
    elif getattr(lm_stats, "J", None) is not None:
        Jz = jnp.asarray(lm_stats.J, dtype=jnp.float64)
    else:
        Jz = _JAC_E_WRT_Z_IX_FWD(z0, I0, L1, params)

    JI, Jx = _JAC_E_WRT_IX_FWD(z0, I0, L1, params)
    JI = jnp.asarray(JI, dtype=jnp.float64)
    Jx = jnp.asarray(Jx, dtype=jnp.float64).reshape((-1, 1))
    B = jnp.concatenate([JI, Jx], axis=1)

    dp_dz = _DP_DZ_REV(z0, params)
    lam_used = float(lm_stats.lam)
    if ridge is not None:
        lam_used = float(lam_used + ridge)

    Jt = jnp.transpose(Jz)
    H = Jt @ Jz
    nZ = H.shape[0]
    H = H + lam_used * jnp.eye(nZ, dtype=H.dtype)

    U = jnp.linalg.solve(H, jnp.transpose(dp_dz))
    JT_B = Jt @ B
    Jp_all = -jnp.transpose(U) @ JT_B

    E = jnp.asarray(lm_stats.E_star, dtype=jnp.float64)
    stationarity_norm = float(jnp.linalg.norm(Jt @ E))
    condition_number = None
    if bool(compute_conditioning):
        condition_number = float(jnp.linalg.cond(H))
    normE = float(getattr(lm_stats, "normE", jnp.linalg.norm(E)))
    status = _classify_conditioning(
        normE=normE,
        stationarity_norm=stationarity_norm,
        condition_number=condition_number,
        weak_normE=float(weak_normE),
        cond_warning=float(cond_warning),
        cond_bad=float(cond_bad),
    )

    p_tip = _p_of_z_params(z0, params)
    nI = int(I0.shape[0])
    return TipSensitivityResult(
        p_tip_dim=p_tip,
        J_p_I=Jp_all[:, :nI],
        J_p_x=Jp_all[:, nI],
        J_E_z=Jz,
        J_E_I=JI,
        J_E_x=Jx[:, 0],
        lam=float(lam_used),
        stationarity_norm=float(stationarity_norm),
        condition_number=condition_number,
        status=str(status),
        method="dense_implicit_transitional",
    )


def compute_dp_dI_via_lm_adjoint(
    *,
    z_star_bar: Array,
    params: SolverParams,
    lm_stats: LMStats,
    coil_currents: Array,
    ridge: Optional[float] = None,
    jac_method_I: str = "fwd",
) -> TipJacobianResult:
    """Compute dp/dI using (approx.) LM-adjoint sensitivity.

    Args:
        z_star_bar: equilibrium solution z* (bar units).
        params: SolverParams used in the FK solve.
        lm_stats: LMStats returned by solver.solve_lm(return_stats=True).
        coil_currents: current vector I, shape (nI,).
        ridge: optional additional ridge term (added to LM damping).
        jac_method_I: differentiation mode for JI. For nI=8, 'fwd' is typical.

    Returns:
        TipJacobianResult containing Jp (3 x nI) and intermediate Jacobians.
    """
    if jac_method_I not in ("fwd", "rev"):
        raise ValueError("jac_method_I must be 'fwd' or 'rev'")
    sens = compute_tip_sensitivities_via_lm_adjoint(
        z_star_bar=z_star_bar,
        params=params,
        lm_stats=lm_stats,
        coil_currents=coil_currents,
        ridge=ridge,
        J_E_z=None,
        materialize_jacobian=True,
    )


def _sites_from_z_params(z_bar: Array, params: SolverParams) -> dict:
    """Build a ``query_sites``-compatible pose dictionary from ``z_bar``."""
    x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    x_tip_dim = x_bar_to_dim(x_nodes_list_bar[-1][-1], params.scales)

    rigid_p = []
    rigid_q = []
    for x_rigid_bar in x_rigid_list_bar:
        x_rigid_dim = x_bar_to_dim(x_rigid_bar, params.scales)
        rigid_p.append(x_rigid_dim[0:3])
        rigid_q.append(x_rigid_dim[3:7])
    if rigid_p:
        rigid_mid_p = jnp.stack(rigid_p, axis=0)
        rigid_mid_q = jnp.stack(rigid_q, axis=0)
    else:
        rigid_mid_p = jnp.zeros((0, 3), dtype=jnp.float64)
        rigid_mid_q = jnp.zeros((0, 4), dtype=jnp.float64)

    return {
        "tip_p_dim": x_tip_dim[0:3],
        "tip_Q_wxyz": x_tip_dim[3:7],
        "rigid_mid_p_dim": rigid_mid_p,
        "rigid_mid_Q_wxyz": rigid_mid_q,
    }


def _task_residual_from_z(z_bar: Array, params: SolverParams, specs: Sequence[ObservationSpec]) -> Array:
    """Compute stacked observation residuals from a packed FK state."""
    sites = _sites_from_z_params(z_bar, params)
    pieces = []
    for spec in specs:
        pieces.append(compute_observation_residual(sites, spec).residual.reshape(-1))
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(pieces, axis=0)


def compute_task_sensitivities_via_lm_adjoint(
    *,
    z_star_bar: Array,
    params: SolverParams,
    lm_stats: LMStats,
    coil_currents: Array,
    specs: Sequence[ObservationSpec],
    ridge: Optional[float] = None,
    J_E_z: Optional[Array] = None,
    compute_conditioning: bool = False,
    weak_normE: float = 1e-1,
    cond_warning: float = 1e8,
    cond_bad: float = 1e10,
) -> TaskSensitivityResult:
    """Compute dense diagnostic sensitivity for arbitrary observation residuals.

    This is a P2 diagnostic path. It materializes dense Jacobians and should not
    be used as the production path for large IK problems.
    """
    if lm_stats is None:
        raise ValueError("lm_stats is required for damping and residual diagnostics")
    if not specs:
        raise ValueError("at least one ObservationSpec is required")

    I0 = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,)
    z0 = jnp.asarray(z_star_bar, dtype=jnp.float64).reshape(-1,)
    L1 = jnp.asarray(params.L1_dim, dtype=jnp.float64).reshape(())

    if J_E_z is not None:
        Jz = jnp.asarray(J_E_z, dtype=jnp.float64)
    elif getattr(lm_stats, "J", None) is not None:
        Jz = jnp.asarray(lm_stats.J, dtype=jnp.float64)
    else:
        Jz = _JAC_E_WRT_Z_IX_FWD(z0, I0, L1, params)

    JI, Jx = _JAC_E_WRT_IX_FWD(z0, I0, L1, params)
    JI = jnp.asarray(JI, dtype=jnp.float64)
    Jx = jnp.asarray(Jx, dtype=jnp.float64).reshape((-1, 1))
    B = jnp.concatenate([JI, Jx], axis=1)

    residual_fn = lambda z: _task_residual_from_z(z, params, specs)
    residual = residual_fn(z0)
    Jr_z = jax.jacrev(residual_fn)(z0)

    lam_used = float(lm_stats.lam)
    if ridge is not None:
        lam_used = float(lam_used + ridge)

    Jt = jnp.transpose(Jz)
    H = Jt @ Jz
    nZ = H.shape[0]
    H = H + lam_used * jnp.eye(nZ, dtype=H.dtype)

    U = jnp.linalg.solve(H, jnp.transpose(Jr_z))
    JT_B = Jt @ B
    Jr_all = -jnp.transpose(U) @ JT_B

    E = jnp.asarray(lm_stats.E_star, dtype=jnp.float64)
    stationarity_norm = float(jnp.linalg.norm(Jt @ E))
    condition_number = None
    if bool(compute_conditioning):
        condition_number = float(jnp.linalg.cond(H))
    normE = float(getattr(lm_stats, "normE", jnp.linalg.norm(E)))
    status = _classify_conditioning(
        normE=normE,
        stationarity_norm=stationarity_norm,
        condition_number=condition_number,
        weak_normE=float(weak_normE),
        cond_warning=float(cond_warning),
        cond_bad=float(cond_bad),
    )

    nI = int(I0.shape[0])
    return TaskSensitivityResult(
        residual=residual,
        J_r_I=Jr_all[:, :nI],
        J_r_x=Jr_all[:, nI],
        J_E_z=Jz,
        J_E_I=JI,
        J_E_x=Jx[:, 0],
        lam=float(lam_used),
        stationarity_norm=float(stationarity_norm),
        condition_number=condition_number,
        status=str(status),
        method="dense_task_implicit_diagnostic",
    )
    return TipJacobianResult(
        p_tip_dim=sens.p_tip_dim,
        J_p_I=sens.J_p_I,
        J_E_z=jnp.asarray(sens.J_E_z),
        J_E_I=jnp.asarray(sens.J_E_I),
        lam=float(sens.lam),
    )
