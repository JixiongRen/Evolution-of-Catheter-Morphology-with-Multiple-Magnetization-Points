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

from dataclasses import dataclass, replace
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


def _stationarity_zix(z_bar: Array, coil_currents: Array, L1_dim: Array, params_static: SolverParams) -> Array:
    """Return the least-squares stationarity vector ``Jz.T @ E``."""
    residual_fn = lambda zz: residual_bar_zIx(zz, coil_currents, L1_dim, params_static)
    E = residual_fn(z_bar)
    Jz = jax.jacfwd(residual_fn)(z_bar)
    return jnp.transpose(Jz) @ E


_JAC_G_WRT_ZIX_FWD = jax.jit(jax.jacfwd(_stationarity_zix, argnums=(0, 1, 2)))


def _p_of_z_params(z_bar: Array, params: SolverParams) -> Array:
    """Helper for differentiating the canonical IK terminal point wrt z."""
    return extract_terminal_p_dim(z_bar, M_list=params.M_list, scales=params.scales)


_DP_DZ_REV = jax.jit(jax.jacrev(_p_of_z_params, argnums=0))


def extract_tip_p_dim(z_bar: Array, *, M_list: Sequence[int], scales) -> Array:
    """Extract the flexible distal node in SI units from packed ``z_bar``.

    This is a compatibility observation. The supervised delivery IK now uses
    ``extract_terminal_p_dim`` as the canonical task point because the physical
    catheter end is the distal end of the last rigid segment.
    """
    x_nodes_list_bar, _k_list_bar, _x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=M_list)
    x_tip_bar = x_nodes_list_bar[-1][-1]
    x_tip_dim = x_bar_to_dim(x_tip_bar, scales)
    return x_tip_dim[0:3]


def extract_terminal_p_dim(z_bar: Array, *, M_list: Sequence[int], scales) -> Array:
    """Extract the physical distal catheter end in SI units from packed ``z_bar``."""
    x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=M_list)
    if x_rigid_list_bar:
        x_terminal_bar = x_rigid_list_bar[-1]
    else:
        x_terminal_bar = x_nodes_list_bar[-1][-1]
    x_terminal_dim = x_bar_to_dim(x_terminal_bar, scales)
    return x_terminal_dim[0:3]


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
    sv_min: Optional[float] = None
    sv_max: Optional[float] = None
    effective_rank: Optional[int] = None
    lm_lam: Optional[float] = None
    ridge_used: float = 0.0
    solve_mode: str = "solve"
    spectral_filter: str = "none"
    spectral_lambda: float = 0.0
    spectral_cutoff: Optional[float] = None
    retained_spectral_energy: Optional[float] = None


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
    sv_min: Optional[float]
    sv_max: Optional[float]
    effective_rank: Optional[int]
    lm_lam: float
    extra_ridge: float
    ridge_used: float
    solve_mode: str
    status: str
    method: str
    spectral_filter: str = "none"
    spectral_lambda: float = 0.0
    spectral_cutoff: Optional[float] = None
    retained_spectral_energy: Optional[float] = None


@dataclass
class ResidualSpectralSolveResult:
    """Return bundle for residual-space spectral linear solves."""

    dz_dc: Array
    sv_min: float
    sv_max: float
    effective_rank: int
    condition_number: float
    spectral_filter: str
    spectral_lambda: float
    spectral_cutoff: float
    retained_spectral_energy: float


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


def _linear_solve_with_svd_fallback(
    H: Array,
    rhs: Array,
    *,
    solve_mode: str,
    svd_rcond: float,
    matrix_svd_rcond: Optional[float] = None,
) -> Tuple[Array, Optional[float], Optional[float], Optional[int], Optional[float], str]:
    """Solve ``H X = rhs`` and return spectrum diagnostics."""
    solve_mode = str(solve_mode)
    if solve_mode not in ("solve", "svd"):
        raise ValueError("solve_mode must be 'solve' or 'svd'")
    if solve_mode == "solve":
        X = jnp.linalg.solve(H, rhs)
        return X, None, None, None, None, "solve"

    U, S, Vh = jnp.linalg.svd(H, full_matrices=False)
    sv_max = jnp.max(S)
    cutoff_rcond = float(svd_rcond if matrix_svd_rcond is None else matrix_svd_rcond)
    cutoff = cutoff_rcond * sv_max
    keep = S > cutoff
    inv_S = jnp.where(keep, 1.0 / S, 0.0)
    X = Vh.T @ (inv_S[:, None] * (U.T @ rhs))
    sv_min_val = float(jnp.min(S))
    sv_max_val = float(sv_max)
    effective_rank = int(jnp.sum(keep))
    condition_number = float(sv_max / jnp.maximum(jnp.min(S), jnp.finfo(S.dtype).tiny))
    return X, sv_min_val, sv_max_val, effective_rank, condition_number, "svd"


def _solve_residual_spectral(
    Jz: Array,
    B: Array,
    *,
    svd_rcond: float,
    spectral_filter: str = "hard",
    spectral_damping: float = 0.0,
) -> ResidualSpectralSolveResult:
    """Solve ``Jz dz/dc = -B`` with auditable residual-space filtering."""
    spectral_filter = str(spectral_filter)
    if spectral_filter not in ("hard", "tikhonov", "hybrid"):
        raise ValueError("spectral_filter must be 'hard', 'tikhonov', or 'hybrid'")

    U, S, Vh = jnp.linalg.svd(Jz, full_matrices=False)
    sv_max = jnp.max(S)
    sv_min = jnp.min(S)
    cutoff = float(svd_rcond) * sv_max
    keep = S > cutoff
    damping = float(spectral_damping)
    if damping <= 0.0:
        damping = float(cutoff)

    if spectral_filter == "hard":
        gains = jnp.where(keep, 1.0 / S, 0.0)
        lambda_used = 0.0
    elif spectral_filter == "tikhonov":
        lambda_used = float(damping)
        gains = S / (S * S + lambda_used * lambda_used)
    else:
        lambda_used = float(damping)
        gains = jnp.where(keep, S / (S * S + lambda_used * lambda_used), 0.0)

    dz_dc = -(Vh.T @ (gains[:, None] * (U.T @ B)))
    spectral_energy = jnp.sum(S * S)
    retained_energy = jnp.sum(jnp.where(keep, S * S, 0.0)) / jnp.maximum(
        spectral_energy, jnp.finfo(S.dtype).tiny
    )
    condition_number = sv_max / jnp.maximum(sv_min, jnp.finfo(S.dtype).tiny)
    return ResidualSpectralSolveResult(
        dz_dc=dz_dc,
        sv_min=float(sv_min),
        sv_max=float(sv_max),
        effective_rank=int(jnp.sum(keep)),
        condition_number=float(condition_number),
        spectral_filter=str(spectral_filter),
        spectral_lambda=float(lambda_used),
        spectral_cutoff=float(cutoff),
        retained_spectral_energy=float(retained_energy),
    )


def _tip_sensitivity_from_linearized_stationarity(
    *,
    z_star_bar: Array,
    params: SolverParams,
    lm_stats: LMStats,
    coil_currents: Array,
    H: Array,
    B: Array,
    Jz: Array,
    JI: Array,
    Jx: Array,
    lam_used: float,
    method: str,
    solve_mode: str = "solve",
    svd_rcond: float = 1e-10,
    compute_conditioning: bool = False,
    weak_normE: float = 1e-1,
    cond_warning: float = 1e8,
    cond_bad: float = 1e10,
    ridge_used: float = 0.0,
    normal_equation_svd_rcond: Optional[float] = None,
) -> TipSensitivityResult:
    """Map an equilibrium linearization into terminal-position sensitivities."""
    z0 = jnp.asarray(z_star_bar, dtype=jnp.float64).reshape(-1,)
    dp_dz = _DP_DZ_REV(z0, params)
    U, sv_min, sv_max, effective_rank, condition_number, used_solve_mode = _linear_solve_with_svd_fallback(
        H,
        jnp.transpose(dp_dz),
        solve_mode=solve_mode,
        svd_rcond=float(svd_rcond),
        matrix_svd_rcond=normal_equation_svd_rcond,
    )
    Jp_all = -jnp.transpose(U) @ B

    E = jnp.asarray(lm_stats.E_star, dtype=jnp.float64)
    Jt = jnp.transpose(Jz)
    stationarity_norm = float(jnp.linalg.norm(Jt @ E))
    if bool(compute_conditioning) and condition_number is None:
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
    if used_solve_mode == "svd" and effective_rank is not None and effective_rank < int(H.shape[0]):
        status = "truncated_svd" if status == "ok" else f"{status}|truncated_svd"

    I0 = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,)
    nI = int(I0.shape[0])
    return TipSensitivityResult(
        p_tip_dim=_p_of_z_params(z0, params),
        J_p_I=Jp_all[:, :nI],
        J_p_x=Jp_all[:, nI],
        J_E_z=Jz,
        J_E_I=JI,
        J_E_x=Jx[:, 0],
        lam=float(lam_used),
        stationarity_norm=float(stationarity_norm),
        condition_number=condition_number,
        status=str(status),
        method=str(method),
        sv_min=sv_min,
        sv_max=sv_max,
        effective_rank=effective_rank,
        lm_lam=float(lm_stats.lam),
        ridge_used=float(ridge_used),
        solve_mode=str(used_solve_mode),
    )


def _tip_sensitivity_from_residual_svd(
    *,
    z_star_bar: Array,
    params: SolverParams,
    lm_stats: LMStats,
    coil_currents: Array,
    Jz: Array,
    JI: Array,
    Jx: Array,
    B: Array,
    svd_rcond: float,
    spectral_filter: str = "hard",
    spectral_damping: float = 0.0,
    compute_conditioning: bool = False,
    weak_normE: float = 1e-1,
    cond_warning: float = 1e8,
    cond_bad: float = 1e10,
    ridge_used: float = 0.0,
) -> TipSensitivityResult:
    """Compute terminal sensitivity by solving the residual Jacobian directly."""
    z0 = jnp.asarray(z_star_bar, dtype=jnp.float64).reshape(-1,)
    dp_dz = _DP_DZ_REV(z0, params)
    spectral = _solve_residual_spectral(
        Jz,
        B,
        svd_rcond=float(svd_rcond),
        spectral_filter=str(spectral_filter),
        spectral_damping=float(spectral_damping),
    )
    Jp_all = dp_dz @ spectral.dz_dc

    condition_number = None
    if bool(compute_conditioning):
        condition_number = float(spectral.condition_number)

    E = jnp.asarray(lm_stats.E_star, dtype=jnp.float64)
    Jt = jnp.transpose(Jz)
    stationarity_norm = float(jnp.linalg.norm(Jt @ E))
    normE = float(getattr(lm_stats, "normE", jnp.linalg.norm(E)))
    status = _classify_conditioning(
        normE=normE,
        stationarity_norm=stationarity_norm,
        condition_number=condition_number,
        weak_normE=float(weak_normE),
        cond_warning=float(cond_warning),
        cond_bad=float(cond_bad),
    )
    if spectral.spectral_filter == "hard" and spectral.effective_rank < int(Jz.shape[1]):
        status = "truncated_svd" if status == "ok" else f"{status}|truncated_svd"
    elif spectral.spectral_filter != "hard":
        status = "spectral_filtered" if status == "ok" else f"{status}|spectral_filtered"

    I0 = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,)
    nI = int(I0.shape[0])
    return TipSensitivityResult(
        p_tip_dim=_p_of_z_params(z0, params),
        J_p_I=Jp_all[:, :nI],
        J_p_x=Jp_all[:, nI],
        J_E_z=Jz,
        J_E_I=JI,
        J_E_x=Jx[:, 0],
        lam=0.0,
        stationarity_norm=float(stationarity_norm),
        condition_number=condition_number,
        status=str(status),
        method="dense_implicit_residual_svd",
        sv_min=float(spectral.sv_min),
        sv_max=float(spectral.sv_max),
        effective_rank=int(spectral.effective_rank),
        lm_lam=float(lm_stats.lam),
        ridge_used=float(ridge_used),
        solve_mode="residual_svd",
        spectral_filter=str(spectral.spectral_filter),
        spectral_lambda=float(spectral.spectral_lambda),
        spectral_cutoff=float(spectral.spectral_cutoff),
        retained_spectral_energy=float(spectral.retained_spectral_energy),
    )


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
    sensitivity_mode: str = "solver_response",
    solve_mode: str = "solve",
    svd_rcond: float = 1e-10,
    spectral_filter: str = "hard",
    spectral_damping: float = 0.0,
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
        sensitivity_mode: Linearization to use. ``"solver_response"`` preserves
            the historical damped LM response, ``"gn_svd"`` removes the solver
            damping and solves the Gauss-Newton stationarity approximation, and
            ``"stationarity_exact"`` differentiates ``Jz.T @ E = 0`` directly.
            ``"residual_svd"`` solves the rectangular residual Jacobian system
            directly and avoids squaring the condition number.
        solve_mode: Linear solver for the stationarity system, either
            ``"solve"`` or ``"svd"``.
        svd_rcond: Relative singular-value cutoff used when ``solve_mode`` is
            ``"svd"``.
        spectral_filter: Residual-space filter for ``"residual_svd"``. The
            default ``"hard"`` preserves the previous truncated SVD behavior.
        spectral_damping: Absolute damping parameter for residual-space
            ``"tikhonov"`` and ``"hybrid"`` filters. A non-positive value uses
            ``svd_rcond * max(singular_value)``.

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

    Jt = jnp.transpose(Jz)
    H = Jt @ Jz
    nZ = H.shape[0]
    extra_ridge = 0.0 if ridge is None else float(ridge)
    sensitivity_mode = str(sensitivity_mode)
    if sensitivity_mode not in ("solver_response", "gn_svd", "stationarity_exact", "residual_svd"):
        raise ValueError(
            "sensitivity_mode must be 'solver_response', 'gn_svd', "
            "'stationarity_exact', or 'residual_svd'"
        )

    if sensitivity_mode == "solver_response":
        lam_used = float(lm_stats.lam) + extra_ridge
        H_used = H + lam_used * jnp.eye(nZ, dtype=H.dtype)
        B_used = Jt @ B
        method = "dense_implicit_solver_response"
        ridge_used = extra_ridge
    elif sensitivity_mode == "gn_svd":
        if str(solve_mode) == "svd" and abs(extra_ridge) <= 1e-9:
            result = _tip_sensitivity_from_residual_svd(
                z_star_bar=z0,
                params=params,
                lm_stats=lm_stats,
                coil_currents=I0,
                Jz=Jz,
                JI=JI,
                Jx=Jx,
                B=B,
                svd_rcond=float(svd_rcond),
                spectral_filter=str(spectral_filter),
                spectral_damping=float(spectral_damping),
                compute_conditioning=compute_conditioning,
                weak_normE=float(weak_normE),
                cond_warning=float(cond_warning),
                cond_bad=float(cond_bad),
                ridge_used=extra_ridge,
            )
            return replace(result, method="dense_implicit_gn_residual_spectrum", lam=0.0)
        lam_used = extra_ridge
        H_used = H + extra_ridge * jnp.eye(nZ, dtype=H.dtype)
        B_used = Jt @ B
        method = "dense_implicit_gn_without_solver_damping"
        ridge_used = extra_ridge
        normal_equation_svd_rcond = float(svd_rcond) ** 2
    elif sensitivity_mode == "residual_svd":
        return _tip_sensitivity_from_residual_svd(
            z_star_bar=z0,
            params=params,
            lm_stats=lm_stats,
            coil_currents=I0,
            Jz=Jz,
            JI=JI,
            Jx=Jx,
            B=B,
            svd_rcond=float(svd_rcond),
            spectral_filter=str(spectral_filter),
            spectral_damping=float(spectral_damping),
            compute_conditioning=compute_conditioning,
            weak_normE=float(weak_normE),
            cond_warning=float(cond_warning),
            cond_bad=float(cond_bad),
            ridge_used=extra_ridge,
        )
    else:
        Gz, GI, Gx = _JAC_G_WRT_ZIX_FWD(z0, I0, L1, params)
        GI = jnp.asarray(GI, dtype=jnp.float64)
        Gx = jnp.asarray(Gx, dtype=jnp.float64).reshape((-1, 1))
        lam_used = extra_ridge
        H_used = jnp.asarray(Gz, dtype=jnp.float64) + extra_ridge * jnp.eye(nZ, dtype=H.dtype)
        B_used = jnp.concatenate([GI, Gx], axis=1)
        method = "dense_implicit_exact_stationarity"
        ridge_used = extra_ridge
        normal_equation_svd_rcond = None

    if sensitivity_mode == "solver_response":
        normal_equation_svd_rcond = float(svd_rcond) ** 2

    return _tip_sensitivity_from_linearized_stationarity(
        z_star_bar=z0,
        params=params,
        lm_stats=lm_stats,
        coil_currents=I0,
        H=H_used,
        B=B_used,
        Jz=Jz,
        JI=JI,
        Jx=Jx,
        lam_used=float(lam_used),
        method=method,
        solve_mode=solve_mode,
        svd_rcond=float(svd_rcond),
        compute_conditioning=compute_conditioning,
        weak_normE=float(weak_normE),
        cond_warning=float(cond_warning),
        cond_bad=float(cond_bad),
        ridge_used=float(ridge_used),
        normal_equation_svd_rcond=normal_equation_svd_rcond,
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
    return TipJacobianResult(
        p_tip_dim=sens.p_tip_dim,
        J_p_I=sens.J_p_I,
        J_E_z=jnp.asarray(sens.J_E_z),
        J_E_I=jnp.asarray(sens.J_E_I),
        lam=float(sens.lam),
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

    if rigid_mid_p.shape[0] > 0:
        terminal_p = rigid_mid_p[-1]
        terminal_q = rigid_mid_q[-1]
    else:
        terminal_p = x_tip_dim[0:3]
        terminal_q = x_tip_dim[3:7]

    return {
        "tip_p_dim": x_tip_dim[0:3],
        "tip_Q_wxyz": x_tip_dim[3:7],
        "terminal_p_dim": terminal_p,
        "terminal_Q_wxyz": terminal_q,
        "rigid_distal_p_dim": rigid_mid_p,
        "rigid_distal_Q_wxyz": rigid_mid_q,
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
    sensitivity_mode: str = "solver_response",
    solve_mode: str = "solve",
    svd_rcond: float = 1e-10,
    spectral_filter: str = "hard",
    spectral_damping: float = 0.0,
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

    solve_mode = str(solve_mode)
    if solve_mode not in ("solve", "svd"):
        raise ValueError("solve_mode must be 'solve' or 'svd'")
    sensitivity_mode = str(sensitivity_mode)
    if sensitivity_mode not in ("solver_response", "gn_svd", "residual_svd"):
        raise ValueError("sensitivity_mode must be 'solver_response', 'gn_svd', or 'residual_svd'")
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

    Jt = jnp.transpose(Jz)
    nZ = int(Jz.shape[1])
    extra_ridge = 0.0 if ridge is None else float(ridge)
    lm_lam = float(lm_stats.lam)

    sv_min = None
    sv_max = None
    effective_rank = None
    spectral_name = "none"
    spectral_lambda = 0.0
    spectral_cutoff = None
    retained_spectral_energy = None
    if sensitivity_mode == "residual_svd":
        spectral = _solve_residual_spectral(
            Jz,
            B,
            svd_rcond=float(svd_rcond),
            spectral_filter=str(spectral_filter),
            spectral_damping=float(spectral_damping),
        )
        Jr_all = Jr_z @ spectral.dz_dc
        lam_used = 0.0
        sv_min = float(spectral.sv_min)
        sv_max = float(spectral.sv_max)
        effective_rank = int(spectral.effective_rank)
        condition_number = float(spectral.condition_number) if bool(compute_conditioning) else None
        spectral_name = str(spectral.spectral_filter)
        spectral_lambda = float(spectral.spectral_lambda)
        spectral_cutoff = float(spectral.spectral_cutoff)
        retained_spectral_energy = float(spectral.retained_spectral_energy)
        method = "dense_task_residual_svd"
    else:
        H = Jt @ Jz
        if sensitivity_mode == "solver_response":
            lam_used = lm_lam + extra_ridge
            method = "dense_task_solver_response"
        else:
            lam_used = extra_ridge
            method = "dense_task_gn_without_solver_damping"
        H = H + float(lam_used) * jnp.eye(nZ, dtype=H.dtype)
        if solve_mode == "svd":
            U_svd, S, Vh = jnp.linalg.svd(H, full_matrices=False)
            sv_max_val = jnp.max(S)
            cutoff = float(svd_rcond) * sv_max_val
            keep = S > cutoff
            inv_S = jnp.where(keep, 1.0 / S, 0.0)
            U = Vh.T @ (inv_S[:, None] * (U_svd.T @ jnp.transpose(Jr_z)))
            sv_min = float(jnp.min(S))
            sv_max = float(sv_max_val)
            effective_rank = int(jnp.sum(keep))
            condition_number = float(sv_max_val / jnp.maximum(jnp.min(S), jnp.finfo(S.dtype).tiny))
        else:
            condition_number = None
            if bool(compute_conditioning):
                S = jnp.linalg.svd(H, compute_uv=False)
                sv_min = float(jnp.min(S))
                sv_max = float(jnp.max(S))
                effective_rank = int(jnp.sum(S > float(svd_rcond) * jnp.max(S)))
            U = jnp.linalg.solve(H, jnp.transpose(Jr_z))
        JT_B = Jt @ B
        Jr_all = -jnp.transpose(U) @ JT_B

    E = jnp.asarray(lm_stats.E_star, dtype=jnp.float64)
    stationarity_norm = float(jnp.linalg.norm(Jt @ E))
    if bool(compute_conditioning) and sensitivity_mode != "residual_svd":
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
    if sensitivity_mode == "residual_svd" and spectral_name == "hard" and effective_rank is not None and effective_rank < int(nZ):
        status = "truncated_svd"
    elif sensitivity_mode == "residual_svd" and spectral_name != "hard":
        status = "spectral_filtered" if status == "ok" else f"{status}|spectral_filtered"
    elif solve_mode == "svd" and effective_rank is not None and effective_rank < int(nZ):
        status = "truncated_svd"

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
        sv_min=sv_min,
        sv_max=sv_max,
        effective_rank=effective_rank,
        lm_lam=lm_lam,
        extra_ridge=extra_ridge,
        ridge_used=extra_ridge,
        solve_mode=solve_mode,
        status=str(status),
        method=str(method),
        spectral_filter=str(spectral_name),
        spectral_lambda=float(spectral_lambda),
        spectral_cutoff=spectral_cutoff,
        retained_spectral_energy=retained_spectral_energy,
    )
