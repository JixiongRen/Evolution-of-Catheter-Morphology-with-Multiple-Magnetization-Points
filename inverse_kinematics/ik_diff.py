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
from forward_kinematics_optimized.equilibrium_solver_nondim import residual_bar_zI, SolverParams, LMStats

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


def _p_of_z_params(z_bar: Array, params: SolverParams) -> Array:
    """Helper for differentiating tip position wrt z (params carries scales/M_list)."""
    return extract_tip_p_dim(z_bar, M_list=params.M_list, scales=params.scales)


_DP_DZ_REV = jax.jit(jax.jacrev(_p_of_z_params, argnums=0))


def extract_tip_p_dim(z_bar: Array, *, M_list: Sequence[int], scales) -> Array:
    """Extract the catheter tip position (SI units) from packed ``z_bar``.

    Important
    ---------
    In this codebase, each segment i contains
      - a flexible piece (discretized nodes), followed by
      - a rigid piece whose *distal* state is stored as ``xR``.

    The physical catheter "tip" is therefore the *distal end of the last rigid
    piece*, i.e. the last ``xR`` block in ``z_bar``.

    If you instead use the last flexible node, you will systematically ignore
    the rigid lengths (and may observe almost no lateral motion), which breaks
    IK.
    """
    _x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=M_list)
    x_tip_bar = x_rigid_list_bar[-1]  # (13,) rigid distal state
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
    I0 = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1,)
    z0 = jnp.asarray(z_star_bar, dtype=jnp.float64).reshape(-1,)

    # 1) Reuse LM Jacobian dE/dz
    if lm_stats.J is None:
        raise ValueError("lm_stats.J is None. Call solve_lm(..., return_stats=True).")
    Jz = jnp.asarray(lm_stats.J, dtype=jnp.float64)

    # 2) Compute JI = dE/dI using a cached compiled transform
    if jac_method_I == "fwd":
        JI = _JAC_E_WRT_I_FWD(z0, I0, params)
    elif jac_method_I == "rev":
        JI = _JAC_E_WRT_I_REV(z0, I0, params)
    else:
        raise ValueError("jac_method_I must be 'fwd' or 'rev'")

    # 3) dp/dz (cached; only 3 rows, reverse-mode is ideal)
    dp_dz = _DP_DZ_REV(z0, params)  # (3, nZ)

    # 4) Solve (J^T J + lam I) U = (dp/dz)^T for U (nZ x 3)
    lam_used = float(lm_stats.lam)
    if ridge is not None:
        lam_used = float(lam_used + ridge)

    Jt = jnp.transpose(Jz)
    H = Jt @ Jz
    nZ = H.shape[0]
    H = H + lam_used * jnp.eye(nZ, dtype=H.dtype)

    # Three RHS in one solve
    U = jnp.linalg.solve(H, jnp.transpose(dp_dz))  # (nZ, 3)

    # 5) Jp = - U^T (J^T JI)
    JT_B = Jt @ JI  # (nZ, nI)
    Jp = - jnp.transpose(U) @ JT_B  # (3, nI)

    p_tip = _p_of_z_params(z0, params)
    return TipJacobianResult(
        p_tip_dim=p_tip,
        J_p_I=Jp,
        J_E_z=Jz,
        J_E_I=JI,
        lam=float(lam_used),
    )
