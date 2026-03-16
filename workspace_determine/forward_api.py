
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
import jax.numpy as jnp

from forward_kinematics_nondim_optimized.nondim import x_bar_to_dim
from forward_kinematics_nondim_optimized.basics_nondim import quat_normalize
from forward_kinematics_nondim_optimized.equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondimJAX, SolverParams
from forward_kinematics_nondim_optimized.utils_nondim import make_initial_guess_multi_bar_jax, unpack_z_bar_jax

Array = jnp.ndarray


@dataclass
class ForwardMeta:
    """Diagnostics for a forward solve (currents -> equilibrium -> tip pose)."""
    success: bool
    n_iter: int
    final_E_raw: float
    final_E_scaled: float
    message: str


@dataclass
class TipPose:
    """Tip pose in SI units."""
    p: np.ndarray  # (3,)
    q: np.ndarray  # (4,) [w,x,y,z], normalized


class ForwardModel:
    """
    Thin wrapper that turns your existing equilibrium solver into a reusable forward map:

        I (8,) -> solve_lm -> z* -> tip pose

    Notes
    -----
    - This class intentionally does not rebuild meshes on each call.
    - Warm-start: pass z0_bar; if omitted, uses last_z_bar if available, else a geometric initial guess.
    """

    def __init__(
        self,
        params: SolverParams,
        meshes: List[Any],
        *,
        max_iter: int = 50,
        lm_lambda_init: float = 1e-2,
        verbose: bool = False,
    ):
        self.params = params
        self.meshes = meshes
        self.verbose = verbose
        self.solver = MultiSegmentEquilibriumSolverNondimJAX(params=params)

        self.max_iter = int(max_iter)
        self.lm_lambda_init = float(lm_lambda_init)

        self._last_z_bar: Optional[Array] = None

    def make_geometric_initial_guess(self) -> Array:
        z0_bar, *_ = make_initial_guess_multi_bar_jax(
            flex_segs=self.params.flex,
            meshes=self.meshes,
            rigid_segs=self.params.rigid,
            scales=self.params.scales,
            p0_dim=self.params.p0_bar * float(self.params.scales.L_ref),
            Q0=self.params.Q0,
            axis_body=self.params.flex[0].v_star if len(self.params.flex) > 0 else jnp.array([0.0, 0.0, 1.0]),
        )
        return z0_bar

    def _set_currents(self, I: np.ndarray) -> SolverParams:
        I = jnp.asarray(I, dtype=jnp.float64).reshape(8,)
        # Prefer dataclass-like replace() if present
        if hasattr(self.params, "replace"):
            return self.params.replace(coil_currents=I)
        # Fallback: reconstruct from __dict__
        d = dict(self.params.__dict__)
        d["coil_currents"] = I
        return SolverParams(**d)

    def solve(self, I: np.ndarray, *, z0_bar: Optional[Array] = None) -> Tuple[TipPose, Array, ForwardMeta]:
        if z0_bar is None:
            z0_bar = self._last_z_bar if self._last_z_bar is not None else self.make_geometric_initial_guess()

        params_I = self._set_currents(I)
        solver = MultiSegmentEquilibriumSolverNondimJAX(params=params_I)

        try:
            # Current equilibrium_solver_nondim.py signature:
            #   solve_lm(z0_bar, *, max_iter=..., tol=..., lm_damping=...)
            z_star, success = solver.solve_lm(
                z0_bar,
                tol=1e-5,
                max_iter=self.max_iter,
                lm_damping=self.lm_lambda_init,
                jac_method="fwd",
            )
            n_iter = -1
            msg = "ok" if success else "not_converged"
        except TypeError:
            # Fallback: if your local signature differs, try minimal call.
            ret = solver.solve_lm(z0_bar)
            if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], (bool, np.bool_)):
                z_star, success = ret[0], bool(ret[1])
                msg = "ok" if success else "not_converged"
            else:
                z_star, success, msg = ret, True, "ok"
            n_iter = -1
        except Exception as e:
            if self.verbose:
                raise
            tip = TipPose(p=np.full((3,), np.nan), q=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
            meta = ForwardMeta(False, -1, float("inf"), float("inf"), f"solve_lm exception: {type(e).__name__}: {e}")
            return tip, z0_bar, meta

        # Diagnostics: residual norm (nondimensional)
        try:
            E = np.asarray(solver.residual_jit(z_star))
            final_E_raw = float(np.linalg.norm(E))
        except Exception:
            final_E_raw = float("nan")
        final_E_scaled = float("nan")

        # Tip pose = distal end of last rigid
        _, _, x_rigid_list_bar = unpack_z_bar_jax(z_star, M_list=params_I.M_list)
        x_tip_bar = x_rigid_list_bar[-1]
        x_tip_dim = x_bar_to_dim(x_tip_bar, params_I.scales)

        p_tip = np.asarray(x_tip_dim[0:3], dtype=float)
        q_tip = np.asarray(quat_normalize(jnp.asarray(x_tip_dim[3:7], dtype=jnp.float64)), dtype=float)

        self._last_z_bar = z_star
        meta = ForwardMeta(
            success=bool(success),
            n_iter=int(n_iter),
            final_E_raw=final_E_raw,
            final_E_scaled=final_E_scaled,
            message=msg,
        )
        return TipPose(p=p_tip, q=q_tip), z_star, meta
