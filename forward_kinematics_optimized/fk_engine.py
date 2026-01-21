from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import os, sys

# Allow running from this directory without installing as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

from advancer_nondim import apply_advancer_protrude_length
from nondim import NondimScales, x_bar_to_dim, x_dim_to_bar
from utils_nondim import unpack_z_bar_jax, pack_z_bar_jax, make_initial_guess_multi_bar_jax
from fk import compute_scales_from_flex, build_solver_params  # reuse exact FK construction logic
from equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondimJAXCached, with_coil_currents

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


@dataclass
class FKState:
    """Forward-kinematics evaluation result (for reuse by IK)."""
    tip_p_dim: Array  # (3,)
    tip_Q_wxyz: Array  # (4,)
    z_bar: Array  # packed decision vector at convergence
    scales: NondimScales
    flex_lengths: Tuple[float, ...]
    rigid_lengths: Tuple[float, ...]


class ForwardKinematicsEngine:
    """Reusable FK engine with built-in warm start.

    Design goals:
      - Keep all static configuration in the engine instance.
      - Support fast repeated calls with varying (coil_currents, L_protrude) by:
          * caching last converged z_bar
          * rescaling last solution between different nondim scales
          * reusing LM warm-start
      - Keep behavior consistent with forward_kinematics/fk.py construction logic
        by delegating solver/params assembly to fk.build_solver_params().

    Notes on warm-start and nondimensionalization:
      fk.py uses L_ref = (sum flex lengths + sum rigid lengths), which varies with L_protrude.
      We therefore rescale the cached z_bar between successive calls whenever scales change.
      Since k-arrays are initialized to zeros in the initial guess, we do not rely on their
      accuracy for warm start; we preserve them but correctness is not critical.
    """

    def __init__(
            self,
            *,
            # Geometry
            flex_lengths_tail: Sequence[float],  # L2..LN (or L1 placeholder + L2..LN; see below)
            rigid_lengths: Sequence[float],  # N values
            M_list: Sequence[int],  # N values (interval count for each flexible segment)
            # Advancer
            L1_min: float = 1e-6,
            # Materials (same semantics as fk.py CLI)
            flex_d_outer: Sequence[float] = (0.0015,),
            flex_E: Sequence[float] = (1.8e6,),
            flex_G: Sequence[float] = (0.6e6,),
            flex_rho: Sequence[float] = (970.0,),
            rigid_d_outer: Sequence[float] = (0.0015,),
            rigid_rho: Sequence[float] = (7500.0,),
            # Base pose / axis
            p0_dim: Sequence[float] = (0.0, 0.0, -0.05),
            Q0_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
            axis_body: Sequence[float] = (0.0, 0.0, 1.0),
            # Loads / actuation
            enable_gravity: bool = False,
            g_world: Sequence[float] = (0.0, 0.0, -9.81),
            enable_magnetics: bool = False,
            calib_file: Optional[str] = None,
            actuation_table_pkl: Optional[str] = None,
            m_body_list: Optional[Sequence[Sequence[float]]] = None,  # N rigid magnets: each (3,)
            # LM / Jacobian settings
            max_iter: int = 100000,
            tol: float = 1e-5,
            lm_damping: float = 1e-1,
            jac_method: str = "fwd",
    ) -> None:
        self.flex_lengths_tail = tuple(float(x) for x in flex_lengths_tail)
        self.rigid_lengths = tuple(float(x) for x in rigid_lengths)
        self.M_list = tuple(int(m) for m in M_list)
        self.L1_min = float(L1_min)

        self.flex_d_outer = tuple(float(x) for x in flex_d_outer)
        self.flex_E = tuple(float(x) for x in flex_E)
        self.flex_G = tuple(float(x) for x in flex_G)
        self.flex_rho = tuple(float(x) for x in flex_rho)
        self.rigid_d_outer = tuple(float(x) for x in rigid_d_outer)
        self.rigid_rho = tuple(float(x) for x in rigid_rho)

        self.p0_dim = jnp.asarray(p0_dim, dtype=jnp.float64).reshape(3, )
        self.Q0 = jnp.asarray(Q0_wxyz, dtype=jnp.float64).reshape(4, )
        self.axis_body = jnp.asarray(axis_body, dtype=jnp.float64).reshape(3, )

        self.enable_gravity = bool(enable_gravity)
        self.g_world = jnp.asarray(g_world, dtype=jnp.float64).reshape(3, )

        self.enable_magnetics = bool(enable_magnetics)
        self.calib_file = calib_file
        self.actuation_table_pkl = actuation_table_pkl

        if m_body_list is None:
            self.m_body_list = None
        else:
            self.m_body_list = tuple(tuple(float(v) for v in m) for m in m_body_list)

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lm_damping = float(lm_damping)
        if jac_method not in ("fwd", "rev"):
            raise ValueError("jac_method must be 'fwd' or 'rev'")
        self.jac_method = jac_method

        # Warm-start cache
        self._last_z_bar: Optional[Array] = None
        self._last_scales: Optional[NondimScales] = None
        self._last_flex_lengths: Optional[Tuple[float, ...]] = None
        self._last_rigid_lengths: Optional[Tuple[float, ...]] = None

        # FK static-cache (avoid rebuilding params/solver each IK iteration)
        self._cached_L_protrude: Optional[float] = None
        self._cached_params_static: Optional[Any] = None
        self._cached_meshes: Optional[Any] = None
        self._cached_scales: Optional[NondimScales] = None
        self._cached_flex_lengths: Optional[Tuple[float, ...]] = None

        # One reusable solver instance to share XLA compilation
        self._solver_cached = MultiSegmentEquilibriumSolverNondimJAXCached()

    # ---------------------------
    # Public API
    # ---------------------------
    def solve(
            self,
            *,
            coil_currents: Sequence[float],
            L_protrude: float,
            warm_start: bool = True,
            override_z0_bar: Optional[Array] = None,
    ) -> Tuple[Array, Any, Any, bool]:
        """Solve equilibrium FK for given (coil_currents, L_protrude).

        Args:
          coil_currents: iterable of length 8 (Supiee), in [A]
          L_protrude: total protruding length from sheath exit (flex + rigid), [m]
          warm_start: if True, uses cached previous solution as LM initial guess
          override_z0_bar: if provided, this takes precedence as the LM initial guess

        Returns:
          (z_star_bar, params, meshes, ok)
            - z_star_bar: packed decision vector at the best-so-far solution
            - params: SolverParams used in this solve
            - meshes: list[UniformMesh]
            - ok: True only when ||E|| < tol (see solver.solve_lm)
        """
        coil_currents = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1, )
        if coil_currents.size == 0:
            raise ValueError("coil_currents is empty")
        # Compute updated flexible lengths using advancer (only L1 changes)
        flex_lengths = self._compute_flex_lengths(L_protrude=float(L_protrude))

        total_L = float(sum(flex_lengths) + sum(self.rigid_lengths))
        scales_tmp = compute_scales_from_flex(
            L_ref=total_L,
            d_outer=float(self.flex_d_outer[0]),
            E=float(self.flex_E[0]),
            G=float(self.flex_G[0]),
        )

        params_static, meshes = self._get_or_build_static_params(L_protrude=float(L_protrude), scales=scales_tmp)
        # IMPORTANT: use the *cached* scales instance from params to avoid any
        # mismatch between z0 construction and residual evaluation.
        scales = params_static.scales
        params = with_coil_currents(params_static, coil_currents)

        flex_segs = list(params.flex)
        rigid_segs = list(params.rigid)

        # Build initial guess z0_bar
        if override_z0_bar is not None:
            z0_bar = jnp.asarray(override_z0_bar, dtype=jnp.float64)
        elif warm_start and self._last_z_bar is not None and self._last_scales is not None:
            z0_bar = self._rescale_z_bar(self._last_z_bar, self._last_scales, scales, self.M_list)
        else:
            z0_bar, *_ = make_initial_guess_multi_bar_jax(
                flex_segs=flex_segs,
                meshes=meshes,
                rigid_segs=rigid_segs,
                scales=scales,
                p0_dim=self.p0_dim,
                Q0=self.Q0,
                axis_body=self.axis_body,
            )

        # Solve LM equilibrium
        z_star_bar, ok = self._solver_cached.solve_lm(
            z0_bar=z0_bar,
            params=params,
            max_iter=self.max_iter,
            tol=self.tol,
            lm_damping=self.lm_damping,
            jac_method=self.jac_method,
        )

        # Update warm-start cache (we cache best-so-far even if ok==False; caller can decide)
        self._last_z_bar = z_star_bar
        self._last_scales = scales
        self._last_flex_lengths = flex_lengths
        self._last_rigid_lengths = self.rigid_lengths

        return z_star_bar, params, meshes, bool(ok)

    def solve_with_stats(
            self,
            *,
            coil_currents: Sequence[float],
            L_protrude: float,
            warm_start: bool = True,
            override_z0_bar: Optional[Array] = None,
            return_stats: bool = True,
    ) -> Tuple[Array, Any, Any, bool, Any]:
        """Like :meth:`solve`, but requests LM diagnostics.

        This is the preferred entry-point for gradient-based IK.

        Returns:
            (z_star_bar, params, meshes, ok, lm_stats)
        """
        coil_currents = jnp.asarray(coil_currents, dtype=jnp.float64).reshape(-1, )
        flex_lengths = self._compute_flex_lengths(L_protrude=float(L_protrude))
        total_L = float(sum(flex_lengths) + sum(self.rigid_lengths))
        scales_tmp = compute_scales_from_flex(
            L_ref=total_L,
            d_outer=float(self.flex_d_outer[0]),
            E=float(self.flex_E[0]),
            G=float(self.flex_G[0]),
        )

        params_static, meshes = self._get_or_build_static_params(L_protrude=float(L_protrude), scales=scales_tmp)
        scales = params_static.scales
        params = with_coil_currents(params_static, coil_currents)

        flex_segs = list(params.flex)
        rigid_segs = list(params.rigid)

        if override_z0_bar is not None:
            z0_bar = jnp.asarray(override_z0_bar, dtype=jnp.float64)
        elif warm_start and self._last_z_bar is not None and self._last_scales is not None:
            z0_bar = self._rescale_z_bar(self._last_z_bar, self._last_scales, scales, self.M_list)
        else:
            z0_bar, *_ = make_initial_guess_multi_bar_jax(
                flex_segs=flex_segs,
                meshes=meshes,
                rigid_segs=rigid_segs,
                scales=scales,
                p0_dim=self.p0_dim,
                Q0=self.Q0,
                axis_body=self.axis_body,
            )

        if return_stats:
            z_star_bar, ok, stats = self._solver_cached.solve_lm(
                z0_bar=z0_bar,
                params=params,
                max_iter=self.max_iter,
                tol=self.tol,
                lm_damping=self.lm_damping,
                jac_method=self.jac_method,
                return_stats=True,
                # FK/IK will usually handle outer logs; keep LM verbose for now
                verbose=True,
            )
        else:
            z_star_bar, ok = self._solver_cached.solve_lm(
                z0_bar=z0_bar,
                params=params,
                max_iter=self.max_iter,
                tol=self.tol,
                lm_damping=self.lm_damping,
                jac_method=self.jac_method,
            )
            stats = None

        self._last_z_bar = z_star_bar
        self._last_scales = scales
        self._last_flex_lengths = flex_lengths
        self._last_rigid_lengths = self.rigid_lengths

        return z_star_bar, params, meshes, bool(ok), stats


    def query_sites(
            self,
            *,
            z_bar: Array,
            scales: NondimScales,
    ) -> Dict[str, Array]:
        """Extract observable poses from a packed equilibrium state.

        This is a *read-only* helper intended for IK and debugging. It does not
        modify the engine state.

        Conventions (dimensional outputs):
          - tip pose: distal node of the last flexible segment (same as ``_extract_tip_pose_dim``)
          - magnet sites: *midpoints of each rigid segment* (one state per rigid segment)

        Args:
            z_bar: packed decision vector returned by :meth:`solve` / :meth:`solve_with_stats`.
            scales: nondimensional scales used for this ``z_bar`` (typically ``params.scales``).

        Returns:
            dict with keys:
              - tip_p_dim: (3,)
              - tip_Q_wxyz: (4,)
              - rigid_mid_p_dim: (N,3) where N=len(rigid_lengths)
              - rigid_mid_Q_wxyz: (N,4)

        Notes:
            In this codebase, each rigid segment is represented by a single state
            ``x_rigid`` whose pose corresponds to the rigid segment midpoint.
            Returning these poses therefore matches the “magnet sites at rigid
            midpoints” convention requested by IK.
        """
        z_bar = jnp.asarray(z_bar, dtype=jnp.float64)

        # Tip
        tip_p_dim, tip_Q_wxyz = self._extract_tip_pose_dim(z_bar, scales)

        # Rigid midpoints (one state per rigid segment)
        _x_nodes_list_bar, _k_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar, M_list=self.M_list)

        def _one_rigid_to_dim(xR_bar: Array):
            xR_dim = x_bar_to_dim(xR_bar, scales)
            p = xR_dim[0:3]
            Q = xR_dim[3:7]
            return p, Q

        if len(x_rigid_list_bar) == 0:
            rigid_p = jnp.zeros((0, 3), dtype=jnp.float64)
            rigid_Q = jnp.zeros((0, 4), dtype=jnp.float64)
        else:
            p_list = []
            q_list = []
            for xR_bar in x_rigid_list_bar:
                p, Q = _one_rigid_to_dim(xR_bar)
                p_list.append(p)
                q_list.append(Q)
            rigid_p = jnp.stack(p_list, axis=0)
            rigid_Q = jnp.stack(q_list, axis=0)

        return {
            'tip_p_dim': tip_p_dim,
            'tip_Q_wxyz': tip_Q_wxyz,
            'rigid_mid_p_dim': rigid_p,
            'rigid_mid_Q_wxyz': rigid_Q,
        }

    def reset_warm_start(self) -> None:
        """Clear cached warm-start state."""
        self._last_z_bar = None
        self._last_scales = None
        self._last_flex_lengths = None
        self._last_rigid_lengths = None

        # Do not clear static-cache by default; callers may still want compilation reuse.

    def reset_static_cache(self) -> None:
        """Clear cached static FK objects (params/meshes) so that they are rebuilt on next call."""
        self._cached_L_protrude = None
        self._cached_params_static = None
        self._cached_meshes = None
        self._cached_scales = None
        self._cached_flex_lengths = None

    # ---------------------------
    # Internals
    # ---------------------------
    def _compute_flex_lengths(self, *, L_protrude: float) -> Tuple[float, ...]:
        """Return full flex lengths [L1, L2, ... LN] given L_protrude.

        Accepts either N-1 tail list (L2..LN) or N list (L1 placeholder + tail).
        """
        # If user passed N-1 values, construct a placeholder list for advancer.
        if len(self.flex_lengths_tail) == len(self.rigid_lengths) - 1:
            flex_in = list(self.flex_lengths_tail)  # interpreted as L2..LN
        elif len(self.flex_lengths_tail) == len(self.rigid_lengths):
            flex_in = list(self.flex_lengths_tail)  # interpreted as [L1_placeholder, L2..LN]
        else:
            raise ValueError(
                f"flex_lengths_tail must have N or N-1 values where N=len(rigid_lengths)={len(self.rigid_lengths)}; "
                f"got {len(self.flex_lengths_tail)}"
            )

        flex_out = apply_advancer_protrude_length(
            flex_lengths_in=flex_in,
            rigid_lengths=list(self.rigid_lengths),
            L_protrude=float(L_protrude),
            L1_min=self.L1_min,
        )
        return tuple(float(x) for x in flex_out)

    def _get_or_build_static_params(self, *, L_protrude: float, scales: NondimScales):
        """Return (params_static, meshes), caching by L_protrude.

        We treat the following as *static* within a typical IK run:
          - discretization (M_list)
          - geometry/materials
          - L_protrude (thus flex_lengths and scales)

        With fixed L_protrude, we can build SolverParams and meshes once, then only
        update coil currents via :func:`with_coil_currents` on each FK call.
        """
        # Cache hit
        if (self._cached_params_static is not None) and (self._cached_meshes is not None) and (self._cached_L_protrude is not None):
            if float(L_protrude) == float(self._cached_L_protrude):
                return self._cached_params_static, self._cached_meshes

        flex_lengths = self._compute_flex_lengths(L_protrude=float(L_protrude))
        params, meshes = build_solver_params(
            flex_lengths=list(flex_lengths),
            rigid_lengths=list(self.rigid_lengths),
            M_list=list(self.M_list),
            flex_d_outer=list(self.flex_d_outer),
            flex_E=list(self.flex_E),
            flex_G=list(self.flex_G),
            flex_rho=list(self.flex_rho),
            rigid_d_outer=list(self.rigid_d_outer),
            rigid_rho=list(self.rigid_rho),
            p0_dim=self.p0_dim,
            Q0=self.Q0,
            axis_body=self.axis_body,
            enable_gravity=self.enable_gravity,
            g_world=self.g_world,
            enable_magnetics=self.enable_magnetics,
            calib_file=self.calib_file,
            actuation_table_pkl=self.actuation_table_pkl,
            # placeholder currents to keep leaf shapes stable
            coil_currents=jnp.zeros((8,), dtype=jnp.float64),
            m_body_list=self.m_body_list,
            scales=scales,
        )

        self._cached_L_protrude = float(L_protrude)
        self._cached_params_static = params
        self._cached_meshes = meshes
        self._cached_scales = scales
        self._cached_flex_lengths = tuple(float(x) for x in flex_lengths)
        return params, meshes

    def _extract_tip_pose_dim(self, z_bar: Array, scales: NondimScales) -> Tuple[Array, Array]:
        """Extract tip position/quaternion (dim) from packed z_bar.

        Tip is defined as the distal node of the last flexible segment.
        """
        x_nodes_list_bar, _, _ = unpack_z_bar_jax(z_bar, M_list=self.M_list)
        x_tip_bar = x_nodes_list_bar[-1][-1]  # (13,)
        x_tip_dim = x_bar_to_dim(x_tip_bar, scales)
        tip_p_dim = x_tip_dim[0:3]
        tip_Q_wxyz = x_tip_dim[3:7]
        return tip_p_dim, tip_Q_wxyz

    @staticmethod
    def _rescale_z_bar(
            z_bar_old: Array,
            s_old: NondimScales,
            s_new: NondimScales,
            M_list: Sequence[int],
    ) -> Array:
        """Rescale cached z_bar from old scales to new scales.

        We rescale:
          - all x-nodes (flex) and rigid x-states using x_bar_to_dim/x_dim_to_bar
        We keep k-arrays unchanged (good enough for warm start in this codebase).
        """
        x_nodes_list_bar, k_array_list_bar, x_rigid_list_bar = unpack_z_bar_jax(z_bar_old, M_list=M_list)

        x_nodes_list_bar_new = []
        for x_nodes_bar in x_nodes_list_bar:
            # x_nodes_bar: (M+1, 13)
            x_dim = jax.vmap(lambda x: x_bar_to_dim(x, s_old))(x_nodes_bar)
            x_bar_new = jax.vmap(lambda x: x_dim_to_bar(x, s_new))(x_dim)
            x_nodes_list_bar_new.append(x_bar_new)

        x_rigid_list_bar_new = []
        for xR_bar in x_rigid_list_bar:
            xR_dim = x_bar_to_dim(xR_bar, s_old)
            xR_new = x_dim_to_bar(xR_dim, s_new)
            x_rigid_list_bar_new.append(xR_new)

        # Keep k arrays unchanged
        return pack_z_bar_jax(x_nodes_list_bar_new, k_array_list_bar, x_rigid_list_bar_new)


if __name__ == "__main__":
    engine = ForwardKinematicsEngine(
        flex_lengths_tail=[0.03, 0.03],  # 这里按你的使用：L2..LN
        rigid_lengths=[0.003, 0.003, 0.003],
        M_list=[10, 5, 5],
        enable_gravity=True,
        enable_magnetics=True,
        calib_file="/abs/path/calibration.json",
        m_body_list=[(0.005, 0, 0), (0, 0, 0.005), (0, 0, -0.005)],
        p0_dim=(0, 0, 0),
        Q0_wxyz=(0.70710678, 0, 0.70710678, 0),
        axis_body=(0, 0, 1),
        # 其余材料/LM参数保持默认或按需传入
    )

    z_star_bar, params, meshes, ok = engine.solve(
        coil_currents=[0, 0, 0, 0, 0, 0, 0, 0],
        L_protrude=0.13,
        warm_start=True,
    )

    # Example: rescale cached state between identical scales (no-op)
    _ = engine._rescale_z_bar(z_bar_old=z_star_bar, s_old=params.scales, s_new=params.scales, M_list=engine.M_list)
    obs = engine.query_sites(z_bar=z_star_bar, scales=params.scales)
    print("tip_p_dim:", obs["tip_p_dim"])
