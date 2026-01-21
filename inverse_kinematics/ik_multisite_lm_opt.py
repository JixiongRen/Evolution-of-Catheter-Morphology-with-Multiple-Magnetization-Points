"""Single-layer LM inverse kinematics (position-only) over (I, x).

Plan A (single-layer joint LM in control space u):
  - Do NOT modify FK.
  - Use ForwardKinematicsEngine.solve_with_stats(...) for FK.
  - Use ForwardKinematicsEngine.query_sites(...) for observables.
  - Optimize u = [u_I..., u_x], with
        I = I_max * tanh(u_I)
        x = x_min + (x_max-x_min) * sigmoid(u_x)

This version constrains ONLY the distal tip position.

Key scaling / stabilization changes (2026-01):
  1) Position residual is normalized by sigma_p:
       r_p = (p_tip - p_des) / sigma_p
     (default sigma_p = eps_p)

  2) w_I, w_x are interpreted as inverse standard scales:
       w_I = 1/sigma_I , w_x = 1/sigma_x
     so r_I = w_I * I, r_x = w_x * (x - x_ref)

  3) Add smoothness / coupling residuals relative to previous accepted state:
       r_dI = w_dI * (I - I_prev)
       r_dx = w_dx * (x - x_prev)
       r_Ix = w_Ix * (I - I_prev) / max(|x - x_prev|, dx_floor)
     where w_* are also inverse scales (1/sigma_*).

  4) Replace parameter-space step_max with physical step limits:
       per-coil |ΔI_i| <= step_I_max
       |Δx|     <= step_x_max
     We clip in (I,x) and then map back to u for a consistent LM linearization.

Jacobian strategy
-----------------
  - dp/dI: LM-adjoint (compute_dp_dI_via_lm_adjoint)
  - dp/dx: finite difference (two extra FK solves per outer iteration)

Notes
-----
* This script is intentionally verbose for debugging.
* It reuses fk.py CLI flags for geometry/material/actuation configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from ik_diff import compute_dp_dI_via_lm_adjoint

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


def _parse_vec3(s: str) -> Array:
    xs = [float(x.strip()) for x in str(s).split(",")]
    if len(xs) != 3:
        raise ValueError(f"Expected 3 values, got: {s}")
    return jnp.asarray(xs, dtype=jnp.float64)


def _make_run_dir(out_dir: str, run_name: Optional[str]) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if run_name is None or str(run_name).strip() == "":
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{ts}"
    run = (root / run_name).resolve()
    run.mkdir(parents=True, exist_ok=True)
    return run


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()

    # IK targets
    p.add_argument("--p_des", type=str, required=True, help="Desired tip position [m], format x,y,z")
    p.add_argument("--eps_p", type=float, default=1e-3, help="Stop if ||p_tip - p_des|| <= eps_p [m].")
    p.add_argument(
        "--sigma_p",
        type=float,
        default=None,
        help="Position residual scale sigma_p [m]. r_p=(p_tip-p_des)/sigma_p. Default: sigma_p=eps_p.",
    )

    # Optimization variables: currents and insertion depth
    p.add_argument("--I_min", type=float, default=None, help="Lower current bound. If not set, uses -I_max.")
    p.add_argument("--I_max", type=float, default=50.0, help="Symmetric current bound: I in [-I_max, I_max]")
    p.add_argument("--x0", type=float, default=None, help="Initial insertion depth x [m]. If not set, uses --L_protrude.")
    p.add_argument("--x_min", type=float, default=None, help="Lower bound for insertion depth x [m]")
    p.add_argument("--x_max", type=float, default=None, help="Upper bound for insertion depth x [m]")
    p.add_argument("--x_ref", type=float, default=None, help="Reference insertion depth for regularization. Default: x0.")

    # Outer LM settings
    p.add_argument("--outer_max_iter", type=int, default=70, help="Max outer LM iterations")
    p.add_argument("--lam_init", type=float, default=1e-2, help="Initial LM damping (outer)")
    p.add_argument("--lam_up", type=float, default=10.0, help="Multiply lambda on reject")
    p.add_argument("--lam_down", type=float, default=0.3, help="Multiply lambda on good accept")
    p.add_argument("--rho_good", type=float, default=0.75, help="rho threshold for decreasing lambda")
    p.add_argument("--rho_bad", type=float, default=0.25, help="rho threshold for increasing lambda")

    # Physical step limits (replace step_max)
    p.add_argument(
        "--step_I_max",
        type=float,
        default=8.0,
        help="Max per-coil current step |ΔI_i| [A] per outer iteration (physical clipping).",
    )
    p.add_argument(
        "--step_x_max",
        type=float,
        default=20e-3,
        help="Max insertion step |Δx| [m] per outer iteration (physical clipping).",
    )

    # Regularization weights (now interpreted as inverse scales)
    p.add_argument(
        "--w_I",
        type=float,
        default=0.05,  # ~ 1/20A
        help="Inverse current scale: w_I = 1/sigma_I (A^-1). Residual r_I = w_I * I.",
    )
    p.add_argument(
        "--w_x",
        type=float,
        default=20.0,  # ~ 1/0.05m
        help="Inverse insertion scale: w_x = 1/sigma_x (m^-1). Residual r_x = w_x * (x - x_ref).",
    )

    # Smoothness / coupling residuals (inverse scales)
    p.add_argument(
        "--w_dI",
        type=float,
        default=0.2,  # ~ 1/5A
        help="Inverse ΔI scale: w_dI = 1/sigma_dI (A^-1). Residual r_dI = w_dI * (I - I_prev).",
    )
    p.add_argument(
        "--w_dx",
        type=float,
        default=200.0,  # ~ 1/0.005m
        help="Inverse Δx scale: w_dx = 1/sigma_dx (m^-1). Residual r_dx = w_dx * (x - x_prev).",
    )
    p.add_argument(
        "--w_Ix",
        type=float,
        default=5e-4,  # ~ 1/2000 (A/m)
        help="Inverse slope scale: w_Ix = 1/sigma_Ix ((A/m)^-1). Residual r_Ix = w_Ix*(I-I_prev)/max(|x-x_prev|,dx_floor).",
    )
    p.add_argument(
        "--dx_floor",
        type=float,
        default=1e-3,
        help="Floor for |x-x_prev| in r_Ix denominator [m] to avoid blow-up when dx≈0.",
    )

    # Sensitivity knobs
    p.add_argument("--ridge_adjoint", type=float, default=0.0, help="Extra ridge added to inner LM damping when computing dp/dI")
    p.add_argument("--fd_x_eps", type=float, default=1e-3, help="Finite difference step for dp/dx [m]")

    # Robustness / FK accept
    p.add_argument(
        "--tol_E_good",
        type=float,
        default=None,
        help="Treat equilibrium as GOOD if ||E|| <= tol_E_good. Default: use fk --tol",
    )
    p.add_argument("--tol_E_weak", type=float, default=1e-2, help="Treat equilibrium as WEAK but usable if ||E|| <= tol_E_weak")
    p.add_argument("--tol_E_bad", type=float, default=1e-1, help="Treat equilibrium as BAD/REJECT if ||E|| > tol_E_bad")
    p.add_argument("--penalty_bad", type=float, default=1e3, help="Penalty added when equilibrium is BAD/REJECT or tip is NaN")

    # IO
    p.add_argument("--out_dir", type=str, default="ik_out", help="Output directory")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name (subfolder)")
    p.add_argument("--print_devices", action="store_true", help="Print JAX devices at start")

    # seed
    p.add_argument("--seed", type=int, default=0, help="Random seed used for initializing currents (I0).")
    return p


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))


def _u_to_Ix(u: Array, *, I_max: float, x_min: float, x_max: float) -> Tuple[Array, float, Array, float]:
    """Map unconstrained parameter u -> (I, x) and return derivatives.

    Returns:
      I: (nI,)
      x: float
      dI_du: (nI,) where dI/du_i
      dx_du: float where dx/du_x
    """
    u = jnp.asarray(u, dtype=jnp.float64).reshape(-1,)
    u_I = u[:-1]
    u_x = u[-1]

    I = float(I_max) * jnp.tanh(u_I)
    dI_du = float(I_max) * (1.0 - jnp.tanh(u_I) ** 2)

    s = _sigmoid(u_x)
    x = float(x_min) + (float(x_max) - float(x_min)) * float(s)
    dx_du = (float(x_max) - float(x_min)) * float(s * (1.0 - s))
    return I, float(x), dI_du, float(dx_du)


def _Ix_to_u(I: Array, x: float, *, I_max: float, x_min: float, x_max: float) -> Array:
    """Initialize u from a feasible (I,x)."""
    I = jnp.asarray(I, dtype=jnp.float64).reshape(-1,)
    I = jnp.clip(I, -float(I_max) + 1e-12, float(I_max) - 1e-12)
    u_I = jnp.arctanh(I / float(I_max))

    # x mapping uses sigmoid; invert safely
    t = (float(x) - float(x_min)) / max(float(x_max) - float(x_min), 1e-12)
    t = float(jnp.clip(t, 1e-9, 1.0 - 1e-9))
    u_x = jnp.log(t / (1.0 - t))
    return jnp.concatenate([u_I, jnp.asarray([u_x], dtype=jnp.float64)], axis=0)


def _classify_equilibrium(*, ok_strict: bool, normE: float, tol_good: float, tol_weak: float, tol_bad: float) -> str:
    if not jnp.isfinite(normE):
        return "REJECT"
    if ok_strict or (normE <= tol_good):
        return "GOOD"
    if normE <= tol_weak:
        return "WEAK"
    if normE > tol_bad:
        return "REJECT"
    return "BAD"


@dataclass
class EvalPack:
    """Bundle of FK evaluation used by outer LM."""

    I: Array
    x: float
    p_tip: Array
    e_p: Array  # (m)
    err: float  # (m)
    r: Array
    cost: float
    penalty: float
    cls: str
    ok_strict: bool
    normE: float
    z_star_bar: Array
    params: Any
    lm_stats: Any


def _compute_residual_vector(
    *,
    p_tip: Array,
    p_des: Array,
    sigma_p: float,
    I: Array,
    x: float,
    w_I: float,
    w_x: float,
    x_ref: float,
    I_prev: Array,
    x_prev: float,
    w_dI: float,
    w_dx: float,
    w_Ix: float,
    dx_floor: float,
    penalty: float,
) -> Tuple[Array, float, Array, float]:
    """Build stacked residual r and its cost.

    Residual order (must match Jacobian assembly):
      r = [
        r_p(3),
        r_I(nI),
        r_x(1),
        r_dI(nI),
        r_dx(1),
        r_Ix(nI),
        r_pen(1),
      ]
    """
    p_tip = jnp.asarray(p_tip, dtype=jnp.float64).reshape(3,)
    p_des = jnp.asarray(p_des, dtype=jnp.float64).reshape(3,)
    I = jnp.asarray(I, dtype=jnp.float64).reshape(-1,)
    I_prev = jnp.asarray(I_prev, dtype=jnp.float64).reshape(-1,)

    sigma_p = float(sigma_p)
    if sigma_p <= 0:
        raise ValueError("sigma_p must be > 0")

    e_p = (p_tip - p_des).reshape(3,)  # meters (for reporting)
    r_p = e_p / sigma_p

    # Regularization (w_* are inverse scales)
    r_I = float(w_I) * I
    r_x = jnp.asarray([float(w_x) * (float(x) - float(x_ref))], dtype=jnp.float64)

    # Smoothness relative to last accepted
    dI = I - I_prev
    dx = float(x) - float(x_prev)
    r_dI = float(w_dI) * dI
    r_dx = jnp.asarray([float(w_dx) * dx], dtype=jnp.float64)

    # Coupling slope (dI/dx) residual: per coil
    denom = max(abs(dx), float(dx_floor))
    r_Ix = float(w_Ix) * (dI / denom)

    r_pen = jnp.asarray([jnp.sqrt(float(max(penalty, 0.0)))], dtype=jnp.float64)

    r = jnp.concatenate([r_p, r_I, r_x, r_dI, r_dx, r_Ix, r_pen], axis=0)
    cost = 0.5 * float(jnp.dot(r, r))
    err = float(jnp.linalg.norm(e_p))
    return r, float(cost), e_p, float(err)


def _fk_eval(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    x: float,
    p_des: Array,
    sigma_p: float,
    tol_good: float,
    tol_weak: float,
    tol_bad: float,
    penalty_bad: float,
    w_I: float,
    w_x: float,
    x_ref: float,
    I_prev: Array,
    x_prev: float,
    w_dI: float,
    w_dx: float,
    w_Ix: float,
    dx_floor: float,
    override_z0_bar: Optional[Array] = None,
) -> EvalPack:
    z_star_bar, params, _meshes, ok, lm_stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(x),
        warm_start=True,
        override_z0_bar=override_z0_bar,
        return_stats=True,
    )

    normE = float(lm_stats.normE) if lm_stats is not None else float("nan")
    ok_strict = bool(lm_stats.ok_strict) if lm_stats is not None else bool(ok)
    cls = _classify_equilibrium(ok_strict=ok_strict, normE=normE, tol_good=tol_good, tol_weak=tol_weak, tol_bad=tol_bad)

    # Tip position (SI) via FK engine observation interface
    obs = engine.query_sites(z_bar=z_star_bar, scales=params.scales)
    p_tip = jnp.asarray(obs["tip_p_dim"], dtype=jnp.float64)

    penalty = 0.0
    if (not jnp.all(jnp.isfinite(p_tip))) or (cls in ("BAD", "REJECT")):
        penalty = float(penalty_bad)

    r, cost, e_p, err = _compute_residual_vector(
        p_tip=p_tip,
        p_des=p_des,
        sigma_p=float(sigma_p),
        I=I,
        x=float(x),
        w_I=float(w_I),
        w_x=float(w_x),
        x_ref=float(x_ref),
        I_prev=I_prev,
        x_prev=float(x_prev),
        w_dI=float(w_dI),
        w_dx=float(w_dx),
        w_Ix=float(w_Ix),
        dx_floor=float(dx_floor),
        penalty=float(penalty),
    )

    return EvalPack(
        I=jnp.asarray(I, dtype=jnp.float64),
        x=float(x),
        p_tip=jnp.asarray(p_tip, dtype=jnp.float64),
        e_p=jnp.asarray(e_p, dtype=jnp.float64),
        err=float(err),
        r=r,
        cost=float(cost),
        penalty=float(penalty),
        cls=str(cls),
        ok_strict=bool(ok_strict),
        normE=float(normE),
        z_star_bar=jnp.asarray(z_star_bar, dtype=jnp.float64),
        params=params,
        lm_stats=lm_stats,
    )


def _compute_dp_dx_fd(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    x: float,
    h: float,
    z0_bar: Optional[Array],
    clip_bounds: Tuple[float, float],
) -> Array:
    """Finite difference dp_tip/dx along x, returns (3,)."""
    x_min, x_max = clip_bounds
    h = float(h)
    if h <= 0.0:
        raise ValueError("fd step h must be > 0")
    xp = min(float(x_max), float(x) + h)
    xm = max(float(x_min), float(x) - h)
    if abs(xp - xm) < 1e-12:
        return jnp.zeros((3,), dtype=jnp.float64)

    zp, params_p, _, _, _ = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(xp),
        warm_start=True,
        override_z0_bar=z0_bar,
        return_stats=True,
    )
    zm, params_m, _, _, _ = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(xm),
        warm_start=True,
        override_z0_bar=z0_bar,
        return_stats=True,
    )

    pp = jnp.asarray(engine.query_sites(z_bar=zp, scales=params_p.scales)["tip_p_dim"], dtype=jnp.float64)
    pm = jnp.asarray(engine.query_sites(z_bar=zm, scales=params_m.scales)["tip_p_dim"], dtype=jnp.float64)
    dpdx = (pp - pm) / float(xp - xm)
    return jnp.asarray(dpdx, dtype=jnp.float64)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.print_devices:
        print("[IK-LM] JAX devices:", jax.devices())

    run_dir = _make_run_dir(args.out_dir, args.run_name)
    cfg = fk_cli.load_config_file(args.config)

    def _get(name: str, default):
        v_cli = getattr(args, name, None)
        if v_cli is not None and (not isinstance(v_cli, bool) or v_cli is True or default is False):
            return v_cli
        return cfg.get(name, default)

    # Parse FK-style args
    flex_lengths = fk_cli._parse_csv_floats(_get("flex_lengths", args.flex_lengths))
    rigid_lengths = fk_cli._parse_csv_floats(_get("rigid_lengths", args.rigid_lengths))
    M_list = fk_cli._parse_csv_ints(_get("M_list", args.M_list))

    # Materials
    flex_d_outer = fk_cli._parse_csv_floats(_get("flex_d_outer", args.flex_d_outer))
    flex_E = fk_cli._parse_csv_floats(_get("flex_E", args.flex_E))
    flex_G = fk_cli._parse_csv_floats(_get("flex_G", args.flex_G))
    flex_rho = fk_cli._parse_csv_floats(_get("flex_rho", args.flex_rho))
    rigid_d_outer = fk_cli._parse_csv_floats(_get("rigid_d_outer", args.rigid_d_outer))
    rigid_rho = fk_cli._parse_csv_floats(_get("rigid_rho", args.rigid_rho))

    # Pose / axis
    p0_dim = _parse_vec3(_get("p0", args.p0))
    Q0 = jnp.asarray([float(x) for x in str(_get("Q0", args.Q0)).split(",")], dtype=jnp.float64)
    axis_body = _parse_vec3(_get("axis_body", args.axis_body))

    enable_gravity = bool(_get("enable_gravity", args.enable_gravity))
    g_world = _parse_vec3(_get("g_world", args.g_world))

    enable_magnetics = bool(_get("enable_magnetics", args.enable_magnetics))
    calib_file = _get("calib_file", args.calib_file)
    actuation_table_pkl = _get("actuation_table_pkl", args.actuation_table_pkl)

    m_body_list = None
    if _get("m_body_list", args.m_body_list) is not None:
        parts = [t.strip() for t in str(_get("m_body_list", args.m_body_list)).split(";") if t.strip()]
        m_body_list = [tuple(float(v) for v in p.split(",")) for p in parts]

    # Guardrail: magnetics requested but missing required inputs
    if enable_magnetics:
        if calib_file is None:
            raise ValueError("--enable_magnetics was set but --calib_file is missing. Without calibration, I has no effect.")
        if m_body_list is None:
            raise ValueError("--enable_magnetics was set but --m_body_list is missing. Without magnet moments, I has no effect.")

    # Decision variable bounds
    I_max = float(args.I_max)
    if I_max <= 0:
        raise ValueError("--I_max must be > 0")

    # x0 defaults to --L_protrude if present
    x0 = args.x0
    if x0 is None:
        x0 = _get("L_protrude", args.L_protrude)
    if x0 is None:
        raise ValueError("Provide --x0 (or --L_protrude as an alias for initial x0).")
    x0 = float(x0)

    # Compute physically valid minimum protrude (tail flex + rigid + L1_min)
    L1_min = float(_get("L1_min", args.L1_min))
    if len(flex_lengths) == len(rigid_lengths) - 1:
        flex_tail = flex_lengths
    elif len(flex_lengths) == len(rigid_lengths):
        flex_tail = flex_lengths[1:]
    else:
        raise ValueError(
            f"flex_lengths must have N or N-1 values where N=len(rigid_lengths)={len(rigid_lengths)}; got {len(flex_lengths)}"
        )
    x_phys_min = float(sum(flex_tail) + sum(rigid_lengths) + L1_min)

    x_min = float(args.x_min) if args.x_min is not None else max(x_phys_min, x0 - 0.03)
    x_max = float(args.x_max) if args.x_max is not None else max(x_min + 1e-6, x0 + 0.03)
    if x_min < x_phys_min:
        x_min = x_phys_min
    if x_max <= x_min:
        raise ValueError("x_max must be > x_min")
    if not (x_min <= x0 <= x_max):
        print(f"[IK-LM] WARNING: x0={x0:.6f} outside [{x_min:.6f},{x_max:.6f}], clipping.")
        x0 = min(max(x0, x_min), x_max)

    x_ref = float(args.x_ref) if args.x_ref is not None else float(x0)

    # Build engine
    engine = ForwardKinematicsEngine(
        flex_lengths_tail=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        L1_min=L1_min,
        flex_d_outer=flex_d_outer,
        flex_E=flex_E,
        flex_G=flex_G,
        flex_rho=flex_rho,
        rigid_d_outer=rigid_d_outer,
        rigid_rho=rigid_rho,
        p0_dim=p0_dim,
        Q0_wxyz=Q0,
        axis_body=axis_body,
        enable_gravity=enable_gravity,
        g_world=g_world,
        enable_magnetics=enable_magnetics,
        calib_file=calib_file,
        actuation_table_pkl=actuation_table_pkl,
        m_body_list=m_body_list,
        max_iter=int(_get("max_iter", args.max_iter)),
        tol=float(_get("tol", args.tol)),
        lm_damping=float(_get("lm_damping", args.lm_damping)),
        jac_method=str(_get("jac_method", args.jac_method)),
    )

    # Targets
    p_des = _parse_vec3(args.p_des)
    eps_p = float(args.eps_p)
    sigma_p = float(args.sigma_p) if args.sigma_p is not None else float(eps_p)
    if sigma_p <= 0:
        raise ValueError("--sigma_p must be > 0")

    # Bounds / init
    seed = int(args.seed)
    I_min = float(args.I_min) if args.I_min is not None else -float(I_max)

    tol_good = float(args.tol_E_good) if args.tol_E_good is not None else float(_get("tol", args.tol))
    tol_weak = float(args.tol_E_weak)
    tol_bad = float(args.tol_E_bad)
    penalty_bad = float(args.penalty_bad)

    # Weights (inverse scales)
    w_I = float(args.w_I)
    w_x = float(args.w_x)
    w_dI = float(args.w_dI)
    w_dx = float(args.w_dx)
    w_Ix = float(args.w_Ix)
    dx_floor = float(args.dx_floor)

    # Physical step limits
    step_I_max = float(args.step_I_max)
    step_x_max = float(args.step_x_max)
    if step_I_max <= 0 or step_x_max <= 0:
        raise ValueError("--step_I_max and --step_x_max must be > 0")

    ridge_adj = float(args.ridge_adjoint)
    fd_x_eps = float(args.fd_x_eps)

    # Initialize u from randomized I0, x=x0
    n_coils = 8
    key = jax.random.PRNGKey(seed)
    I0 = jax.random.uniform(
        key,
        shape=(n_coils,),
        minval=float(I_min),
        maxval=float(I_max),
        dtype=jnp.float64,
    )
    u = _Ix_to_u(I0, x0, I_max=I_max, x_min=x_min, x_max=x_max)

    # Save run config
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "argv": vars(args),
                "fk_cfg": cfg,
                "p_des": [float(v) for v in p_des.tolist()],
                "seed": seed,
                "I_min": I_min,
                "I_max": I_max,
                "x0": x0,
                "x_bounds": [x_min, x_max],
                "x_ref": x_ref,
                "sigma_p": sigma_p,
                "sigmas_from_w": {
                    "sigma_I": (1.0 / w_I) if w_I > 0 else None,
                    "sigma_x": (1.0 / w_x) if w_x > 0 else None,
                    "sigma_dI": (1.0 / w_dI) if w_dI > 0 else None,
                    "sigma_dx": (1.0 / w_dx) if w_dx > 0 else None,
                    "sigma_Ix": (1.0 / w_Ix) if w_Ix > 0 else None,
                },
                "step_limits": {"step_I_max": step_I_max, "step_x_max": step_x_max},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _safe_sigma(w: float) -> str:
        if w <= 0:
            return "inf"
        return f"{(1.0 / w):.3g}"

    print("[IK-LM] ==============================")
    print(f"[IK-LM] run_dir = {str(run_dir)}")
    print(f"[IK-LM] p_des = {p_des.tolist()} (m), eps_p = {eps_p*1e3:.3f} (mm)")
    print(f"[IK-LM] sigma_p = {sigma_p:.3e} (m)  (r_p = e_p/sigma_p)")
    print(f"[IK-LM] I bound: [{I_min:.1f}, {I_max:.1f}] A (param via tanh)")
    print(f"[IK-LM] x0={x0:.6f} m, x in [{x_min:.6f}, {x_max:.6f}] m, x_ref={x_ref:.6f} m")
    print("[IK-LM] weights interpreted as inverse scales (w=1/sigma):")
    print(f"[IK-LM]   w_I ={w_I :.3e}  -> sigma_I ={_safe_sigma(w_I)} A")
    print(f"[IK-LM]   w_x ={w_x :.3e}  -> sigma_x ={_safe_sigma(w_x)} m")
    print(f"[IK-LM]   w_dI={w_dI:.3e}  -> sigma_dI={_safe_sigma(w_dI)} A")
    print(f"[IK-LM]   w_dx={w_dx:.3e}  -> sigma_dx={_safe_sigma(w_dx)} m")
    print(f"[IK-LM]   w_Ix={w_Ix:.3e}  -> sigma_Ix={_safe_sigma(w_Ix)} (A/m)")
    print(f"[IK-LM] physical step limits: step_I_max={step_I_max:.3g} A, step_x_max={step_x_max:.3g} m")
    print(
        f"[IK-LM] outer LM: max_iter={int(args.outer_max_iter)}, lam_init={float(args.lam_init):.3e}, "
        f"lam_up={float(args.lam_up):.3g}, lam_down={float(args.lam_down):.3g}, "
        f"rho_bad={float(args.rho_bad):.2f}, rho_good={float(args.rho_good):.2f}"
    )
    print("[IK-LM] ==============================")

    lam = float(args.lam_init)

    # Warm-start z for FD in x
    z_warm: Optional[Array] = None

    # Initial evaluation: prev accepted = current
    I, x, _, _ = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
    I_prev = jnp.asarray(I, dtype=jnp.float64)
    x_prev = float(x)

    cur = _fk_eval(
        engine,
        I=I,
        x=x,
        p_des=p_des,
        sigma_p=sigma_p,
        tol_good=tol_good,
        tol_weak=tol_weak,
        tol_bad=tol_bad,
        penalty_bad=penalty_bad,
        w_I=w_I,
        w_x=w_x,
        x_ref=x_ref,
        I_prev=I_prev,
        x_prev=x_prev,
        w_dI=w_dI,
        w_dx=w_dx,
        w_Ix=w_Ix,
        dx_floor=dx_floor,
        override_z0_bar=z_warm,
    )
    z_warm = cur.z_star_bar

    print(
        f"[IK-LM] iter=000 cls={cur.cls:<5s} ok_strict={cur.ok_strict} ||E||={cur.normE:.3e} "
        f"x={cur.x:.6f} p_tip={cur.p_tip.tolist()} ||e||={cur.err*1e3:.3f}mm cost={cur.cost:.6e}"
    )

    if cur.err <= eps_p and cur.cls != "REJECT":
        print(f"[IK-LM] Converged at init: ||e||={cur.err*1e3:.3f}mm")
        return

    for k in range(1, int(args.outer_max_iter) + 1):
        # 1) Sensitivities dp/dI and dp/dx (at current accepted state)
        try:
            jac_res = compute_dp_dI_via_lm_adjoint(
                z_star_bar=cur.z_star_bar,
                params=cur.params,
                lm_stats=cur.lm_stats,
                coil_currents=cur.I,
                ridge=ridge_adj,
                jac_method_I="fwd",
            )
            Jp_I = jnp.asarray(jac_res.J_p_I, dtype=jnp.float64)  # (3, nI)
        except Exception as e:
            print(f"[IK-LM] WARNING: dp/dI adjoint failed: {repr(e)}")
            Jp_I = jnp.zeros((3, n_coils), dtype=jnp.float64)

        dp_dx = _compute_dp_dx_fd(
            engine,
            I=cur.I,
            x=cur.x,
            h=fd_x_eps,
            z0_bar=z_warm,
            clip_bounds=(x_min, x_max),
        )  # (3,)

        # 2) Build Jacobian wrt u
        # Residual order:
        #   [r_p(3), r_I(n), r_x(1), r_dI(n), r_dx(1), r_Ix(n), r_pen(1)]
        I_u, x_u, dI_du, dx_du = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
        dI_du = jnp.asarray(dI_du, dtype=jnp.float64).reshape(-1,)

        # Chain: dp/duI = dp/dI * dI/du
        J_p_uI = Jp_I * dI_du.reshape(1, -1)  # (3,n)
        J_p_ux = dp_dx.reshape(3, 1) * float(dx_du)  # (3,1)
        J_top = jnp.concatenate([J_p_uI, J_p_ux], axis=1) / float(sigma_p)  # scale r_p by 1/sigma_p

        # r_I = w_I * I
        J_I_uI = float(w_I) * jnp.diag(dI_du)
        J_I_ux = jnp.zeros((n_coils, 1), dtype=jnp.float64)
        J_I = jnp.concatenate([J_I_uI, J_I_ux], axis=1)

        # r_x = w_x * (x - x_ref)
        J_x_uI = jnp.zeros((1, n_coils), dtype=jnp.float64)
        J_x_ux = jnp.asarray([[float(w_x) * float(dx_du)]], dtype=jnp.float64)
        J_x = jnp.concatenate([J_x_uI, J_x_ux], axis=1)

        # r_dI = w_dI * (I - I_prev)   (I_prev fixed)
        J_dI_uI = float(w_dI) * jnp.diag(dI_du)
        J_dI_ux = jnp.zeros((n_coils, 1), dtype=jnp.float64)
        J_dI = jnp.concatenate([J_dI_uI, J_dI_ux], axis=1)

        # r_dx = w_dx * (x - x_prev)   (x_prev fixed)
        J_dx_uI = jnp.zeros((1, n_coils), dtype=jnp.float64)
        J_dx_ux = jnp.asarray([[float(w_dx) * float(dx_du)]], dtype=jnp.float64)
        J_dx = jnp.concatenate([J_dx_uI, J_dx_ux], axis=1)

        # r_Ix = w_Ix * (I - I_prev) / max(|x - x_prev|, dx_floor)
        # NOTE: We ignore d(denom)/du_x for robustness (treat denom as constant).
        dx_to_prev = float(x_u) - float(x_prev)
        denom = max(abs(dx_to_prev), float(dx_floor))
        J_Ix_uI = (float(w_Ix) / denom) * jnp.diag(dI_du)
        J_Ix_ux = jnp.zeros((n_coils, 1), dtype=jnp.float64)
        J_Ix = jnp.concatenate([J_Ix_uI, J_Ix_ux], axis=1)

        # penalty residual treated constant wrt u
        J_pen = jnp.zeros((1, n_coils + 1), dtype=jnp.float64)

        J = jnp.concatenate([J_top, J_I, J_x, J_dI, J_dx, J_Ix, J_pen], axis=0)
        r = cur.r

        # 3) LM step in u
        g = J.T @ r
        H = J.T @ J
        diagH = jnp.clip(jnp.diag(H), a_min=1e-12)
        D = jnp.diag(diagH)
        A = H + float(lam) * D

        try:
            du = -jnp.linalg.solve(A, g)
        except Exception as e:
            print(f"[IK-LM] ERROR: linear solve failed at iter={k}: {repr(e)}")
            lam = min(lam * float(args.lam_up), 1e12)
            continue

        # 4) Physical step clipping in (I,x), then map back to u
        u_prop = u + du
        I_prop, x_prop, _, _ = _u_to_Ix(u_prop, I_max=I_max, x_min=x_min, x_max=x_max)

        dI_prop = jnp.asarray(I_prop - cur.I, dtype=jnp.float64)
        dx_prop = float(x_prop - cur.x)

        dI_clip = jnp.clip(dI_prop, -float(step_I_max), float(step_I_max))
        dx_clip = float(jnp.clip(dx_prop, -float(step_x_max), float(step_x_max)))

        I_trial = jnp.asarray(cur.I + dI_clip, dtype=jnp.float64)
        I_trial = jnp.clip(I_trial, -float(I_max), float(I_max))
        x_trial = float(cur.x + dx_clip)
        x_trial = min(max(x_trial, float(x_min)), float(x_max))

        u_trial = _Ix_to_u(I_trial, x_trial, I_max=I_max, x_min=x_min, x_max=x_max)
        du_eff = u_trial - u

        dI_norm = float(jnp.linalg.norm(dI_clip))
        dI_max_abs = float(jnp.max(jnp.abs(dI_clip)))
        dx_abs = abs(dx_clip)
        du_eff_norm = float(jnp.linalg.norm(du_eff))

        # Predicted reduction using linear model with du_eff
        r_lin = r + J @ du_eff
        pred_red = 0.5 * float(jnp.dot(r, r) - jnp.dot(r_lin, r_lin))

        # 5) Evaluate candidate
        trial = _fk_eval(
            engine,
            I=I_trial,
            x=x_trial,
            p_des=p_des,
            sigma_p=sigma_p,
            tol_good=tol_good,
            tol_weak=tol_weak,
            tol_bad=tol_bad,
            penalty_bad=penalty_bad,
            w_I=w_I,
            w_x=w_x,
            x_ref=x_ref,
            I_prev=I_prev,
            x_prev=x_prev,
            w_dI=w_dI,
            w_dx=w_dx,
            w_Ix=w_Ix,
            dx_floor=dx_floor,
            override_z0_bar=z_warm,
        )

        act_red = float(cur.cost - trial.cost)
        rho = act_red / max(pred_red, 1e-18)

        accept = (trial.cost < cur.cost) and (rho > 0.0) and (trial.cls != "REJECT")
        status = "ACCEPT" if accept else "REJECT"
        print(
            f"[IK-LM] iter={k:03d} lam={lam:.2e} rho={rho:.3f} {status} "
            f"du_eff={du_eff_norm:.2e} |dI|_max={dI_max_abs:.3f}A ||dI||={dI_norm:.3f}A |dx|={dx_abs*1e3:.3f}mm "
            f"pred_red={pred_red:.3e} act_red={act_red:.3e} "
            f"cls={trial.cls:<5s} ||E||={trial.normE:.3e} x={trial.x:.6f} ||e||={trial.err*1e3:.3f}mm cost={trial.cost:.6e}"
        )

        # 6) Update lambda and state
        if accept:
            u = u_trial
            cur = trial
            z_warm = cur.z_star_bar

            # update "previous accepted" reference (for dI/dx residuals)
            I_prev = jnp.asarray(cur.I, dtype=jnp.float64)
            x_prev = float(cur.x)

            if rho >= float(args.rho_good):
                lam = max(lam * float(args.lam_down), 1e-12)
            elif rho <= float(args.rho_bad):
                lam = min(lam * float(args.lam_up), 1e12)

            if cur.err <= eps_p and cur.cls != "REJECT":
                print(f"[IK-LM] Converged: iter={k}, ||e||={cur.err*1e3:.3f}mm, x={cur.x:.6f}")
                break
        else:
            lam = min(lam * float(args.lam_up), 1e12)

    # Final summary
    I_fin, x_fin, _, _ = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": bool(cur.err <= eps_p and cur.cls != "REJECT"),
                "p_des": [float(v) for v in p_des.tolist()],
                "p_tip": [float(v) for v in cur.p_tip.tolist()],
                "err_m": float(cur.err),
                "err_mm": float(cur.err * 1e3),
                "I": [float(v) for v in I_fin.tolist()],
                "x": float(x_fin),
                "normE": float(cur.normE),
                "cls": str(cur.cls),
                "sigma_p": float(sigma_p),
                "sigmas_from_w": {
                    "sigma_I": (1.0 / w_I) if w_I > 0 else None,
                    "sigma_x": (1.0 / w_x) if w_x > 0 else None,
                    "sigma_dI": (1.0 / w_dI) if w_dI > 0 else None,
                    "sigma_dx": (1.0 / w_dx) if w_dx > 0 else None,
                    "sigma_Ix": (1.0 / w_Ix) if w_Ix > 0 else None,
                },
                "step_limits": {"step_I_max": step_I_max, "step_x_max": step_x_max},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[IK-LM] Saved result.json")


if __name__ == "__main__":
    main()
