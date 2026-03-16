"""Single-layer LM inverse kinematics (position-only) over (I, x).

This script implements the "Plan A" requested workflow:

  - Do NOT modify FK.
  - Use ForwardKinematicsEngine.solve_with_stats(...) to evaluate FK at a given
    (coil currents I, protruding length x=L_protrude).
  - Use ForwardKinematicsEngine.query_sites(...) to extract observables.
  - Run a single outer Levenberg–Marquardt loop in the *control* space
    u = [u_I, u_x], where
        I = I_max * tanh(u_I)
        x = x_min + (x_max-x_min) * sigmoid(u_x)

In this version, ONLY the distal tip position is constrained.
The multi-site pose residual will be introduced incrementally in later steps.

Jacobian strategy
-----------------
  - dp/dI: use the existing LM-adjoint sensitivity (ik_diff.compute_dp_dI_via_lm_adjoint)
  - dp/dx: finite difference in x (two extra FK solves per outer iteration)

Why regularization is necessary (for position-only)
--------------------------------------------------
With only a 3D position target but 9 decision variables (8 currents + x), the
problem is underdetermined. We therefore include lightweight regularization
residuals by default:
  - current penalty: r_I = w_I * I
  - insertion penalty: r_x = w_x * (x - x_ref)

These penalties make the least-squares well-posed and steer the solver toward
small currents and a preferred insertion depth.

Usage (example)
--------------
python inverse_kinematics_bak/ik_multisite_lm.py \
  --enable_magnetics --calib_file=/abs/path/calibration.json \
  --m_body_list=0,0,-0.005301;0,0,0.005301;0.005301,0,0 \
  --p_des=0.05,0.01,0.04 \
  --x0=0.11 --x_min=0.08 --x_max=0.13 \
  --I_max=50

Notes
-----
* This script is intentionally verbose to support debugging.
* It reuses fk.py CLI flags for geometry/material/actuation configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from .ik_diff import compute_dp_dI_via_lm_adjoint
from .ik_artifacts import (
    extract_centerline_dim,
    plot_centerlines_3d,
    plot_metrics,
    save_centerline,
    save_history_json,
)

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
    (run / "centerlines").mkdir(parents=True, exist_ok=True)
    return run


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()

    # IK targets
    p.add_argument("--p_des", type=str, required=True, help="Desired tip position [m], format x,y,z")
    p.add_argument("--eps_p", type=float, default=2e-3, help="Stop if ||p_tip - p_des|| <= eps_p [m].")

    # Optimization variables: currents and insertion depth
    p.add_argument(
        "--I_min",
        type=float,
        default=None,
        help="Lower current bound. If not set, uses -I_max.",
    )
    p.add_argument("--I_max", type=float, default=50.0, help="Symmetric current bound: I in [-I_max, I_max]")
    p.add_argument("--x0", type=float, default=None, help="Initial insertion depth x [m]. If not set, uses --L_protrude.")
    p.add_argument("--x_min", type=float, default=None, help="Lower bound for insertion depth x [m]")
    p.add_argument("--x_max", type=float, default=None, help="Upper bound for insertion depth x [m]")
    p.add_argument(
        "--x_ref",
        type=float,
        default=None,
        help="Reference insertion depth for regularization. Default: x0.",
    )

    # Outer LM settings
    p.add_argument("--outer_max_iter", type=int, default=70, help="Max outer LM iterations")
    p.add_argument("--lam_init", type=float, default=1e-2, help="Initial LM damping (outer)")
    p.add_argument("--lam_up", type=float, default=10.0, help="Multiply lambda on reject")
    p.add_argument("--lam_down", type=float, default=0.3, help="Multiply lambda on good accept")
    p.add_argument("--rho_good", type=float, default=0.75, help="rho threshold for decreasing lambda")
    p.add_argument("--rho_bad", type=float, default=0.25, help="rho threshold for increasing lambda")
    p.add_argument("--step_max", type=float, default=1.0, help="Max ||du|| (in parameter space) per iteration")

    # Regularization weights (residual scaling)
    p.add_argument("--w_I", type=float, default=1e-3, help="Residual weight for current penalty r_I = w_I * I")
    p.add_argument("--w_x", type=float, default=1e-2, help="Residual weight for insertion penalty r_x = w_x * (x-x_ref)")

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
    if (not jnp.isfinite(normE)):
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
    e_p: Array
    err: float
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
    I: Array,
    x: float,
    w_I: float,
    w_x: float,
    x_ref: float,
    penalty: float,
) -> Tuple[Array, float]:
    """Build stacked residual r and its cost."""
    e_p = jnp.asarray(p_tip - p_des, dtype=jnp.float64).reshape(3,)
    r_p = e_p
    r_I = float(w_I) * jnp.asarray(I, dtype=jnp.float64).reshape(-1,)
    r_x = jnp.asarray([float(w_x) * (float(x) - float(x_ref))], dtype=jnp.float64)
    r_pen = jnp.asarray([jnp.sqrt(float(max(penalty, 0.0)))], dtype=jnp.float64)
    r = jnp.concatenate([r_p, r_I, r_x, r_pen], axis=0)
    cost = 0.5 * float(jnp.dot(r, r))
    return r, cost


def _fk_eval(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    x: float,
    p_des: Array,
    tol_good: float,
    tol_weak: float,
    tol_bad: float,
    penalty_bad: float,
    w_I: float,
    w_x: float,
    x_ref: float,
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

    r, cost = _compute_residual_vector(
        p_tip=p_tip,
        p_des=p_des,
        I=I,
        x=float(x),
        w_I=float(w_I),
        w_x=float(w_x),
        x_ref=float(x_ref),
        penalty=float(penalty),
    )
    e_p = jnp.asarray(p_tip - p_des, dtype=jnp.float64)
    err = float(jnp.linalg.norm(e_p))
    return EvalPack(
        I=jnp.asarray(I, dtype=jnp.float64),
        x=float(x),
        p_tip=jnp.asarray(p_tip, dtype=jnp.float64),
        e_p=e_p,
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
) -> float:
    """Finite difference dp_tip/dx along x, returns a scalar 3x1 vector? (actually 3,)."""
    x_min, x_max = clip_bounds
    h = float(h)
    if h <= 0.0:
        raise ValueError("fd step h must be > 0")
    xp = min(float(x_max), float(x) + h)
    xm = max(float(x_min), float(x) - h)
    if abs(xp - xm) < 1e-12:
        return jnp.zeros((3,), dtype=jnp.float64)

    # Use warm start z0_bar if provided, otherwise engine warm cache is used.
    zp, params_p, _, _, stats_p = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(xp),
        warm_start=True,
        override_z0_bar=z0_bar,
        return_stats=True,
    )
    zm, params_m, _, _, stats_m = engine.solve_with_stats(
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
        # FK will raise later; keep message here too
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
        L_protrude_max=x_max,
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

    seed = int(args.seed)
    I_min = float(args.I_min) if args.I_min is not None else -float(I_max)

    tol_good = float(args.tol_E_good) if args.tol_E_good is not None else float(_get("tol", args.tol))
    tol_weak = float(args.tol_E_weak)
    tol_bad = float(args.tol_E_bad)
    penalty_bad = float(args.penalty_bad)

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
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[IK-LM] ==============================")
    print(f"[IK-LM] run_dir = {str(run_dir)}")
    print(f"[IK-LM] p_des = {p_des.tolist()} (m), eps_p = {eps_p*1e3:.3f} (mm)")
    print(f"[IK-LM] I bound: [-{I_max}, +{I_max}] A")
    print(f"[IK-LM] x0={x0:.6f} m, x in [{x_min:.6f}, {x_max:.6f}] m, x_ref={x_ref:.6f} m")
    print(f"[IK-LM] regularization: w_I={float(args.w_I):.3e}, w_x={float(args.w_x):.3e}")
    print(
        f"[IK-LM] outer LM: max_iter={int(args.outer_max_iter)}, lam_init={float(args.lam_init):.3e}, "
        f"lam_up={float(args.lam_up):.3g}, lam_down={float(args.lam_down):.3g}, "
        f"rho_bad={float(args.rho_bad):.2f}, rho_good={float(args.rho_good):.2f}, step_max={float(args.step_max):.3g}"
    )
    print("[IK-LM] ==============================")

    lam = float(args.lam_init)
    step_max = float(args.step_max)
    w_I = float(args.w_I)
    w_x = float(args.w_x)
    ridge_adj = float(args.ridge_adjoint)
    fd_x_eps = float(args.fd_x_eps)

    # Warm-start z for FD in x
    z_warm: Optional[Array] = None

    # History containers (aligned with ik_artifacts.plot_metrics)
    history: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "outer": [],
        "accepted": [],
        "success": False,
        "final": {},
    }

    # Initial evaluation
    I, x, dI_du, dx_du = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
    cur = _fk_eval(
        engine,
        I=I,
        x=x,
        p_des=p_des,
        tol_good=tol_good,
        tol_weak=tol_weak,
        tol_bad=tol_bad,
        penalty_bad=penalty_bad,
        w_I=w_I,
        w_x=w_x,
        x_ref=x_ref,
        override_z0_bar=z_warm,
    )
    z_warm = cur.z_star_bar
    print(
        f"[IK-LM] iter=000 cls={cur.cls:<5s} ok_strict={cur.ok_strict} ||E||={cur.normE:.3e} "
        f"x={cur.x:.6f} p_tip={cur.p_tip.tolist()} ||e||={cur.err*1e3:.3f}mm cost={cur.cost:.6e}"
    )

    # Initial centerline
    centerline0 = extract_centerline_dim(cur.z_star_bar, M_list=cur.params.M_list, scales=cur.params.scales)
    save_centerline(centerline0, out_path=run_dir / "centerlines" / "init.npy")
    p_tip_init = centerline0[-1] if centerline0.size else None

    accepted_centerlines = [centerline0]
    history["accepted"].append(
        {
            "tag": "init",
            "iter": 0,
            "err_mm": float(cur.err * 1e3),
            "normE": float(cur.normE),
            "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
            "mu": float(lam),
            "I": [float(v) for v in cur.I.tolist()],
            "x": float(cur.x),
            "p_tip": [float(v) for v in cur.p_tip.tolist()],
        }
    )

    history["outer"].append(
        {
            "iter": 0,
            "mu": float(lam),
            "err_mm": float(cur.err * 1e3),
            "normE": float(cur.normE),
            "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
        }
    )

    if cur.err <= eps_p and cur.cls != "REJECT":
        print(f"[IK-LM] Converged at init: ||e||={cur.err*1e3:.3f}mm")
        history["success"] = True
        history["final"] = {
            "I": [float(v) for v in cur.I.tolist()],
            "x": float(cur.x),
            "p_tip": [float(v) for v in cur.p_tip.tolist()],
            "err_mm": float(cur.err * 1e3),
            "normE": float(cur.normE),
            "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
        }
        save_history_json(history, out_path=run_dir / "history.json")
        p_tip_final = accepted_centerlines[-1][-1] if accepted_centerlines[-1].size else None
        plot_centerlines_3d(
            centerlines=accepted_centerlines,
            p_tip_init=p_tip_init,
            p_tip_final=p_tip_final,
            p_des=jax.device_get(p_des),
            out_path=run_dir / "centerlines_3d.png",
            title="IK-LM centerline evolution (accepted)",
        )
        plot_metrics(history, out_dir=run_dir)
        return

    for k in range(1, int(args.outer_max_iter) + 1):
        # 1) Sensitivities dp/dI and dp/dx
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
        #    r = [e_p, w_I*I, w_x*(x-x_ref), sqrt(penalty)]
        I_u, x_u, dI_du, dx_du = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
        dI_du = jnp.asarray(dI_du, dtype=jnp.float64).reshape(-1,)
        # Chain: dp/duI = dp/dI * dI/du
        J_p_uI = Jp_I * dI_du.reshape(1, -1)
        J_p_ux = dp_dx.reshape(3, 1) * float(dx_du)

        # r_p part
        J_top = jnp.concatenate([J_p_uI, J_p_ux], axis=1)  # (3, nI+1)

        # r_I part
        J_I_uI = float(w_I) * jnp.diag(dI_du)
        J_I_ux = jnp.zeros((n_coils, 1), dtype=jnp.float64)
        J_midI = jnp.concatenate([J_I_uI, J_I_ux], axis=1)  # (nI, nI+1)

        # r_x part
        J_x_uI = jnp.zeros((1, n_coils), dtype=jnp.float64)
        J_x_ux = jnp.asarray([[float(w_x) * float(dx_du)]], dtype=jnp.float64)
        J_midx = jnp.concatenate([J_x_uI, J_x_ux], axis=1)  # (1, nI+1)

        # penalty residual is constant wrt u (we treat it as fixed for the linearization)
        J_pen = jnp.zeros((1, n_coils + 1), dtype=jnp.float64)

        J = jnp.concatenate([J_top, J_midI, J_midx, J_pen], axis=0)  # (m, n)
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

        du_norm = float(jnp.linalg.norm(du))
        if du_norm > step_max:
            du = du * (step_max / max(du_norm, 1e-12))
            du_norm = float(jnp.linalg.norm(du))

        # Predicted reduction using linear model
        r_lin = r + J @ du
        pred_red = 0.5 * float(jnp.dot(r, r) - jnp.dot(r_lin, r_lin))

        # 4) Evaluate candidate
        u_trial = u + du
        I_trial, x_trial, _, _ = _u_to_Ix(u_trial, I_max=I_max, x_min=x_min, x_max=x_max)
        trial = _fk_eval(
            engine,
            I=I_trial,
            x=x_trial,
            p_des=p_des,
            tol_good=tol_good,
            tol_weak=tol_weak,
            tol_bad=tol_bad,
            penalty_bad=penalty_bad,
            w_I=w_I,
            w_x=w_x,
            x_ref=x_ref,
            override_z0_bar=z_warm,
        )

        act_red = float(cur.cost - trial.cost)
        rho = act_red / max(pred_red, 1e-18)

        accept = (trial.cost < cur.cost) and (rho > 0.0) and (trial.cls != "REJECT")
        status = "ACCEPT" if accept else "REJECT"
        print(
            f"[IK-LM] iter={k:03d} lam={lam:.2e} ||du||={du_norm:.3e} pred_red={pred_red:.3e} act_red={act_red:.3e} rho={rho:.3f} {status} "
            f"cls={trial.cls:<5s} ||E||={trial.normE:.3e} x={trial.x:.6f} ||e||={trial.err*1e3:.3f}mm cost={trial.cost:.6e}"
        )

        history["outer"].append(
            {
                "iter": int(k),
                "mu": float(lam),
                "err_mm": float(trial.err * 1e3) if accept else float(cur.err * 1e3),
                "normE": float(trial.normE) if accept else float(cur.normE),
                "lm_iter": int(getattr(trial.lm_stats, "n_iter", -1)) if accept else int(getattr(cur.lm_stats, "n_iter", -1)),
            }
        )

        # 5) Update lambda and state
        if accept:
            u = u_trial
            cur = trial
            z_warm = cur.z_star_bar

            cl = extract_centerline_dim(cur.z_star_bar, M_list=cur.params.M_list, scales=cur.params.scales)
            accepted_centerlines.append(cl)
            save_centerline(
                cl,
                out_path=run_dir / "centerlines" / f"acc_{len(accepted_centerlines)-1:03d}.npy",
            )
            history["accepted"].append(
                {
                    "tag": "accept",
                    "iter": int(k),
                    "err_mm": float(cur.err * 1e3),
                    "normE": float(cur.normE),
                    "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
                    "mu": float(lam),
                    "I": [float(v) for v in cur.I.tolist()],
                    "x": float(cur.x),
                    "p_tip": [float(v) for v in cur.p_tip.tolist()],
                }
            )

            if rho >= float(args.rho_good):
                lam = max(lam * float(args.lam_down), 1e-12)
            elif rho <= float(args.rho_bad):
                lam = min(lam * float(args.lam_up), 1e12)

            if cur.err <= eps_p and cur.cls != "REJECT":
                print(f"[IK-LM] Converged: iter={k}, ||e||={cur.err*1e3:.3f}mm, x={cur.x:.6f}")
                history["success"] = True
                break
        else:
            lam = min(lam * float(args.lam_up), 1e12)

    # Final summary
    I_fin, x_fin, _, _ = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)

    history["final"] = {
        "I": [float(v) for v in I_fin.tolist()],
        "x": float(x_fin),
        "p_tip": [float(v) for v in cur.p_tip.tolist()],
        "err_mm": float(cur.err * 1e3),
        "normE": float(cur.normE),
        "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
    }

    save_history_json(history, out_path=run_dir / "history.json")

    p_tip_final = accepted_centerlines[-1][-1] if accepted_centerlines[-1].size else None
    plot_centerlines_3d(
        centerlines=accepted_centerlines,
        p_tip_init=p_tip_init,
        p_tip_final=p_tip_final,
        p_des=jax.device_get(p_des),
        out_path=run_dir / "centerlines_3d.png",
        title="IK-LM centerline evolution (accepted)",
    )
    plot_metrics(history, out_dir=run_dir)

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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[IK-LM] Saved result.json")


if __name__ == "__main__":
    main()
