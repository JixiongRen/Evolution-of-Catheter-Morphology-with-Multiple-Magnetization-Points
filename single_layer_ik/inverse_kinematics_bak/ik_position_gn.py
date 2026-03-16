"""Gradient-based IK (position only) using LM-adjoint dp/dI.

This script is CLI-aligned with fk.py: it reuses the same segment/material/
actuation flags and adds IK-specific flags.

Key features (requested)
------------------------
1) Extremely verbose, step-by-step logging, including accept/reject reasons.
2) Modular: sensitivity (ik_diff.py) and artifact saving/plotting (ik_artifacts.py).
3) JAX/GPU-first: Jacobians via JAX AD; inner equilibrium via JAX LM.
4) Only position is considered (no orientation).

Outer loop uses a damped Gauss-Newton step in coil currents I, with:
  - trust-region step cap (max_step_A)
  - LM-style ratio test (rho = actual/predicted reduction) to update mu
  - weak-accept tolerances near convergence to avoid floating-point deadlocks
  - optional exploration when Jp is nearly zero

Artifacts
---------
For every *accepted* IK update, we save:
  - full catheter centerline (SI)
  - key scalar metrics (err, cost, inner LM iters, rho, mu, ...)
After IK finishes, we generate summary plots:
  - centerline overlays with initial tip (red) and final tip (green)
  - curves for err, inner ||E||, inner LM iters, and outer mu

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
from ik_diff import compute_dp_dI_via_lm_adjoint, extract_tip_p_dim
from ik_artifacts import (
    extract_centerline_dim,
    plot_centerlines_3d,
    plot_metrics,
    save_centerline,
    save_history_json,
)

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()

    # --- IK specific ---
    p.add_argument("--p_des", type=str, required=True, help="Desired tip position [m], format x,y,z")
    p.add_argument("--eps_p", type=float, default=10e-3, help="Position tolerance [m]. Default 1mm")

    p.add_argument("--I_min", type=float, default=-50.0, help="Lower bound for every coil current [A]")
    p.add_argument("--I_max", type=float, default=50.0, help="Upper bound for every coil current [A]")
    p.add_argument("--I0", type=str, default=None, help="Optional initial currents, comma-separated. Default zeros")
    p.add_argument(
        "--I0_random", action="store_true",
        help="If set and --I0 is not provided, initialize I0 randomly within bounds."
    )
    p.add_argument(
        "--I0_random_scale", type=float, default=0.2,
        help="Random I0 scale as fraction of (I_max-I_min). Only used when --I0_random."
    )

    p.add_argument("--outer_max_iter", type=int, default=50, help="Outer GN iterations")
    p.add_argument("--line_search_max", type=int, default=6, help="Backtracking line-search trials per iter")

    # Penalty / acceptance based on equilibrium residual
    p.add_argument(
        "--tol_E_good", type=float, default=None,
        help="Treat equilibrium as GOOD if ||E|| <= tol_E_good. Default: use fk --tol"
    )
    p.add_argument("--tol_E_weak", type=float, default=1e-2, help="Treat equilibrium as WEAK but usable if ||E|| <= tol_E_weak")
    p.add_argument("--tol_E_bad", type=float, default=1e-1, help="Treat equilibrium as BAD/REJECT if ||E|| > tol_E_bad")
    p.add_argument("--penalty_bad", type=float, default=1e3, help="Penalty added to cost when equilibrium is BAD or tip is NaN")

    # Regularization
    p.add_argument("--alpha_I", type=float, default=0.0, help="Quadratic current regularization weight")

    # GN damping
    p.add_argument("--mu_init", type=float, default=1e-2, help="Initial GN damping")
    p.add_argument("--mu_min", type=float, default=1e-12, help="Lower bound for mu")
    # NOTE: A too-large mu_max can freeze GN steps near convergence.
    # We keep a conservative default; users can override if needed.
    p.add_argument("--mu_max", type=float, default=1e+8, help="Upper bound for mu")
    p.add_argument("--mu_up", type=float, default=10.0, help="Multiply mu when we need to be more conservative")
    p.add_argument("--mu_down", type=float, default=0.3, help="Multiply mu when model is reliable")
    p.add_argument("--rho_low", type=float, default=0.25, help="rho threshold for increasing mu")
    p.add_argument("--rho_high", type=float, default=0.75, help="rho threshold for decreasing mu")
    p.add_argument("--ridge_adjoint", type=float, default=0.0, help="Extra ridge added to LM damping when computing dp/dI")

    # A) Trust region / step cap
    p.add_argument("--max_step_A", type=float, default=2.0, help="Max ||dI|| (A) per outer iteration before line-search")

    # C) Weak accept tolerances
    p.add_argument("--ls_cost_tol_abs", type=float, default=1e-12, help="Absolute cost tolerance for weak accept")
    p.add_argument("--ls_cost_tol_rel", type=float, default=1e-6, help="Relative cost tolerance for weak accept")
    p.add_argument(
        "--ls_err_tol_mm",
        type=float,
        default=0.05,
        help="Accept if ||e|| drops by at least this (mm), even if cost is flat",
    )

    # B) Predicted reduction floor: avoid pathological rho when pred_red ~ 0
    p.add_argument(
        "--pred_red_floor_abs",
        type=float,
        default=1e-14,
        help="Treat |pred_red| below this as unreliable (absolute)",
    )
    p.add_argument(
        "--pred_red_floor_rel",
        type=float,
        default=1e-10,
        help="Treat |pred_red| below this*cost as unreliable (relative)",
    )

    # Debug / robustness knobs
    # D) Minimum outer step to avoid numerical deadlocks (e.g., dI ~ 1e-12 A)
    p.add_argument(
        "--min_step_norm_A",
        type=float,
        default=0.2,
        help="If >0, scale GN step so ||dI|| >= this value (A) before line-search",
    )
    # Back-compat alias (older runs)
    p.add_argument(
        "--min_step_norm",
        type=float,
        default=None,
        help="Alias of --min_step_norm_A (deprecated)",
    )
    p.add_argument("--Jp_norm_warn", type=float, default=1e-9, help="Warn + explore if ||Jp||_F is below this (m/A)")
    p.add_argument("--explore_scale", type=float, default=0.05, help="Random exploration scale as fraction of (I_max-I_min) when GN is uninformative")

    # Artifacts
    p.add_argument("--out_dir", type=str, default="ik_out", help="Output directory for artifacts")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name (subfolder under out_dir)")
    p.add_argument("--save_all_trials", action="store_true", help="If set, also store every line-search trial in history")

    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--print_devices", action="store_true", help="Print JAX devices at start")

    return p


def _parse_vec3(s: str) -> Array:
    xs = [float(x.strip()) for x in s.split(",")]
    if len(xs) != 3:
        raise ValueError(f"Expected 3 values, got: {s}")
    return jnp.asarray(xs, dtype=jnp.float64)


def _parse_I(s: Optional[str], n: int) -> Array:
    if s is None:
        return jnp.zeros((n,), dtype=jnp.float64)
    xs = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(xs) != n:
        raise ValueError(f"Expected {n} currents, got {len(xs)} in: {s}")
    return jnp.asarray(xs, dtype=jnp.float64)


def _clip_I(I: Array, I_min: float, I_max: float) -> Tuple[Array, int]:
    I_clipped = jnp.clip(I, I_min, I_max)
    n_clip = int(jnp.sum(I_clipped != I))
    return I_clipped, n_clip


def classify_equilibrium(*, ok_strict: bool, normE: float, tol_good: float, tol_weak: float, tol_bad: float) -> str:
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
class FKEval:
    I: Array
    ok_strict: bool
    normE: float
    cls: str
    p_tip: Array
    e_p: Array
    err: float
    cost: float
    penalty: float
    z_star_bar: Array
    params: Any
    lm_stats: Any


def compute_cost(err: float, I: Array, *, alpha_I: float, penalty: float) -> float:
    return 0.5 * float(err * err) + float(alpha_I) * float(jnp.dot(I, I)) + float(penalty)


def fk_eval(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    L_protrude: float,
    p_des: Array,
    tol_good: float,
    tol_weak: float,
    tol_bad: float,
    penalty_bad: float,
    alpha_I: float,
) -> FKEval:
    z_star_bar, params, _meshes, ok, lm_stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=L_protrude,
        warm_start=True,
        return_stats=True,
    )

    # Tip position
    p_tip = extract_tip_p_dim(z_star_bar, M_list=params.M_list, scales=params.scales)

    # Equilibrium quality
    normE = float(lm_stats.normE)
    cls = classify_equilibrium(ok_strict=bool(ok), normE=normE, tol_good=tol_good, tol_weak=tol_weak, tol_bad=tol_bad)

    # Position error
    if jnp.any(~jnp.isfinite(p_tip)):
        cls = "REJECT"

    e_p = p_tip - p_des
    err = float(jnp.linalg.norm(e_p))

    penalty = 0.0
    if cls in ("BAD", "REJECT"):
        penalty = float(penalty_bad)

    cost = compute_cost(err, I, alpha_I=alpha_I, penalty=penalty)

    return FKEval(
        I=I,
        ok_strict=bool(ok),
        normE=normE,
        cls=cls,
        p_tip=p_tip,
        e_p=e_p,
        err=err,
        cost=cost,
        penalty=penalty,
        z_star_bar=z_star_bar,
        params=params,
        lm_stats=lm_stats,
    )


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


def main() -> None:
    args = build_argparser().parse_args()

    if args.print_devices:
        print("[IK] JAX devices:", jax.devices())

    run_dir = _make_run_dir(args.out_dir, args.run_name)

    cfg = fk_cli.load_config_file(args.config)

    def _get(name: str, default):
        v_cli = getattr(args, name, None)
        if v_cli is not None and (not isinstance(v_cli, bool) or v_cli is True or default is False):
            return v_cli
        return cfg.get(name, default)

    # Parse fk-style args
    flex_lengths = fk_cli._parse_csv_floats(_get("flex_lengths", args.flex_lengths))
    rigid_lengths = fk_cli._parse_csv_floats(_get("rigid_lengths", args.rigid_lengths))
    M_list = fk_cli._parse_csv_ints(_get("M_list", args.M_list))

    L_protrude = float(_get("L_protrude", args.L_protrude))
    if L_protrude is None:
        raise ValueError("--L_protrude is required for IK (advancer state)")

    # Materials
    flex_d_outer = fk_cli._parse_csv_floats(_get("flex_d_outer", args.flex_d_outer))
    flex_E = fk_cli._parse_csv_floats(_get("flex_E", args.flex_E))
    flex_G = fk_cli._parse_csv_floats(_get("flex_G", args.flex_G))
    flex_rho = fk_cli._parse_csv_floats(_get("flex_rho", args.flex_rho))
    rigid_d_outer = fk_cli._parse_csv_floats(_get("rigid_d_outer", args.rigid_d_outer))
    rigid_rho = fk_cli._parse_csv_floats(_get("rigid_rho", args.rigid_rho))

    # Pose / axis
    p0_dim = _parse_vec3(_get("p0", args.p0))
    Q0 = jnp.asarray([float(x) for x in _get("Q0", args.Q0).split(",")], dtype=jnp.float64)
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

    # Build engine
    engine = ForwardKinematicsEngine(
        flex_lengths_tail=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        L1_min=float(_get("L1_min", args.L1_min)),
        L_protrude_max=float(L_protrude),
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

    # IK targets
    p_des = _parse_vec3(args.p_des)
    eps_p = float(args.eps_p)

    # Bounds and initial I
    n_coils = 8
    I_min = float(args.I_min)
    I_max = float(args.I_max)

    if args.I0 is not None:
        I = _parse_I(args.I0, n_coils)
    elif bool(args.I0_random):
        key = jax.random.PRNGKey(int(args.seed))
        scale = float(args.I0_random_scale) * (I_max - I_min)
        I = scale * jax.random.normal(key, shape=(n_coils,), dtype=jnp.float64)
    else:
        I = jnp.zeros((n_coils,), dtype=jnp.float64)

    I, n_clip0 = _clip_I(I, I_min, I_max)

    tol_good = float(args.tol_E_good) if args.tol_E_good is not None else float(_get("tol", args.tol))
    tol_weak = float(args.tol_E_weak)
    tol_bad = float(args.tol_E_bad)

    print("[IK] ==============================")
    print(f"[IK] run_dir = {str(run_dir)}")
    print(f"[IK] p_des = {p_des.tolist()} (m), eps_p = {eps_p*1e3:.3f} (mm)")
    print(f"[IK] I bounds = [{I_min}, {I_max}] A, initial clipped={n_clip0}")
    print(f"[IK] Equilibrium thresholds: tol_good={tol_good:.3e}, tol_weak={tol_weak:.3e}, tol_bad={tol_bad:.3e}")
    print(
        f"[IK] GN/LS: mu_init={float(args.mu_init):.3e}, max_step_A={float(args.max_step_A):.3g}A, "
        f"weak_accept: cost_tol_abs={float(args.ls_cost_tol_abs):.1e}, cost_tol_rel={float(args.ls_cost_tol_rel):.1e}, "
        f"err_tol={float(args.ls_err_tol_mm):.3g}mm"
    )
    print("[IK] ==============================")

    # Guardrail: magnetics requested but missing required inputs
    if bool(getattr(args, "enable_magnetics", False)):
        if getattr(args, "calib_file", None) is None:
            raise ValueError("--enable_magnetics was set but --calib_file is missing. Without calibration, I has no effect.")
        if getattr(args, "m_body_list", None) is None:
            raise ValueError("--enable_magnetics was set but --m_body_list is missing. Without magnet moments, I has no effect.")

    # Save run config
    config_out = {
        "argv": vars(args),
        "fk_cfg": cfg,
        "p_des": [float(x) for x in p_des.tolist()],
        "L_protrude": float(L_protrude),
        "I_bounds": [I_min, I_max],
    }
    (run_dir / "run_config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    mu = float(args.mu_init)
    mu_min = float(args.mu_min)
    mu_max = float(args.mu_max)

    # History containers
    history: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "accepted": [],
        "outer": [],
        "success": False,
        "final": {},
    }

    # Evaluate initial point
    cur = fk_eval(
        engine,
        I=I,
        L_protrude=L_protrude,
        p_des=p_des,
        tol_good=tol_good,
        tol_weak=tol_weak,
        tol_bad=tol_bad,
        penalty_bad=float(args.penalty_bad),
        alpha_I=float(args.alpha_I),
    )

    # Initial centerline / tip
    centerline0 = extract_centerline_dim(cur.z_star_bar, M_list=cur.params.M_list, scales=cur.params.scales)
    save_centerline(centerline0, out_path=run_dir / "centerlines" / "init.npy")
    p_tip_init = centerline0[-1]

    accepted_centerlines = [centerline0]
    accepted_meta = [
        {
            "tag": "init",
            "iter": -1,
            "err_mm": float(cur.err * 1e3),
            "normE": float(cur.normE),
            "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
            "mu": float(mu),
            "I": [float(x) for x in cur.I.tolist()],
            "p_tip": [float(x) for x in cur.p_tip.tolist()],
        }
    ]

    # Outer loop
    for k in range(int(args.outer_max_iter)):
        I = cur.I

        # Log current state
        e_mm = (cur.e_p * 1e3).tolist()
        lm_iter = int(getattr(cur.lm_stats, "n_iter", -1))
        print(
            f"[IK] iter={k:03d}  cls={cur.cls:<6} ok_strict={cur.ok_strict} "
            f"||E||={cur.normE:.3e} (lm_iter={lm_iter})  "
            f"p_tip={cur.p_tip.tolist()}  e_p(mm)={e_mm}  ||e||={cur.err*1e3:.3f}mm  "
            f"cost={cur.cost:.6g}  penalty={cur.penalty:.3g}  "
            f"mu={mu:.3e}  I_minmax=[{float(jnp.min(I)):.2f},{float(jnp.max(I)):.2f}]"
        )

        if cur.err <= eps_p and cur.cls != "REJECT":
            print(f"[IK] SUCCESS: ||e||={cur.err*1e3:.3f}mm <= {eps_p*1e3:.3f}mm")
            history["success"] = True
            break

        outer_rec: Dict[str, Any] = {
            "iter": k,
            "mu": float(mu),
            "I": [float(x) for x in I.tolist()],
            "cls": cur.cls,
            "ok_strict": bool(cur.ok_strict),
            "normE": float(cur.normE),
            "lm_iter": lm_iter,
            "p_tip": [float(x) for x in cur.p_tip.tolist()],
            "err_mm": float(cur.err * 1e3),
            "cost": float(cur.cost),
            "line_search": [],
            "accepted": False,
        }

        if cur.cls == "REJECT":
            # If we cannot linearize, increase damping and try a small random perturbation
            mu = min(max(mu * float(args.mu_up), mu_min), mu_max)
            print(f"[IK] WARN: equilibrium REJECT -> mu := {mu:.3e} and perturb currents")
            key = jax.random.PRNGKey(int(args.seed) + k)
            dI = float(args.explore_scale) * (I_max - I_min) * jax.random.normal(key, shape=I.shape, dtype=I.dtype)
            I_try, _ = _clip_I(I + dI, I_min, I_max)
            cur = fk_eval(
                engine,
                I=I_try,
                L_protrude=L_protrude,
                p_des=p_des,
                tol_good=tol_good,
                tol_weak=tol_weak,
                tol_bad=tol_bad,
                penalty_bad=float(args.penalty_bad),
                alpha_I=float(args.alpha_I),
            )
            history["outer"].append(outer_rec)
            continue

        # --- Linearization: dp/dI ---
        jac = compute_dp_dI_via_lm_adjoint(
            z_star_bar=cur.z_star_bar,
            params=cur.params,
            lm_stats=cur.lm_stats,
            coil_currents=cur.I,
            ridge=float(args.ridge_adjoint) if float(args.ridge_adjoint) > 0 else None,
        )
        Jp = jac.J_p_I  # (3, 8)

        Jp_norm = float(jnp.linalg.norm(Jp))
        Jp_col_max = jnp.max(jnp.abs(Jp), axis=0)  # (8,)
        print(f"[IK]   Jp: ||Jp||_F={Jp_norm:.3e} (m/A), max|col|={jnp.asarray(Jp_col_max).tolist()}")

        outer_rec["Jp_norm"] = float(Jp_norm)
        outer_rec["Jp_col_max"] = [float(x) for x in jnp.asarray(Jp_col_max).tolist()]

        if Jp_norm < float(args.Jp_norm_warn):
            # GN direction is effectively undefined -> explore instead
            print(f"[IK]   WARN: ||Jp||_F too small ({Jp_norm:.3e} < {float(args.Jp_norm_warn):.3e}). Random exploration.")
            key = jax.random.PRNGKey(int(args.seed) + 100000 + k)
            dI_explore = float(args.explore_scale) * (I_max - I_min) * jax.random.normal(key, shape=I.shape, dtype=I.dtype)
            I_try, _ = _clip_I(I + dI_explore, I_min, I_max)
            cur = fk_eval(
                engine,
                I=I_try,
                L_protrude=L_protrude,
                p_des=p_des,
                tol_good=tol_good,
                tol_weak=tol_weak,
                tol_bad=tol_bad,
                penalty_bad=float(args.penalty_bad),
                alpha_I=float(args.alpha_I),
            )
            history["outer"].append(outer_rec)
            continue

        # --- GN step ---
        # Solve (Jp^T Jp + mu I) dI = - Jp^T e
        JT = jnp.transpose(Jp)
        A = JT @ Jp + mu * jnp.eye(JT.shape[0], dtype=Jp.dtype)
        b = -(JT @ cur.e_p)
        dI = jnp.linalg.solve(A, b)

        step_norm_raw = float(jnp.linalg.norm(dI))

        # A) trust-region cap
        max_step = float(args.max_step_A)
        if max_step > 0 and step_norm_raw > max_step:
            scale = max_step / (step_norm_raw + 1e-12)
            dI = dI * scale
        step_norm = float(jnp.linalg.norm(dI))

        # D) enforce a minimum step norm before line-search (avoid dI ~ 1e-12A deadlocks)
        min_step = float(args.min_step_norm_A)
        if args.min_step_norm is not None:
            min_step = float(args.min_step_norm)
        if min_step > 0 and step_norm > 0 and step_norm < min_step:
            scale = min_step / (step_norm + 1e-12)
            dI = dI * scale
            step_norm = float(jnp.linalg.norm(dI))

        # Predicted reduction at alpha=1 (error-only)
        pred_e_1 = cur.e_p + Jp @ dI
        pred_reduction_1 = 0.5 * float(jnp.dot(cur.e_p, cur.e_p) - jnp.dot(pred_e_1, pred_e_1))

        print(
            f"[IK]   GN: mu={mu:.3e} ||dI_raw||={step_norm_raw:.3e} ||dI||={step_norm:.3e} "
            f"pred_red(err-only,alpha=1)={pred_reduction_1:.3e}"
        )

        outer_rec["dI_norm_raw"] = float(step_norm_raw)
        outer_rec["dI_norm"] = float(step_norm)
        outer_rec["pred_red_err_only_alpha1"] = float(pred_reduction_1)

        # --- Line search ---
        accepted = False
        tol_cost = float(args.ls_cost_tol_abs) + float(args.ls_cost_tol_rel) * float(cur.cost)
        err_tol = float(args.ls_err_tol_mm) * 1e-3

        # B) predicted reduction floor (absolute + relative to current cost)
        pred_floor = max(float(args.pred_red_floor_abs), float(args.pred_red_floor_rel) * float(cur.cost))

        for t in range(int(args.line_search_max)):
            alpha = 0.5 ** t
            dI_trial = alpha * dI
            I_try, n_clip = _clip_I(I + dI_trial, I_min, I_max)

            trial = fk_eval(
                engine,
                I=I_try,
                L_protrude=L_protrude,
                p_des=p_des,
                tol_good=tol_good,
                tol_weak=tol_weak,
                tol_bad=tol_bad,
                penalty_bad=float(args.penalty_bad),
                alpha_I=float(args.alpha_I),
            )

            # B) rho ratio: actual/predicted reduction
            # predicted cost using linearized e and exact I-regularization (penalty treated as constant)
            e_pred = cur.e_p + Jp @ dI_trial
            cost_pred = 0.5 * float(jnp.dot(e_pred, e_pred)) + float(args.alpha_I) * float(jnp.dot(I_try, I_try)) + float(cur.penalty)
            pred_red = float(cur.cost - cost_pred)
            act_red = float(cur.cost - trial.cost)

            # B) robust ratio test
            # - If pred_red is tiny or negative, rho becomes ill-conditioned and should be treated as unreliable.
            # - Use pred_floor (abs + rel) computed once per outer iteration.
            if (pred_red <= 0.0) or (abs(pred_red) < pred_floor):
                denom = 0.0
                rho = 0.0
            else:
                denom = pred_red
                rho = (act_red / denom)

            dI_trial_norm = float(jnp.linalg.norm(I_try - I))
            I_try_min = float(jnp.min(I_try))
            I_try_max = float(jnp.max(I_try))

            # C) weak accept
            cost_ok = (trial.cost <= cur.cost + tol_cost)
            err_ok = (trial.err <= cur.err - err_tol) if err_tol > 0 else (trial.err <= cur.err)
            accept = (trial.cls != "REJECT") and (cost_ok or err_ok)

            # decide reason
            if accept:
                reason = "cost" if cost_ok else "err"
            else:
                if trial.cls == "REJECT":
                    reason = "REJECT(cls)"
                else:
                    reason = "no_improve"

            trial_rec = {
                "t": t,
                "alpha": float(alpha),
                "clip": int(n_clip),
                "dI_trial_norm": float(dI_trial_norm),
                "I_try_minmax": [I_try_min, I_try_max],
                "cls": trial.cls,
                "ok_strict": bool(trial.ok_strict),
                "normE": float(trial.normE),
                "lm_iter": int(getattr(trial.lm_stats, "n_iter", -1)),
                "err_mm": float(trial.err * 1e3),
                "cost": float(trial.cost),
                "act_red": float(act_red),
                "pred_red": float(pred_red),
                "rho": float(rho),
                "accept": bool(accept),
                "accept_reason": reason,
            }
            if bool(args.save_all_trials):
                outer_rec["line_search"].append(trial_rec)

            print(
                f"[IK]   LS[{t}] alpha={alpha:.3f} clip={n_clip} "
                f"||dI_trial||={dI_trial_norm:.3e} Iminmax=[{I_try_min:.2f},{I_try_max:.2f}] "
                f"cls={trial.cls:<6} ||E||={trial.normE:.3e} (lm_iter={int(getattr(trial.lm_stats,'n_iter',-1))}) "
                f"||e||={trial.err*1e3:.3f}mm cost={trial.cost:.6g} "
                f"act_red={act_red:.3e} pred_red={pred_red:.3e} rho={rho:.2f} "
                f"{'ACCEPT' if accept else 'REJECT'}({reason})"
            )

            if accept:
                # Update mu based on rho
                if denom == 0.0:
                    # prediction unreliable (pred_red ~ 0 or < 0). Near convergence this often happens because
                    # mu is too large and the step becomes numerically ineffective. Prefer decreasing mu.
                    mu = min(max(mu * float(args.mu_down), mu_min), mu_max)
                    mu_note = "mu_down(pred~0)"
                else:
                    if rho < float(args.rho_low):
                        mu = min(max(mu * float(args.mu_up), mu_min), mu_max)
                        mu_note = "mu_up(rho_low)"
                    elif rho > float(args.rho_high):
                        mu = min(max(mu * float(args.mu_down), mu_min), mu_max)
                        mu_note = "mu_down(rho_high)"
                    else:
                        mu_note = "mu_keep"

                print(f"[IK]   ACCEPT: rho={rho:.2f} -> {mu_note}, mu={mu:.3e}")

                accepted = True
                outer_rec["accepted"] = True
                outer_rec["accept_trial"] = trial_rec

                # Apply update
                cur = trial

                # Save artifacts for accepted step
                cl = extract_centerline_dim(cur.z_star_bar, M_list=cur.params.M_list, scales=cur.params.scales)
                accepted_centerlines.append(cl)

                accepted_meta.append(
                    {
                        "tag": "accept",
                        "iter": int(k),
                        "trial": int(t),
                        "alpha": float(alpha),
                        "err_mm": float(cur.err * 1e3),
                        "normE": float(cur.normE),
                        "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
                        "mu": float(mu),
                        "rho": float(rho),
                        "dI_norm": float(dI_trial_norm),
                        "I": [float(x) for x in cur.I.tolist()],
                        "p_tip": [float(x) for x in cur.p_tip.tolist()],
                    }
                )
                # Save this centerline to disk immediately
                save_centerline(
                    cl,
                    out_path=run_dir / "centerlines" / f"acc_{len(accepted_centerlines)-1:03d}.npy",
                )

                break

        if not accepted:
            # D) If no acceptable step, decide how to update mu.
            # If the step/prediction is numerically ineffective (very small pred_red or tiny dI),
            # we *decrease* mu to allow larger steps. Otherwise we increase mu to be more conservative.
            step_small = (step_norm <= max(1e-6, 0.5 * min_step)) if min_step > 0 else (step_norm <= 1e-6)
            pred_small = (abs(pred_reduction_1) < pred_floor)
            if step_small or pred_small:
                mu = min(max(mu * float(args.mu_down), mu_min), mu_max)
                note = "mu_down(ls_fail_small_step/pred)"
            else:
                mu = min(max(mu * float(args.mu_up), mu_min), mu_max)
                note = "mu_up(ls_fail)"
            print(f"[IK]   LS failed: {note} -> mu={mu:.3e} (step_norm={step_norm:.3e}, pred_red1={pred_reduction_1:.3e}, floor={pred_floor:.3e})")

        history["outer"].append(outer_rec)

    # Finalize
    final_tip = [float(x) for x in cur.p_tip.tolist()]
    history["final"] = {
        "I": [float(x) for x in cur.I.tolist()],
        "p_tip": final_tip,
        "err_mm": float(cur.err * 1e3),
        "normE": float(cur.normE),
        "lm_iter": int(getattr(cur.lm_stats, "n_iter", -1)),
        "success": bool(history["success"]),
    }
    history["accepted"] = accepted_meta

    # Save history
    hist_path = run_dir / "history.json"
    save_history_json(history, out_path=hist_path)
    print(f"[IK] saved: {str(hist_path)}")

    # Post plots
    p_tip_final = accepted_centerlines[-1][-1]
    img_path = run_dir / "centerlines_3d.png"
    plot_centerlines_3d(
        centerlines=accepted_centerlines,
        p_tip_init=p_tip_init,
        p_tip_final=p_tip_final,
        p_des=jax.device_get(p_des),
        out_path=img_path,
    )
    print(f"[IK] saved: {str(img_path)}")
    plot_metrics(history, out_dir=run_dir)
    print(f"[IK] saved metrics under: {str(run_dir)}")

    # Friendly summary
    if history["success"]:
        print(f"[IK] DONE: success, saved artifacts to: {str(run_dir)}")
    else:
        print(f"[IK] DONE: not converged, saved artifacts to: {str(run_dir)}")


if __name__ == "__main__":
    main()
