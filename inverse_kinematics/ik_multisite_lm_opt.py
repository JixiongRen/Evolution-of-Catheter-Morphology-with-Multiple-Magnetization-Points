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
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _MODULE_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import jax
import jax.numpy as jnp
import numpy as np

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from ik_diff import compute_tip_sensitivities_via_lm_adjoint
from ik_artifacts import extract_centerline_dim, save_centerline, save_history_json
from ik_branch_state import BranchStateRecorder, current_git_commit, json_sha256

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


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(str(tmp), str(path))
    except OSError:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_run_config(run_config_path: Path, *, updates: dict) -> None:
    if run_config_path.exists():
        try:
            cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}
    cfg.update(updates)
    _atomic_write_json(run_config_path, cfg)


def _new_runtime_profile(enabled: bool) -> dict:
    return {"enabled": bool(enabled), "sections": {}, "warmup": {}}


def _profile_add(profile: Optional[dict], section: str, label: str, elapsed_s: float) -> None:
    if not profile or not profile.get("enabled", False):
        return
    sections = profile.setdefault("sections", {})
    labels = sections.setdefault(str(section), {})
    item = labels.setdefault(str(label), {"count": 0, "total_s": 0.0, "first_s": None, "max_s": 0.0})
    elapsed_s = float(elapsed_s)
    item["count"] = int(item["count"]) + 1
    item["total_s"] = float(item["total_s"]) + elapsed_s
    item["first_s"] = elapsed_s if item["first_s"] is None else item["first_s"]
    item["max_s"] = max(float(item["max_s"]), elapsed_s)


def _write_runtime_profile(run_dir: Path, profile: Optional[dict]) -> None:
    if profile and profile.get("enabled", False):
        _atomic_write_json(run_dir / "runtime_profile.json", profile)


def _runtime_environment() -> dict[str, Any]:
    devices = []
    for dev in jax.devices():
        devices.append(
            {
                "id": getattr(dev, "id", None),
                "platform": getattr(dev, "platform", None),
                "device_kind": getattr(dev, "device_kind", None),
                "repr": str(dev),
            }
        )
    return {
        "code_commit": current_git_commit(_REPO_ROOT),
        "python_executable": sys.executable,
        "jax_version": getattr(jax, "__version__", None),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "devices": devices,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()

    # IK targets
    p.add_argument("--p_des", type=str, required=True, help="Desired tip position [m], format x,y,z")
    p.add_argument("--eps_p", type=float, default=10e-3, help="Stop if ||p_tip - p_des|| <= eps_p [m].")
    p.add_argument(
        "--sigma_p",
        type=float,
        default=None,
        help="Position residual scale sigma_p [m]. r_p=(p_tip-p_des)/sigma_p. Default: sigma_p=eps_p.",
    )

    # Optimization variables: currents and insertion depth
    p.add_argument("--I_min", type=float, default=None, help="Lower current bound. If not set, uses -I_max.")
    p.add_argument("--I_max", type=float, default=50.0, help="Symmetric current bound: I in [-I_max, I_max]")
    p.add_argument("--I0", type=str, default=None, help="Initial 8-coil currents [A], comma-separated. Defaults to random initialization.")
    p.add_argument("--x0", type=float, default=None, help="Initial insertion depth x [m]. If not set, uses --L_protrude.")
    p.add_argument("--x_min", type=float, default=None, help="Lower bound for insertion depth x [m]")
    p.add_argument("--x_max", type=float, default=None, help="Upper bound for insertion depth x [m]")
    p.add_argument("--x_ref", type=float, default=None, help="Reference insertion depth for regularization. Default: x0.")

    # Outer LM settings
    p.add_argument("--outer_max_iter", type=int, default=7000, help="Max outer LM iterations")
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

    p.add_argument(
        "--pred_red_tol",
        type=float,
        default=1e-10,
        help="Stop if predicted reduction is below this threshold (numerical floor).",
    )
    p.add_argument(
        "--step_I_tol",
        type=float,
        default=1e-6,
        help="Stop if max |ΔI_i| is below this threshold [A] (effective step is zero).",
    )
    p.add_argument(
        "--step_x_tol",
        type=float,
        default=1e-9,
        help="Stop if |Δx| is below this threshold [m] (effective step is zero).",
    )
    p.add_argument(
        "--act_red_tol",
        type=float,
        default=1e-10,
        help="Treat |act_red| below this threshold as numerical noise when pred_red is tiny.",
    )
    p.add_argument("--p2_trust_region", action="store_true", help="Enable conservative P2 trust-region acceptance checks.")
    p.add_argument("--p2_terminal_error_first", action="store_true", help="Require terminal error improvement before regularization-only improvements.")
    p.add_argument("--p2_reliable_subspace", action="store_true", help="Project the outer LM step onto task-space reliable singular directions.")
    p.add_argument("--p2_reliable_task_rcond", type=float, default=1e-3, help="Relative cutoff for task-space reliable singular directions.")
    p.add_argument("--p2_min_full_task_rank", type=int, default=2, help="Minimum task-space rank required for a full control update.")
    p.add_argument("--p2_max_backtracks", type=int, default=0, help="Maximum number of backtracking retries for a rejected candidate.")
    p.add_argument("--p2_backtrack_shrink", type=float, default=0.5, help="Step shrink ratio used by P2 backtracking.")
    p.add_argument(
        "--p2_disable_Ix_jacobian_below_dx",
        type=float,
        default=-1.0,
        help="Disable the r_Ix current Jacobian when |x-x_prev| is below this threshold [m]. Negative keeps legacy behavior.",
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
    p.add_argument("--x_sensitivity_method", choices=["implicit", "fd"], default="implicit", help="How to compute dp/dx in the outer IK linearization")
    p.add_argument(
        "--sensitivity_mode",
        choices=["solver_response", "gn_svd", "stationarity_exact", "residual_svd"],
        default="residual_svd",
        help="Implicit sensitivity linearization used by the supervised IK outer loop.",
    )
    p.add_argument("--sensitivity_solve_mode", choices=["solve", "svd"], default="svd", help="Linear solve mode for square implicit sensitivity systems")
    p.add_argument("--sensitivity_svd_rcond", type=float, default=1e-9, help="Relative singular-value cutoff for implicit sensitivity diagnostics")
    p.add_argument("--sensitivity_spectral_filter", choices=["hard", "tikhonov", "hybrid"], default="hard", help="Residual-space spectral filter for residual_svd sensitivity")
    p.add_argument("--sensitivity_spectral_damping", type=float, default=0.0, help="Absolute damping used by tikhonov or hybrid residual-space filters")
    p.add_argument("--store_accepted_jacobian", action="store_true", help="Store dense FK Jacobian on accepted trials for legacy diagnostics")
    p.add_argument("--sensitivity_conditioning", action="store_true", help="Compute dense conditioning diagnostics for implicit sensitivity")

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
    p.add_argument("--runtime_profile", action="store_true", help="Write runtime_profile.json with FK/IK timing buckets")
    p.add_argument("--warmup_fk_compile", action="store_true", help="Precompile fixed-shape FK residual/Jacobian/one-step solve before IK")
    p.add_argument("--warmup_ik_adjoint", action="store_true", help="Precompute and reuse the first dp/dI adjoint sensitivity")
    p.add_argument("--branch_state_capture", action="store_true", help="Save every FK candidate state for branch replay diagnostics")

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
    p_tip_true: Array
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
    warm_start_z_bar: Optional[Array] = None


def _true_rigid_distal_tip(engine: ForwardKinematicsEngine, obs: dict[str, Array]) -> Array:
    """Return the physical distal end of the last rigid segment."""
    if "terminal_p_dim" in obs:
        return jnp.asarray(obs["terminal_p_dim"], dtype=jnp.float64).reshape(3,)
    rigid_distal_p = jnp.asarray(obs.get("rigid_distal_p_dim", obs.get("rigid_mid_p_dim")), dtype=jnp.float64)
    if rigid_distal_p.shape[0] == 0:
        return jnp.asarray(obs["tip_p_dim"], dtype=jnp.float64).reshape(3,)
    return rigid_distal_p[-1].reshape(3,)


def _tip_report_fields(pack: EvalPack, p_des: Array) -> dict[str, Any]:
    """Return canonical terminal and flexible-tip metrics for JSON artifacts."""
    p_flex = np.asarray(jax.device_get(pack.p_tip), dtype=float).reshape(3)
    p_true = np.asarray(jax.device_get(pack.p_tip_true), dtype=float).reshape(3)
    p_target = np.asarray(jax.device_get(p_des), dtype=float).reshape(3)
    err_flex = float(np.linalg.norm(p_flex - p_target))
    err_true = float(np.linalg.norm(p_true - p_target))
    return {
        "p_tip_m": [float(v) for v in p_true.tolist()],
        "p_tip_flex_m": [float(v) for v in p_flex.tolist()],
        "p_tip_true_m": [float(v) for v in p_true.tolist()],
        "err_m": err_true,
        "err_mm": float(err_true * 1e3),
        "err_flex_m": err_flex,
        "err_flex_mm": float(err_flex * 1e3),
        "err_true_m": err_true,
        "err_true_mm": float(err_true * 1e3),
    }


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
    stats_include_jacobian: bool = True,
    runtime_profile: Optional[dict] = None,
    profile_label: str = "fk_eval",
) -> EvalPack:
    if override_z0_bar is not None:
        warm_start_z_bar = jnp.asarray(override_z0_bar, dtype=jnp.float64)
    elif getattr(engine, "_last_z_bar", None) is not None:
        warm_start_z_bar = jnp.asarray(engine._last_z_bar, dtype=jnp.float64)
    else:
        warm_start_z_bar = None

    t0 = time.perf_counter()
    z_star_bar, params, _meshes, ok, lm_stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(x),
        warm_start=True,
        override_z0_bar=override_z0_bar,
        return_stats=True,
        stats_include_jacobian=bool(stats_include_jacobian),
    )
    _profile_add(runtime_profile, "fk_solve", profile_label, time.perf_counter() - t0)

    normE = float(lm_stats.normE) if lm_stats is not None else float("nan")
    ok_strict = bool(lm_stats.ok_strict) if lm_stats is not None else bool(ok)
    cls = _classify_equilibrium(ok_strict=ok_strict, normE=normE, tol_good=tol_good, tol_weak=tol_weak, tol_bad=tol_bad)

    # Tip position (SI) via FK engine observation interface
    t0 = time.perf_counter()
    obs = engine.query_sites(z_bar=z_star_bar, scales=params.scales)
    p_tip = jnp.asarray(obs["tip_p_dim"], dtype=jnp.float64)
    p_tip_true = jnp.asarray(_true_rigid_distal_tip(engine, obs), dtype=jnp.float64)
    _profile_add(runtime_profile, "fk_observe", profile_label, time.perf_counter() - t0)

    penalty = 0.0
    if (not jnp.all(jnp.isfinite(p_tip_true))) or (cls in ("BAD", "REJECT")):
        penalty = float(penalty_bad)

    r, cost, e_p, err = _compute_residual_vector(
        p_tip=p_tip_true,
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
        p_tip_true=jnp.asarray(p_tip_true, dtype=jnp.float64),
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
        warm_start_z_bar=warm_start_z_bar,
    )


def _ensure_lm_jacobian_stats(
    engine: ForwardKinematicsEngine,
    pack: EvalPack,
    *,
    runtime_profile: Optional[dict] = None,
) -> EvalPack:
    """Attach the dense FK Jacobian only when the next adjoint step needs it."""
    stats = pack.lm_stats
    if stats is None or getattr(stats, "J", None) is not None:
        return pack
    t0 = time.perf_counter()
    J = engine._solver_cached.jacobian_jit(pack.z_star_bar, pack.params, method=engine.jac_method)
    if hasattr(J, "block_until_ready"):
        J.block_until_ready()
    stats.J = jnp.array(J)
    _profile_add(runtime_profile, "jacobian_stats", "accepted_trial", time.perf_counter() - t0)
    return pack


def _compute_local_fk_jacobian(
    engine: ForwardKinematicsEngine,
    pack: EvalPack,
    *,
    runtime_profile: Optional[dict] = None,
    label: str = "implicit_adjoint_jz",
) -> Array:
    """Materialize ``dE/dz`` for sensitivity without storing it in LMStats."""
    t0 = time.perf_counter()
    J = engine._solver_cached.jacobian_jit(pack.z_star_bar, pack.params, method=engine.jac_method)
    if hasattr(J, "block_until_ready"):
        J.block_until_ready()
    _profile_add(runtime_profile, "sensitivity", label, time.perf_counter() - t0)
    return jnp.asarray(J, dtype=jnp.float64)


def _compute_dp_dx_fd(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    x: float,
    h: float,
    z0_bar: Optional[Array],
    clip_bounds: Tuple[float, float],
    debug_flag: bool = False,
    runtime_profile: Optional[dict] = None,
) -> Array:
    """Finite difference dp_tip/dx along x, returns (3,)."""
    if not debug_flag:
        x_min, x_max = clip_bounds
        h = float(h)
        if h <= 0.0:
            raise ValueError("fd step h must be > 0")
        xp = min(float(x_max), float(x) + h)
        xm = max(float(x_min), float(x) - h)
        if abs(xp - xm) < 1e-12:
            return jnp.zeros((3,), dtype=jnp.float64)

        t0 = time.perf_counter()
        zp, params_p, _, _, _ = engine.solve_with_stats(
            coil_currents=I,
            L_protrude=float(xp),
            warm_start=True,
            override_z0_bar=z0_bar,
            return_stats=False,
        )
        _profile_add(runtime_profile, "fd_x_fk", "plus", time.perf_counter() - t0)
        t0 = time.perf_counter()
        zm, params_m, _, _, _ = engine.solve_with_stats(
            coil_currents=I,
            L_protrude=float(xm),
            warm_start=True,
            override_z0_bar=z0_bar,
            return_stats=False,
        )
        _profile_add(runtime_profile, "fd_x_fk", "minus", time.perf_counter() - t0)

        obs_p = engine.query_sites(z_bar=zp, scales=params_p.scales)
        obs_m = engine.query_sites(z_bar=zm, scales=params_m.scales)
        pp = jnp.asarray(_true_rigid_distal_tip(engine, obs_p), dtype=jnp.float64)
        pm = jnp.asarray(_true_rigid_distal_tip(engine, obs_m), dtype=jnp.float64)
        dpdx = (pp - pm) / float(xp - xm)
        return jnp.asarray(dpdx, dtype=jnp.float64)
    else:
        return jnp.zeros((3,), dtype=jnp.float64)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.print_devices:
        print("[IK-LM] JAX devices:", jax.devices())

    run_dir = _make_run_dir(args.out_dir, args.run_name)
    runtime_profile = _new_runtime_profile(bool(args.runtime_profile))
    runtime_environment = _runtime_environment()
    if runtime_profile.get("enabled", False):
        runtime_profile["environment"] = runtime_environment
    run_config_path = run_dir / "run_config.json"
    poses_dir = (run_dir / "poses")
    poses_trial_dir = (poses_dir / "trial")
    poses_accept_dir = (poses_dir / "accept")
    poses_trial_dir.mkdir(parents=True, exist_ok=True)
    poses_accept_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.json"
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
    t0_engine = time.perf_counter()
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
    _profile_add(runtime_profile, "setup", "engine_build", time.perf_counter() - t0_engine)

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
    p2_trust_region = bool(args.p2_trust_region)
    p2_terminal_error_first = bool(args.p2_terminal_error_first)
    p2_reliable_subspace = bool(args.p2_reliable_subspace)
    p2_reliable_task_rcond = float(args.p2_reliable_task_rcond)
    p2_min_full_task_rank = int(args.p2_min_full_task_rank)
    p2_max_backtracks = max(0, int(args.p2_max_backtracks))
    p2_backtrack_shrink = float(args.p2_backtrack_shrink)
    p2_disable_Ix_jacobian_below_dx = float(args.p2_disable_Ix_jacobian_below_dx)
    if not (0.0 < p2_backtrack_shrink < 1.0):
        raise ValueError("--p2_backtrack_shrink must be in (0, 1)")

    ridge_adj = float(args.ridge_adjoint)
    fd_x_eps = float(args.fd_x_eps)
    x_sensitivity_method = str(args.x_sensitivity_method)
    sensitivity_mode = str(args.sensitivity_mode)
    sensitivity_solve_mode = str(args.sensitivity_solve_mode)
    sensitivity_svd_rcond = float(args.sensitivity_svd_rcond)
    sensitivity_spectral_filter = str(args.sensitivity_spectral_filter)
    sensitivity_spectral_damping = float(args.sensitivity_spectral_damping)
    sensitivity_conditioning = bool(args.sensitivity_conditioning)
    sensitivity_config = {
        "x_sensitivity_method": x_sensitivity_method,
        "sensitivity_mode": sensitivity_mode,
        "sensitivity_solve_mode": sensitivity_solve_mode,
        "sensitivity_svd_rcond": sensitivity_svd_rcond,
        "sensitivity_spectral_filter": sensitivity_spectral_filter,
        "sensitivity_spectral_damping": sensitivity_spectral_damping,
        "ridge_adjoint": ridge_adj,
        "sensitivity_conditioning": sensitivity_conditioning,
    }

    # Initialize u from user-specified or randomized I0, x=x0
    n_coils = 8
    if args.I0 is not None:
        I0_values = [float(v) for v in str(args.I0).split(",")]
        if len(I0_values) != n_coils:
            raise ValueError("--I0 must contain exactly 8 comma-separated currents.")
        I0 = jnp.asarray(I0_values, dtype=jnp.float64)
        I0 = jnp.clip(I0, float(I_min), float(I_max))
    else:
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
    run_config_path.write_text(
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
                "sensitivity_config": sensitivity_config,
                "runtime_environment": runtime_environment,
                "sigmas_from_w": {
                    "sigma_I": (1.0 / w_I) if w_I > 0 else None,
                    "sigma_x": (1.0 / w_x) if w_x > 0 else None,
                    "sigma_dI": (1.0 / w_dI) if w_dI > 0 else None,
                    "sigma_dx": (1.0 / w_dx) if w_dx > 0 else None,
                    "sigma_Ix": (1.0 / w_Ix) if w_Ix > 0 else None,
                },
                "step_limits": {"step_I_max": step_I_max, "step_x_max": step_x_max},
                "p2_outer_solver": {
                    "trust_region": p2_trust_region,
                    "terminal_error_first": p2_terminal_error_first,
                    "reliable_subspace": p2_reliable_subspace,
                    "reliable_task_rcond": p2_reliable_task_rcond,
                    "min_full_task_rank": p2_min_full_task_rank,
                    "max_backtracks": p2_max_backtracks,
                    "backtrack_shrink": p2_backtrack_shrink,
                    "disable_Ix_jacobian_below_dx": p2_disable_Ix_jacobian_below_dx,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history: dict = {
        "meta": {
            "run_dir": str(run_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "p_des": [float(v) for v in p_des.tolist()],
            "eps_p_m": float(eps_p),
            "sigma_p_m": float(sigma_p),
            "sensitivity_config": sensitivity_config,
        },
        "outer": [],
    }
    branch_recorder = BranchStateRecorder(
        run_dir=run_dir,
        enabled=bool(args.branch_state_capture),
        metadata={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code_commit": current_git_commit(_REPO_ROOT),
            "argv_hash": json_sha256(vars(args)),
            "fk_cfg_hash": json_sha256(cfg),
            "geometry_hash": json_sha256(
                {
                    "flex_lengths": flex_lengths,
                    "rigid_lengths": rigid_lengths,
                    "M_list": M_list,
                    "L1_min": L1_min,
                    "x_bounds": [x_min, x_max],
                    "p0_dim": p0_dim,
                    "Q0": Q0,
                    "axis_body": axis_body,
                }
            ),
            "material_hash": json_sha256(
                {
                    "flex_d_outer": flex_d_outer,
                    "flex_E": flex_E,
                    "flex_G": flex_G,
                    "flex_rho": flex_rho,
                    "rigid_d_outer": rigid_d_outer,
                    "rigid_rho": rigid_rho,
                }
            ),
            "magnet_hash": json_sha256(
                {
                    "enable_magnetics": enable_magnetics,
                    "calib_file": calib_file,
                    "actuation_table_pkl": actuation_table_pkl,
                    "m_body_list": m_body_list,
                }
            ),
            "solver_hash": json_sha256(
                {
                    "max_iter": int(_get("max_iter", args.max_iter)),
                    "tol": float(_get("tol", args.tol)),
                    "lm_damping": float(_get("lm_damping", args.lm_damping)),
                    "jac_method": str(_get("jac_method", args.jac_method)),
                    "outer_max_iter": int(args.outer_max_iter),
                    "eps_p": eps_p,
                    "sigma_p": sigma_p,
                }
            ),
            "sensitivity_hash": json_sha256(sensitivity_config),
            "branch_gate_note": "A saved state is traceable evidence only; same-branch claims require replay and roundtrip gates.",
        },
    )
    last_branch_eval_id: Optional[str] = None

    def _save_pose(*, which: str, iter_idx: int, pack: EvalPack) -> Path:
        t0 = time.perf_counter()
        cl = extract_centerline_dim(pack.z_star_bar, M_list=engine.M_list, scales=pack.params.scales)
        out_dir = poses_accept_dir if which == "accept" else poses_trial_dir
        out_path = out_dir / f"centerline_iter_{int(iter_idx):04d}.npy"
        save_centerline(cl, out_path)
        _profile_add(runtime_profile, "io", f"save_pose_{which}", time.perf_counter() - t0)
        return out_path

    def _save_branch_state(
        *,
        phase: str,
        iter_idx: int,
        pack: EvalPack,
        status: str,
        accept: bool,
        parent_eval_id: Optional[str],
        pose_path: Optional[Path],
        outer: Optional[dict[str, Any]] = None,
    ):
        t0 = time.perf_counter()
        centerline = extract_centerline_dim(pack.z_star_bar, M_list=engine.M_list, scales=pack.params.scales)
        centerline_relpath = str(pose_path.relative_to(run_dir)) if pose_path is not None else None
        record = branch_recorder.save(
            phase=phase,
            iter_idx=int(iter_idx),
            pack=pack,
            p_des=p_des,
            centerline_m=centerline,
            centerline_relpath=centerline_relpath,
            status=status,
            accept=bool(accept),
            parent_eval_id=parent_eval_id,
            outer=outer,
        )
        if record is not None:
            _profile_add(runtime_profile, "io", f"branch_state_{phase}", time.perf_counter() - t0)
        return record

    def _branch_record_fields(record) -> dict[str, Any]:
        if record is None:
            return {}
        return {
            "branch_state": {
                "eval_id": record.eval_id,
                "npz_path": str(record.npz_path.relative_to(run_dir)),
                "z_star_hash": record.z_hash,
                "warm_start_hash": record.warm_start_hash,
                "centerline_hash": record.centerline_hash,
            }
        }

    def _append_history(*, d: dict) -> None:
        t0 = time.perf_counter()
        history["outer"].append(d)
        save_history_json(history, history_path)
        _profile_add(runtime_profile, "io", "save_history", time.perf_counter() - t0)

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

    if bool(args.warmup_fk_compile):
        t0 = time.perf_counter()
        warmup = engine.warm_up(
            coil_currents=jnp.zeros((8,), dtype=jnp.float64),
            L_protrude=float(x0),
            include_jacobian=True,
            include_lm_step=True,
        )
        runtime_profile["warmup"] = warmup
        _profile_add(runtime_profile, "setup", "warmup_fk_compile", time.perf_counter() - t0)
        _write_runtime_profile(run_dir, runtime_profile)

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
        stats_include_jacobian=False,
        runtime_profile=runtime_profile,
        profile_label="init",
    )
    z_warm = cur.z_star_bar

    init_pose_path = _save_pose(which="accept", iter_idx=0, pack=cur)
    init_branch_record = _save_branch_state(
        phase="init",
        iter_idx=0,
        pack=cur,
        status="INIT",
        accept=True,
        parent_eval_id=None,
        pose_path=init_pose_path,
        outer={"lam": float(lam)},
    )
    if init_branch_record is not None:
        last_branch_eval_id = init_branch_record.eval_id
    init_tip_fields = _tip_report_fields(cur, p_des)
    _update_run_config(
        run_config_path,
        updates={
            "p_des": [float(v) for v in p_des.tolist()],
            "init": {
                "x_m": float(cur.x),
                "I_A": [float(v) for v in np.asarray(jax.device_get(cur.I)).reshape(-1,).tolist()],
                **init_tip_fields,
                "centerline_npy": str(init_pose_path.relative_to(run_dir)),
                **_branch_record_fields(init_branch_record),
            },
        },
    )

    _append_history(
        d={
            "iter": 0,
            "status": "INIT",
            "accept": True,
            "lam": float(lam),
            "rho": None,
            "pred_red": None,
            "act_red": None,
            "du_eff_norm": 0.0,
            "dI_max_abs": 0.0,
            "dI_norm": 0.0,
            "dx_m": 0.0,
            "dx_abs_m": 0.0,
            "x_m": float(cur.x),
            "I_A": [float(v) for v in np.asarray(jax.device_get(cur.I)).reshape(-1,).tolist()],
            "dI_A": [0.0 for _ in range(int(n_coils))],
            **init_tip_fields,
            "cost": float(cur.cost),
            "cls": str(cur.cls),
            "normE": float(cur.normE),
            "pose": {
                "centerline_npy": str(init_pose_path.relative_to(run_dir)),
                "which": "accept",
            },
            **_branch_record_fields(init_branch_record),
        }
    )

    print(
        f"[IK-LM] iter=000 cls={cur.cls:<5s} ok_strict={cur.ok_strict} ||E||={cur.normE:.3e} "
        f"x={cur.x:.6f} p_terminal={cur.p_tip_true.tolist()} ||e||={cur.err*1e3:.3f}mm cost={cur.cost:.6e}"
    )

    if cur.err <= eps_p and cur.cls != "REJECT":
        print(f"[IK-LM] Converged at init: ||e||={cur.err*1e3:.3f}mm")
        _write_runtime_profile(run_dir, runtime_profile)
        return

    jac_res_cached: Optional[Any] = None
    if bool(args.warmup_ik_adjoint):
        try:
            t0 = time.perf_counter()
            Jz_local = _compute_local_fk_jacobian(engine, cur, runtime_profile=runtime_profile, label="warmup_implicit_adjoint_jz")
            jac_res_cached = compute_tip_sensitivities_via_lm_adjoint(
                z_star_bar=cur.z_star_bar,
                params=cur.params,
                lm_stats=cur.lm_stats,
                coil_currents=cur.I,
                ridge=ridge_adj,
                J_E_z=Jz_local,
                materialize_jacobian=True,
                compute_conditioning=sensitivity_conditioning,
                sensitivity_mode=sensitivity_mode,
                solve_mode=sensitivity_solve_mode,
                svd_rcond=sensitivity_svd_rcond,
                spectral_filter=sensitivity_spectral_filter,
                spectral_damping=sensitivity_spectral_damping,
            )
            if hasattr(jac_res_cached.J_p_I, "block_until_ready"):
                jac_res_cached.J_p_I.block_until_ready()
            if hasattr(jac_res_cached.J_p_x, "block_until_ready"):
                jac_res_cached.J_p_x.block_until_ready()
            elapsed = time.perf_counter() - t0
            runtime_profile.setdefault("warmup", {})["ik_adjoint_s"] = float(elapsed)
            runtime_profile.setdefault("warmup", {})["ik_adjoint_status"] = str(jac_res_cached.status)
            runtime_profile.setdefault("warmup", {})["ik_adjoint_method"] = str(jac_res_cached.method)
            _profile_add(runtime_profile, "setup", "warmup_ik_adjoint", elapsed)
            _write_runtime_profile(run_dir, runtime_profile)
        except Exception as e:
            jac_res_cached = None
            print(f"[IK-LM] WARNING: warmup implicit adjoint failed: {repr(e)}")

    for k in range(1, int(args.outer_max_iter) + 1):
        # 1) Sensitivities dp/dI and dp/dx (at current accepted state)
        sensitivity_info = {
            "x_method": str(x_sensitivity_method),
            "sensitivity_mode": str(sensitivity_mode),
            "sensitivity_solve_mode": str(sensitivity_solve_mode),
            "sensitivity_svd_rcond": float(sensitivity_svd_rcond),
            "sensitivity_spectral_filter": str(sensitivity_spectral_filter),
            "sensitivity_spectral_damping": float(sensitivity_spectral_damping),
            "ridge_adjoint": float(ridge_adj),
            "jz_source": "not_computed",
            "status": "not_computed",
            "method": "none",
            "stationarity_norm": None,
            "condition_number": None,
            "effective_rank": None,
            "sv_min": None,
            "sv_max": None,
            "spectral_filter": "none",
            "spectral_lambda": None,
            "spectral_cutoff": None,
            "retained_spectral_energy": None,
        }
        dp_dx = jnp.zeros((3,), dtype=jnp.float64)
        try:
            if jac_res_cached is not None:
                t0 = time.perf_counter()
                jac_res = jac_res_cached
                jac_res_cached = None
                sensitivity_jz_source = "warmup_cached"
                _profile_add(runtime_profile, "sensitivity", "implicit_adjoint_reused", time.perf_counter() - t0)
            else:
                t0 = time.perf_counter()
                Jz_local = _compute_local_fk_jacobian(engine, cur, runtime_profile=runtime_profile, label="implicit_adjoint_jz")
                sensitivity_jz_source = "computed"
                jac_res = compute_tip_sensitivities_via_lm_adjoint(
                    z_star_bar=cur.z_star_bar,
                    params=cur.params,
                    lm_stats=cur.lm_stats,
                    coil_currents=cur.I,
                    ridge=ridge_adj,
                    J_E_z=Jz_local,
                    materialize_jacobian=True,
                    compute_conditioning=sensitivity_conditioning,
                    sensitivity_mode=sensitivity_mode,
                    solve_mode=sensitivity_solve_mode,
                    svd_rcond=sensitivity_svd_rcond,
                    spectral_filter=sensitivity_spectral_filter,
                    spectral_damping=sensitivity_spectral_damping,
                )
                _profile_add(runtime_profile, "sensitivity", "implicit_adjoint", time.perf_counter() - t0)
            Jp_I = jnp.asarray(jac_res.J_p_I, dtype=jnp.float64)  # (3, nI)
            if x_sensitivity_method == "implicit":
                dp_dx = jnp.asarray(jac_res.J_p_x, dtype=jnp.float64).reshape(3,)
            sensitivity_info = {
                "x_method": str(x_sensitivity_method),
                "sensitivity_mode": str(sensitivity_mode),
                "sensitivity_solve_mode": str(sensitivity_solve_mode),
                "sensitivity_svd_rcond": float(sensitivity_svd_rcond),
                "sensitivity_spectral_filter": str(sensitivity_spectral_filter),
                "sensitivity_spectral_damping": float(sensitivity_spectral_damping),
                "ridge_adjoint": float(ridge_adj),
                "jz_source": sensitivity_jz_source,
                "status": str(jac_res.status),
                "method": str(jac_res.method),
                "stationarity_norm": float(jac_res.stationarity_norm),
                "condition_number": None if jac_res.condition_number is None else float(jac_res.condition_number),
                "effective_rank": None if jac_res.effective_rank is None else int(jac_res.effective_rank),
                "sv_min": None if jac_res.sv_min is None else float(jac_res.sv_min),
                "sv_max": None if jac_res.sv_max is None else float(jac_res.sv_max),
                "spectral_filter": str(getattr(jac_res, "spectral_filter", "none")),
                "spectral_lambda": float(getattr(jac_res, "spectral_lambda", 0.0)),
                "spectral_cutoff": None
                if getattr(jac_res, "spectral_cutoff", None) is None
                else float(getattr(jac_res, "spectral_cutoff")),
                "retained_spectral_energy": None
                if getattr(jac_res, "retained_spectral_energy", None) is None
                else float(getattr(jac_res, "retained_spectral_energy")),
            }
        except Exception as e:
            print(f"[IK-LM] WARNING: implicit adjoint failed: {repr(e)}")
            Jp_I = jnp.zeros((3, n_coils), dtype=jnp.float64)
            sensitivity_info["status"] = "failed"
            sensitivity_info["error"] = repr(e)
            sensitivity_info["jz_source"] = "failed"

        if x_sensitivity_method == "fd" or sensitivity_info.get("status") == "failed":
            dp_dx = _compute_dp_dx_fd(
                engine,
                I=cur.I,
                x=cur.x,
                h=fd_x_eps,
                z0_bar=z_warm,
                clip_bounds=(x_min, x_max),
                debug_flag=False,
                runtime_profile=runtime_profile,
            )  # (3,)
            sensitivity_info["x_method"] = "fd_fallback" if sensitivity_info.get("status") == "failed" else "fd"

        # 2) Build Jacobian wrt u
        # Residual order:
        #   [r_p(3), r_I(n), r_x(1), r_dI(n), r_dx(1), r_Ix(n), r_pen(1)]
        I_u, x_u, dI_du, dx_du = _u_to_Ix(u, I_max=I_max, x_min=x_min, x_max=x_max)
        dI_du = jnp.asarray(dI_du, dtype=jnp.float64).reshape(-1,)

        # Chain: dp/duI = dp/dI * dI/du
        J_p_uI = Jp_I * dI_du.reshape(1, -1)  # (3,n)
        J_p_ux = dp_dx.reshape(3, 1) * float(dx_du)  # (3,1)
        J_top = jnp.concatenate([J_p_uI, J_p_ux], axis=1) / float(sigma_p)  # scale r_p by 1/sigma_p
        task_svals = jnp.linalg.svd(J_top, compute_uv=False)
        task_sv_max = float(jnp.max(task_svals)) if task_svals.size else 0.0
        task_cutoff = task_sv_max * max(float(p2_reliable_task_rcond), 0.0)
        task_rank = int(jnp.sum(task_svals > max(task_cutoff, 1e-12)))
        p2_force_reliable = bool(p2_reliable_subspace or task_rank < int(p2_min_full_task_rank) or sensitivity_info.get("status") == "failed")

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
        ix_jacobian_enabled = not (p2_disable_Ix_jacobian_below_dx >= 0.0 and abs(dx_to_prev) <= p2_disable_Ix_jacobian_below_dx)
        if ix_jacobian_enabled:
            J_Ix_uI = (float(w_Ix) / denom) * jnp.diag(dI_du)
        else:
            J_Ix_uI = jnp.zeros((n_coils, n_coils), dtype=jnp.float64)
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
            if p2_force_reliable:
                U_task, S_task, Vh_task = jnp.linalg.svd(J_top, full_matrices=False)
                reliable_dim = int(jnp.sum(S_task > max(task_cutoff, 1e-12)))
                if reliable_dim <= 0:
                    du = jnp.zeros((n_coils + 1,), dtype=jnp.float64)
                else:
                    V_rel = Vh_task[:reliable_dim, :].T
                    J_rel = J @ V_rel
                    H_rel = J_rel.T @ J_rel
                    g_rel = J_rel.T @ r
                    diag_rel = jnp.clip(jnp.diag(H_rel), a_min=1e-12)
                    A_rel = H_rel + float(lam) * jnp.diag(diag_rel)
                    du = V_rel @ (-jnp.linalg.solve(A_rel, g_rel))
            else:
                reliable_dim = n_coils + 1
                du = -jnp.linalg.solve(A, g)
        except Exception as e:
            print(f"[IK-LM] ERROR: linear solve failed at iter={k}: {repr(e)}")
            lam = min(lam * float(args.lam_up), 1e12)
            continue

        # 4) Physical trust-region evaluation in (I, x).
        max_backtracks = p2_max_backtracks if p2_trust_region else 0
        accept = False
        trial = None
        trial_branch_record = None
        u_trial = u
        dI_clip = jnp.zeros((n_coils,), dtype=jnp.float64)
        dx_clip = 0.0
        dI_norm = 0.0
        dI_max_abs = 0.0
        dx_abs = 0.0
        du_eff_norm = 0.0
        rho = 0.0
        pred_red = 0.0
        act_red = 0.0
        terminal_improved = False
        cost_improved = False
        stop_outer_loop = False

        for backtrack_idx in range(max_backtracks + 1):
            backtrack_scale = float(p2_backtrack_shrink) ** int(backtrack_idx)
            u_prop = u + backtrack_scale * du
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

            r_lin = r + J @ du_eff
            pred_red = 0.5 * float(jnp.dot(r, r) - jnp.dot(r_lin, r_lin))

            if (dI_max_abs <= float(args.step_I_tol)) and (dx_abs <= float(args.step_x_tol)):
                if backtrack_idx == 0:
                    print(
                        f"[IK-LM] stop: effective step is zero (|dI|_max={dI_max_abs:.3e}A, |dx|={dx_abs:.3e}m). "
                        f"Likely parameter saturation / clipping / numerical floor."
                    )
                    stop_outer_loop = True
                break

            if pred_red <= float(args.pred_red_tol):
                if backtrack_idx == 0:
                    print(f"[IK-LM] stop: pred_red={pred_red:.3e} <= pred_red_tol={float(args.pred_red_tol):.3e}")
                    stop_outer_loop = True
                break

            # 5) Evaluate candidate.
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
                stats_include_jacobian=False,
                runtime_profile=runtime_profile,
                profile_label="trial",
            )

            act_red = float(cur.cost - trial.cost)
            rho = act_red / max(pred_red, 1e-18)
            terminal_improved = bool(trial.err < cur.err)
            cost_improved = bool(trial.cost < cur.cost)

            if p2_terminal_error_first:
                accept = bool(terminal_improved and trial.cls != "REJECT")
            elif (pred_red <= 10.0 * float(args.pred_red_tol)) and (abs(act_red) <= float(args.act_red_tol)):
                accept = bool(trial.cls != "REJECT")
            else:
                accept = bool(cost_improved and (rho > 0.0) and trial.cls != "REJECT")
            status = "ACCEPT" if accept else "REJECT"

            trial_pose_path = _save_pose(which="trial", iter_idx=k * 100 + backtrack_idx, pack=trial)
            trial_tip_fields = _tip_report_fields(trial, p_des)
            trial_outer_info = {
                "lam": float(lam),
                "rho": float(rho),
                "pred_red": float(pred_red),
                "act_red": float(act_red),
                "terminal_improved": bool(terminal_improved),
                "cost_improved": bool(cost_improved),
                "du_eff_norm": float(du_eff_norm),
                "dI_max_abs": float(dI_max_abs),
                "dI_norm": float(dI_norm),
                "dx_m": float(dx_clip),
                "dx_abs_m": float(dx_abs),
                "backtrack_index": int(backtrack_idx),
                "backtrack_scale": float(backtrack_scale),
                "task_rank": int(task_rank),
                "task_singular_values": [float(v) for v in np.asarray(jax.device_get(task_svals)).reshape(-1,).tolist()],
                "reliable_subspace_enabled": bool(p2_force_reliable),
                "reliable_subspace_dim": int(reliable_dim),
                "ix_jacobian_enabled": bool(ix_jacobian_enabled),
                "p2_terminal_error_first": bool(p2_terminal_error_first),
                "sensitivity": sensitivity_info,
            }
            trial_branch_record = _save_branch_state(
                phase="trial",
                iter_idx=k * 100 + backtrack_idx,
                pack=trial,
                status=status,
                accept=bool(accept),
                parent_eval_id=last_branch_eval_id,
                pose_path=trial_pose_path,
                outer=trial_outer_info,
            )

            _append_history(
                d={
                    "iter": int(k),
                    "backtrack_index": int(backtrack_idx),
                    "backtrack_scale": float(backtrack_scale),
                    "status": str(status),
                    "accept": bool(accept),
                    "lam": float(lam),
                    "rho": float(rho),
                    "pred_red": float(pred_red),
                    "act_red": float(act_red),
                    "terminal_improved": bool(terminal_improved),
                    "cost_improved": bool(cost_improved),
                    "task_rank": int(task_rank),
                    "task_singular_values": [float(v) for v in np.asarray(jax.device_get(task_svals)).reshape(-1,).tolist()],
                    "reliable_subspace_enabled": bool(p2_force_reliable),
                    "reliable_subspace_dim": int(reliable_dim),
                    "ix_jacobian_enabled": bool(ix_jacobian_enabled),
                    "du_eff_norm": float(du_eff_norm),
                    "dI_max_abs": float(dI_max_abs),
                    "dI_norm": float(dI_norm),
                    "dx_m": float(dx_clip),
                    "dx_abs_m": float(dx_abs),
                    "x_m": float(trial.x),
                    "I_A": [float(v) for v in np.asarray(jax.device_get(I_trial)).reshape(-1,).tolist()],
                    "dI_A": [float(v) for v in np.asarray(jax.device_get(dI_clip)).reshape(-1,).tolist()],
                    **trial_tip_fields,
                    "cost": float(trial.cost),
                    "cls": str(trial.cls),
                    "normE": float(trial.normE),
                    "sensitivity": sensitivity_info,
                    "pose": {
                        "centerline_npy": str(trial_pose_path.relative_to(run_dir)),
                        "which": "trial",
                    },
                    **_branch_record_fields(trial_branch_record),
                }
            )

            print(
                f"[IK-LM] iter={k:03d}.{backtrack_idx} lam={lam:.2e} rho={rho:.3f} {status} "
                f"rank={task_rank} rel_dim={reliable_dim} scale={backtrack_scale:.3f} "
                f"du_eff={du_eff_norm:.2e} |dI|_max={dI_max_abs:.3f}A ||dI||={dI_norm:.3f}A |dx|={dx_abs*1e3:.3f}mm "
                f"pred_red={pred_red:.3e} act_red={act_red:.3e} terr={terminal_improved} "
                f"cls={trial.cls:<5s} ||E||={trial.normE:.3e} x={trial.x:.6f} ||e||={trial.err*1e3:.3f}mm cost={trial.cost:.6e}"
            )

            if accept:
                break

        if stop_outer_loop:
            break
        if trial is None:
            lam = min(lam * float(args.lam_up), 1e12)
            continue

        # 6) Update lambda and state
        if accept:
            if bool(args.store_accepted_jacobian):
                trial = _ensure_lm_jacobian_stats(engine, trial, runtime_profile=runtime_profile)
            u = u_trial
            cur = trial
            z_warm = cur.z_star_bar

            acc_pose_path = _save_pose(which="accept", iter_idx=k, pack=cur)
            cur_tip_fields = _tip_report_fields(cur, p_des)
            _update_run_config(
                run_config_path,
                updates={
                    "last_accept": {
                        "iter": int(k),
                        "x_m": float(cur.x),
                        "I_A": [float(v) for v in np.asarray(jax.device_get(cur.I)).reshape(-1,).tolist()],
                        **cur_tip_fields,
                        "centerline_npy": str(acc_pose_path.relative_to(run_dir)),
                        **_branch_record_fields(trial_branch_record),
                    }
                },
            )

            # update "previous accepted" reference (for dI/dx residuals)
            I_prev = jnp.asarray(cur.I, dtype=jnp.float64)
            x_prev = float(cur.x)
            if trial_branch_record is not None:
                last_branch_eval_id = trial_branch_record.eval_id

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

    try:
        fin_pack = _fk_eval(
            engine,
            I=jnp.asarray(I_fin, dtype=jnp.float64),
            x=float(x_fin),
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
            stats_include_jacobian=False,
            runtime_profile=runtime_profile,
            profile_label="final",
        )
        fin_pose_path = _save_pose(which="accept", iter_idx=int(len(history.get("outer", []))), pack=fin_pack)
        fin_branch_record = _save_branch_state(
            phase="final",
            iter_idx=int(len(history.get("outer", []))),
            pack=fin_pack,
            status="FINAL",
            accept=True,
            parent_eval_id=last_branch_eval_id,
            pose_path=fin_pose_path,
            outer={"source": "final_summary"},
        )
        fin_tip_fields = _tip_report_fields(fin_pack, p_des)
        _update_run_config(
            run_config_path,
            updates={
                "final": {
                    "x_m": float(fin_pack.x),
                    "I_A": [float(v) for v in np.asarray(jax.device_get(fin_pack.I)).reshape(-1,).tolist()],
                    **fin_tip_fields,
                    "centerline_npy": str(fin_pose_path.relative_to(run_dir)),
                    **_branch_record_fields(fin_branch_record),
                }
            },
        )
    except Exception as e:
        print(f"[IK-LM] WARNING: failed to save final snapshot: {repr(e)}")
    t0 = time.perf_counter()
    result_tip_fields = _tip_report_fields(cur, p_des)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "success": bool(cur.err <= eps_p and cur.cls != "REJECT"),
                "success_flex_eps": bool(result_tip_fields["err_flex_m"] <= eps_p and cur.cls != "REJECT"),
                "success_true_eps": bool(result_tip_fields["err_true_m"] <= eps_p and cur.cls != "REJECT"),
                "p_des": [float(v) for v in p_des.tolist()],
                "p_tip": result_tip_fields["p_tip_m"],
                "p_tip_flex": result_tip_fields["p_tip_flex_m"],
                "p_tip_true": result_tip_fields["p_tip_true_m"],
                **result_tip_fields,
                "I": [float(v) for v in I_fin.tolist()],
                "x": float(x_fin),
                "normE": float(cur.normE),
                "cls": str(cur.cls),
                "sigma_p": float(sigma_p),
                "sensitivity_config": sensitivity_config,
                "runtime_environment": runtime_environment,
                "sigmas_from_w": {
                    "sigma_I": (1.0 / w_I) if w_I > 0 else None,
                    "sigma_x": (1.0 / w_x) if w_x > 0 else None,
                    "sigma_dI": (1.0 / w_dI) if w_dI > 0 else None,
                    "sigma_dx": (1.0 / w_dx) if w_dx > 0 else None,
                    "sigma_Ix": (1.0 / w_Ix) if w_Ix > 0 else None,
                },
                "step_limits": {"step_I_max": step_I_max, "step_x_max": step_x_max},
                "p2_outer_solver": {
                    "trust_region": p2_trust_region,
                    "terminal_error_first": p2_terminal_error_first,
                    "reliable_subspace": p2_reliable_subspace,
                    "reliable_task_rcond": p2_reliable_task_rcond,
                    "min_full_task_rank": p2_min_full_task_rank,
                    "max_backtracks": p2_max_backtracks,
                    "backtrack_shrink": p2_backtrack_shrink,
                    "disable_Ix_jacobian_below_dx": p2_disable_Ix_jacobian_below_dx,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _profile_add(runtime_profile, "io", "save_result", time.perf_counter() - t0)
    _write_runtime_profile(run_dir, runtime_profile)
    print("[IK-LM] Saved result.json")


if __name__ == "__main__":
    main()
