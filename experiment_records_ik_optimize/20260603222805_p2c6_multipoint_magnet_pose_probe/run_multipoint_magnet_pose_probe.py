"""Run P2c6 multipoint magnet-pose shooting feasibility probes."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C5_DIR = ROOT / "experiment_records_ik_optimize" / "20260603204650_p2c5_jax_clean_shooting_speedup"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_kinematics.ik_observations import quaternion_so3_residual, quat_to_rotmat_wxyz
from inverse_kinematics.ik_terminal_shooting_jax import (
    clean_multisite_residual,
    forward_segment_wrench,
    output_to_numpy,
)

import run_clean_multi_case as p2c4
import run_jax_clean_shooting as p2c5
import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
LAMBDA_MAX = 1.0e10


@dataclass(frozen=True)
class BundleSpec:
    """One multipoint residual bundle."""

    name: str
    suite: str
    use_positions: bool
    use_previous_axis: bool
    use_previous_so3: bool
    max_iter_per_stage: int = 16


@dataclass
class LmResult:
    """Dense LM result for one multipoint probe."""

    y: np.ndarray
    out: dict[str, np.ndarray]
    history: list[dict[str, Any]]
    success: bool
    cost: float
    n_iter: int
    reject_count: int
    lambda_final: float
    steady_solve_s: float
    residual_eval_s: float
    jacobian_eval_s: float
    linear_solve_s: float
    condition_estimate: float
    singular_values: list[float]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _block(x: Any) -> Any:
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x:
            _block(item)
    return x


def _so3_error_deg(current_q: np.ndarray, target_q: np.ndarray) -> float:
    r = quaternion_so3_residual(jnp.asarray(current_q), jnp.asarray(target_q), sigma_angle=1.0)
    return float(np.degrees(np.linalg.norm(np.asarray(jax.device_get(r), dtype=float))))


def _axis_from_q(q: np.ndarray) -> np.ndarray:
    axis_body = np.asarray([0.0, 0.0, 1.0], dtype=float)
    R = np.asarray(jax.device_get(quat_to_rotmat_wxyz(jnp.asarray(q, dtype=jnp.float64))), dtype=float)
    axis = R @ axis_body
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _axis_error_deg(current_q: np.ndarray, target_q: np.ndarray) -> float:
    cur = _axis_from_q(current_q)
    tar = _axis_from_q(target_q)
    cos_angle = float(np.clip(cur @ tar, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _active_magnet_mask(case: Any) -> np.ndarray:
    return np.asarray([np.linalg.norm(np.asarray(m, dtype=float)) > 0.0 for m in case.m_body_list], dtype=bool)


def _previous_active_mask(active: np.ndarray) -> np.ndarray:
    idxs = np.flatnonzero(active)
    mask = np.zeros_like(active, dtype=bool)
    if idxs.size >= 2:
        mask[int(idxs[-2])] = True
    elif idxs.size == 1:
        mask[int(idxs[-1])] = True
    return mask


def _targets_from_oracle(oracle: dict[str, Any], engine: Any) -> dict[str, np.ndarray]:
    terminal_p, terminal_q = p2c2._oracle_terminal_pose(oracle, engine)
    rigid_mid_p, rigid_mid_q = p2c2._oracle_rigid_midpoint_poses(oracle, engine)
    sites = oracle["sites"]
    rigid_dist_p = np.asarray(jax.device_get(sites["rigid_mid_p_dim"]), dtype=float)
    rigid_dist_q = np.asarray(jax.device_get(sites["rigid_mid_Q_wxyz"]), dtype=float)
    return {
        "terminal_p": np.asarray(terminal_p, dtype=float).reshape(3),
        "terminal_q": np.asarray(terminal_q, dtype=float).reshape(4),
        "rigid_mid_p": np.asarray(rigid_mid_p, dtype=float).reshape((-1, 3)),
        "rigid_mid_q": np.asarray(rigid_mid_q, dtype=float).reshape((-1, 4)),
        "rigid_dist_p": np.asarray(rigid_dist_p, dtype=float).reshape((-1, 3)),
        "rigid_dist_q": np.asarray(rigid_dist_q, dtype=float).reshape((-1, 4)),
    }


def _make_functions(params: Any, targets: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
    target_terminal_p = jnp.asarray(targets["terminal_p"], dtype=jnp.float64)
    target_terminal_q = jnp.asarray(targets["terminal_q"], dtype=jnp.float64)
    target_rigid_p = jnp.asarray(targets["rigid_mid_p"], dtype=jnp.float64)
    target_rigid_q = jnp.asarray(targets["rigid_mid_q"], dtype=jnp.float64)
    site_position_mask = jnp.asarray(masks["position"], dtype=jnp.float64)
    site_axis_mask = jnp.asarray(masks["axis"], dtype=jnp.float64)
    site_so3_mask = jnp.asarray(masks["so3"], dtype=jnp.float64)
    axis_body = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64)

    def residual(y: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
        return clean_multisite_residual(
            params,
            y,
            target_terminal_p=target_terminal_p,
            target_terminal_q=target_terminal_q,
            target_rigid_p=target_rigid_p,
            target_rigid_q=target_rigid_q,
            site_position_mask=site_position_mask,
            site_axis_mask=site_axis_mask,
            site_so3_mask=site_so3_mask,
            axis_body=axis_body,
            steps_per_flex=4,
            sigma_terminal_p=1.0e-3,
            sigma_terminal_angle=1.0,
            sigma_rigid_p=1.0e-3,
            sigma_rigid_angle=1.0,
            boundary_weight=0.5,
            terminal_position_scale=scales[0],
            terminal_so3_scale=scales[1],
            rigid_position_scale=scales[2],
            rigid_axis_scale=scales[3],
            rigid_so3_scale=scales[4],
            boundary_scale=scales[5],
        )

    def output(y: jnp.ndarray):
        return forward_segment_wrench(params, y, steps_per_flex=4)

    residual_jit = jax.jit(residual)
    jacobian_jit = jax.jit(jax.jacfwd(residual, argnums=0))
    output_jit = jax.jit(output)
    return residual_jit, jacobian_jit, output_jit


def _stages(bundle: BundleSpec) -> list[np.ndarray]:
    pos_final = 1.0 if bundle.use_positions else 0.0
    axis_final = 1.0 if bundle.use_previous_axis else 0.0
    so3_final = 1.0 if bundle.use_previous_so3 else 0.0
    return [
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float),
        np.asarray([1.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=float),
        np.asarray([1.0, 1.0, 0.3 * pos_final, 0.0, 0.0, 1.0], dtype=float),
        np.asarray([1.0, 1.0, pos_final, 0.3 * axis_final, 0.3 * so3_final, 1.0], dtype=float),
        np.asarray([1.0, 1.0, pos_final, axis_final, so3_final, 1.0], dtype=float),
    ]


def _time_compile(residual_jit: Any, jacobian_jit: Any, output_jit: Any, y0: np.ndarray, final_scales: np.ndarray) -> dict[str, float]:
    y = jnp.asarray(y0, dtype=jnp.float64)
    scales = jnp.asarray(final_scales, dtype=jnp.float64)
    t0 = time.perf_counter()
    _block(residual_jit(y, scales))
    residual_compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _block(jacobian_jit(y, scales))
    jacobian_compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _block(output_jit(y))
    output_compile_s = time.perf_counter() - t0
    return {
        "compile_s": float(residual_compile_s + jacobian_compile_s + output_compile_s),
        "residual_compile_s": float(residual_compile_s),
        "jacobian_compile_s": float(jacobian_compile_s),
        "output_compile_s": float(output_compile_s),
    }


def _metrics(out: dict[str, np.ndarray], targets: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, float]:
    terminal_position_error = float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float) - targets["terminal_p"]))
    terminal_so3_error_deg = _so3_error_deg(np.asarray(out["tip_q"], dtype=float), targets["terminal_q"])
    pos_mask = np.asarray(masks["position"], dtype=bool)
    axis_mask = np.asarray(masks["axis"], dtype=bool)
    so3_mask = np.asarray(masks["so3"], dtype=bool)
    rigid_diffs = np.linalg.norm(np.asarray(out["rigid_mid_p"], dtype=float) - targets["rigid_mid_p"], axis=1)
    rigid_selected = rigid_diffs[pos_mask] if np.any(pos_mask) else np.asarray([], dtype=float)
    axis_errors = np.asarray(
        [_axis_error_deg(out["rigid_mid_q"][i], targets["rigid_mid_q"][i]) for i in range(len(targets["rigid_mid_q"]))],
        dtype=float,
    )
    axis_selected = axis_errors[axis_mask] if np.any(axis_mask) else np.asarray([], dtype=float)
    so3_errors = np.asarray(
        [_so3_error_deg(out["rigid_mid_q"][i], targets["rigid_mid_q"][i]) for i in range(len(targets["rigid_mid_q"]))],
        dtype=float,
    )
    so3_selected = so3_errors[so3_mask] if np.any(so3_mask) else np.asarray([], dtype=float)
    return {
        "terminal_position_error": terminal_position_error,
        "terminal_so3_error_deg": terminal_so3_error_deg,
        "rigid_position_rms_mm": float(np.sqrt(np.mean(rigid_selected**2)) * 1000.0) if rigid_selected.size else 0.0,
        "rigid_position_max_mm": float(np.max(rigid_selected) * 1000.0) if rigid_selected.size else 0.0,
        "axis_rms_deg": float(np.sqrt(np.mean(axis_selected**2))) if axis_selected.size else 0.0,
        "axis_max_deg": float(np.max(axis_selected)) if axis_selected.size else 0.0,
        "rigid_so3_rms_deg": float(np.sqrt(np.mean(so3_selected**2))) if so3_selected.size else 0.0,
        "rigid_so3_max_deg": float(np.max(so3_selected)) if so3_selected.size else 0.0,
        "boundary_residual_norm": float(np.linalg.norm(np.asarray(out["boundary"], dtype=float))),
    }


def _passes(metrics: dict[str, float], bundle: BundleSpec) -> bool:
    if metrics["terminal_position_error"] > 1.0e-3:
        return False
    if metrics["terminal_so3_error_deg"] > 2.0:
        return False
    if metrics["boundary_residual_norm"] > 0.2:
        return False
    if bundle.use_positions and metrics["rigid_position_rms_mm"] > 1.0:
        return False
    if bundle.use_previous_axis and metrics["axis_max_deg"] > 5.0:
        return False
    if bundle.use_previous_so3 and metrics["rigid_so3_max_deg"] > 5.0:
        return False
    return True


def _stage_passes(metrics: dict[str, float], scales: np.ndarray, bundle: BundleSpec) -> bool:
    """Return whether the active residual groups are good enough for a stage."""
    if metrics["terminal_position_error"] > 1.0e-3:
        return False
    if metrics["boundary_residual_norm"] > 0.2:
        return False
    if float(scales[1]) > 0.0 and metrics["terminal_so3_error_deg"] > 2.0:
        return False
    if float(scales[2]) > 0.0 and bundle.use_positions and metrics["rigid_position_rms_mm"] > 1.0:
        return False
    if float(scales[3]) > 0.0 and bundle.use_previous_axis and metrics["axis_max_deg"] > 5.0:
        return False
    if float(scales[4]) > 0.0 and bundle.use_previous_so3 and metrics["rigid_so3_max_deg"] > 5.0:
        return False
    return True


def _condition_estimate(jacobian_jit: Any, y: np.ndarray, scales: np.ndarray) -> tuple[float, list[float]]:
    J = np.asarray(jax.device_get(jacobian_jit(jnp.asarray(y, dtype=jnp.float64), jnp.asarray(scales, dtype=jnp.float64))), dtype=float)
    s = np.linalg.svd(J, compute_uv=False)
    positive = s[s > 1e-12]
    cond = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return cond, [float(v) for v in s]


def _lm_solve(
    *,
    y0: np.ndarray,
    residual_jit: Any,
    jacobian_jit: Any,
    output_jit: Any,
    targets: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    bundle: BundleSpec,
    step_cap: float = 5.0,
) -> LmResult:
    y = np.asarray(y0, dtype=float).reshape(-1)
    lam = 1e-3
    reject_count = 0
    history: list[dict[str, Any]] = []
    residual_eval_s = 0.0
    jacobian_eval_s = 0.0
    linear_solve_s = 0.0
    stage_table = _stages(bundle)
    t_solve = time.perf_counter()

    for stage_index, scales in enumerate(stage_table):
        for iter_index in range(bundle.max_iter_per_stage):
            jy = jnp.asarray(y, dtype=jnp.float64)
            jscales = jnp.asarray(scales, dtype=jnp.float64)
            current_out = output_to_numpy(output_jit(jy))
            current_metrics = _metrics(current_out, targets, masks)
            if _stage_passes(current_metrics, scales, bundle):
                if stage_index != len(stage_table) - 1:
                    break
                steady_solve_s = time.perf_counter() - t_solve
                cond, singular_values = _condition_estimate(jacobian_jit, y, scales)
                r_final = np.asarray(jax.device_get(residual_jit(jy, jscales)), dtype=float)
                return LmResult(
                    y=y,
                    out=current_out,
                    history=history,
                    success=bool(steady_solve_s <= 30.0 and reject_count <= 5 and lam < LAMBDA_MAX),
                    cost=0.5 * float(r_final @ r_final),
                    n_iter=len(history),
                    reject_count=reject_count,
                    lambda_final=float(lam),
                    steady_solve_s=float(steady_solve_s),
                    residual_eval_s=float(residual_eval_s),
                    jacobian_eval_s=float(jacobian_eval_s),
                    linear_solve_s=float(linear_solve_s),
                    condition_estimate=cond,
                    singular_values=singular_values,
                )
            t0 = time.perf_counter()
            r = np.asarray(jax.device_get(residual_jit(jy, jscales)), dtype=float)
            residual_eval_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            J = np.asarray(jax.device_get(jacobian_jit(jy, jscales)), dtype=float)
            jacobian_eval_s += time.perf_counter() - t0
            cost = 0.5 * float(r @ r)
            g = J.T @ r
            H0 = J.T @ J
            D = np.clip(np.diag(H0), 1e-12, None)
            accepted = False
            rho = float("nan")
            step_norm = float("nan")
            trial_metrics: dict[str, float] = {}
            for _attempt in range(8):
                t0 = time.perf_counter()
                H = H0 + lam * np.diag(D)
                try:
                    dy = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    dy = np.linalg.lstsq(H, -g, rcond=None)[0]
                linear_solve_s += time.perf_counter() - t0
                step_norm = float(np.linalg.norm(dy))
                if step_norm > step_cap:
                    dy = dy * (step_cap / step_norm)
                    step_norm = float(np.linalg.norm(dy))
                pred = -float(g @ dy + 0.5 * dy @ (H0 @ dy))
                if not np.isfinite(pred) or pred <= 1.0e-14:
                    lam = min(lam * 10.0, LAMBDA_MAX)
                    continue
                y_trial = y + dy
                t0 = time.perf_counter()
                r_trial = np.asarray(
                    jax.device_get(residual_jit(jnp.asarray(y_trial, dtype=jnp.float64), jscales)),
                    dtype=float,
                )
                residual_eval_s += time.perf_counter() - t0
                cost_trial = 0.5 * float(r_trial @ r_trial)
                actual = cost - cost_trial
                rho = actual / pred
                out_trial = output_to_numpy(output_jit(jnp.asarray(y_trial, dtype=jnp.float64)))
                trial_metrics = _metrics(out_trial, targets, masks)
                finite = np.all(np.isfinite(y_trial)) and np.all(np.isfinite(r_trial))
                boundary_ok = trial_metrics["boundary_residual_norm"] <= 5.0 * max(1.0, current_metrics["boundary_residual_norm"])
                if finite and actual > 0.0 and boundary_ok and trial_metrics["terminal_so3_error_deg"] < 175.0:
                    y = y_trial
                    accepted = True
                    if rho > 0.75:
                        lam = max(lam * 0.3, 1e-12)
                    elif rho > 0.25:
                        lam = max(lam * 0.5, 1e-12)
                    break
                lam = min(lam * 10.0, LAMBDA_MAX)
            if not accepted:
                reject_count += 1
            history.append(
                {
                    "stage_index": int(stage_index),
                    "iter": int(iter_index),
                    "cost": float(cost),
                    "lambda": float(lam),
                    "rho": float(rho),
                    "accepted": bool(accepted),
                    "step_norm": float(step_norm),
                    **trial_metrics,
                }
            )
            if lam >= LAMBDA_MAX:
                break
            if accepted and _stage_passes(trial_metrics, scales, bundle):
                break
        if lam >= LAMBDA_MAX:
            break

    steady_solve_s = time.perf_counter() - t_solve
    final_out = output_to_numpy(output_jit(jnp.asarray(y, dtype=jnp.float64)))
    final_metrics = _metrics(final_out, targets, masks)
    final_scales = stage_table[-1]
    cond, singular_values = _condition_estimate(jacobian_jit, y, final_scales)
    r_final = np.asarray(
        jax.device_get(residual_jit(jnp.asarray(y, dtype=jnp.float64), jnp.asarray(final_scales, dtype=jnp.float64))),
        dtype=float,
    )
    success = _passes(final_metrics, bundle) and steady_solve_s <= 30.0 and reject_count <= 5 and lam < LAMBDA_MAX
    return LmResult(
        y=y,
        out=final_out,
        history=history,
        success=bool(success),
        cost=0.5 * float(r_final @ r_final),
        n_iter=len(history),
        reject_count=int(reject_count),
        lambda_final=float(lam),
        steady_solve_s=float(steady_solve_s),
        residual_eval_s=float(residual_eval_s),
        jacobian_eval_s=float(jacobian_eval_s),
        linear_solve_s=float(linear_solve_s),
        condition_estimate=cond,
        singular_values=singular_values,
    )


def _p0_site_gate(params: Any, y0: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, Any]:
    py = p2c2.p2c1._forward_modes(params, y0, "segment_wrench", steps_per_flex=4)
    jx = output_to_numpy(forward_segment_wrench(params, jnp.asarray(y0, dtype=jnp.float64), steps_per_flex=4))
    parity = {
        "terminal_jax_python_diff_m": float(np.linalg.norm(jx["tip_p"] - py["tip_p"])),
        "rigid_mid_jax_python_max_diff_m": float(np.max(np.linalg.norm(jx["rigid_mid_p"] - py["rigid_mid_p"], axis=1))),
        "boundary_jax_python_diff": float(np.linalg.norm(jx["boundary"] - py["boundary"])),
        "fk_target_terminal_finite": bool(np.all(np.isfinite(targets["terminal_p"]))),
        "fk_target_rigid_mid_finite": bool(np.all(np.isfinite(targets["rigid_mid_p"]))),
        "fk_target_rigid_dist_finite": bool(np.all(np.isfinite(targets["rigid_dist_p"]))),
        "convention": {
            "terminal_end": "last rigid distal end",
            "rigid_mid": "rigid_state_along_dim from proximal at sigma=0.5*L",
            "rigid_dist": "stored FK rigid state interpreted as distal",
        },
    }
    parity["passed"] = (
        parity["terminal_jax_python_diff_m"] < 1e-9
        and parity["rigid_mid_jax_python_max_diff_m"] < 1e-9
        and parity["boundary_jax_python_diff"] < 1e-9
        and parity["fk_target_terminal_finite"]
        and parity["fk_target_rigid_mid_finite"]
        and parity["fk_target_rigid_dist_finite"]
    )
    return parity


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_case(case_dir: Path, oracle: dict[str, Any], outputs: dict[str, dict[str, np.ndarray]], targets: dict[str, np.ndarray]) -> None:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.0, label="FK oracle")
    for name, out in outputs.items():
        cl = np.asarray(out["centerline"], dtype=float)
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], "--", lw=1.8, label=name)
    rp = targets["rigid_mid_p"]
    ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], c="purple", s=35, label="rigid mid targets")
    tp = targets["terminal_p"]
    ax.scatter([tp[0]], [tp[1]], [tp[2]], c="green", s=60, label="terminal target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(case_dir.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "multipoint_site_diagnostics.png", dpi=180)
    plt.close(fig)


def _plot_summary(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['case']}\\n{r['bundle']}\\n{r['init_mode']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.55), 6))
    ax.bar(x, [r["steady_solve_s"] for r in rows], width=0.8, label="steady solve")
    ax.axhline(30.0, color="red", ls="--", lw=1.2, label="30s gate")
    ax.set_ylabel("time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c6_time_summary.png", dpi=180)
    plt.close(fig)


def _row_from_result(
    *,
    case: Any,
    bundle: BundleSpec,
    init_mode: str,
    result: LmResult,
    compile_profile: dict[str, float],
    metrics: dict[str, float],
    masks: dict[str, np.ndarray],
    oracle_wall_s: float,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "bundle": bundle.name,
        "suite": bundle.suite,
        "init_mode": init_mode,
        "success": bool(result.success),
        "steady_solve_s": result.steady_solve_s,
        "compile_s": compile_profile["compile_s"],
        "residual_eval_s": result.residual_eval_s,
        "jacobian_eval_s": result.jacobian_eval_s,
        "linear_solve_s": result.linear_solve_s,
        "n_iter": result.n_iter,
        "reject_count": result.reject_count,
        "lambda_final": result.lambda_final,
        "condition_estimate": result.condition_estimate,
        "cost": result.cost,
        "oracle_wall_s": float(oracle_wall_s),
        "active_position_mask": ",".join(str(int(v)) for v in masks["position"]),
        "axis_mask": ",".join(str(int(v)) for v in masks["axis"]),
        "so3_mask": ",".join(str(int(v)) for v in masks["so3"]),
        **metrics,
        **compile_profile,
    }


def _bundles_for_plan(p1_pass_count: int | None = None, p2_pass_count: int | None = None) -> list[BundleSpec]:
    bundles = [
        BundleSpec("p1_position", "multipoint-position", True, False, False),
    ]
    if p1_pass_count is not None and p1_pass_count >= 6:
        bundles.append(BundleSpec("p2_position_axis", "multipoint-axis", True, True, False))
    if p2_pass_count is not None and p2_pass_count >= 6:
        bundles.append(BundleSpec("p3_previous_full_pose", "optional-full-pose-stress", True, False, True))
    return bundles


def main() -> None:
    t_all = time.perf_counter()
    rows: list[dict[str, Any]] = []
    p0_rows: list[dict[str, Any]] = []
    case_payloads: list[dict[str, Any]] = []
    cases = p2c4._cases()
    p1_bundle = BundleSpec("p1_position", "multipoint-position", True, False, False)
    p2_bundle = BundleSpec("p2_position_axis", "multipoint-axis", True, True, False)
    p3_bundle = BundleSpec("p3_previous_full_pose", "optional-full-pose-stress", True, False, True)

    for case_index, case in enumerate(cases):
        case_dir = OUT_DIR / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
        params = oracle["params"]
        targets = _targets_from_oracle(oracle, engine)
        y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params, "segment_wrench")
        p0 = _p0_site_gate(params, y_oracle, targets)
        p0["case"] = case.name
        p0_rows.append(p0)
        if not p0["passed"]:
            continue

        active = _active_magnet_mask(case)
        previous = _previous_active_mask(active)
        init_modes = {
            "oracle_y0": y_oracle,
            "zero_y0": np.zeros_like(y_oracle),
        }
        outputs_for_plot: dict[str, dict[str, np.ndarray]] = {}

        bundle_sequence: list[BundleSpec] = [p1_bundle]
        for bundle in bundle_sequence:
            masks = {
                "position": active if bundle.use_positions else np.zeros_like(active, dtype=bool),
                "axis": previous if bundle.use_previous_axis else np.zeros_like(active, dtype=bool),
                "so3": previous if bundle.use_previous_so3 else np.zeros_like(active, dtype=bool),
            }
            residual_jit, jacobian_jit, output_jit = _make_functions(params, targets, masks)
            final_scales = _stages(bundle)[-1]
            for init_mode, y0 in init_modes.items():
                compile_profile = _time_compile(residual_jit, jacobian_jit, output_jit, y0, final_scales)
                result = _lm_solve(
                    y0=y0,
                    residual_jit=residual_jit,
                    jacobian_jit=jacobian_jit,
                    output_jit=output_jit,
                    targets=targets,
                    masks=masks,
                    bundle=bundle,
                )
                metrics = _metrics(result.out, targets, masks)
                row = _row_from_result(
                    case=case,
                    bundle=bundle,
                    init_mode=init_mode,
                    result=result,
                    compile_profile=compile_profile,
                    metrics=metrics,
                    masks=masks,
                    oracle_wall_s=float(oracle["wall_s"]),
                )
                rows.append(row)
                sub_dir = case_dir / bundle.name / init_mode
                sub_dir.mkdir(parents=True, exist_ok=True)
                np.save(sub_dir / "state_y.npy", result.y)
                np.save(sub_dir / "centerline.npy", result.out["centerline"])
                np.save(sub_dir / "singular_values.npy", np.asarray(result.singular_values, dtype=float))
                (sub_dir / "history.json").write_text(json.dumps(_jsonable(result.history), ensure_ascii=False, indent=2), encoding="utf-8")
                (sub_dir / "summary.json").write_text(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), encoding="utf-8")
                outputs_for_plot[f"{bundle.name}_{init_mode}"] = result.out

        case_payloads.append(
            {
                "case": case.name,
                "oracle_ok": bool(oracle["ok"]),
                "oracle_normE": float(oracle["stats"].normE),
                "oracle_wall_s": float(oracle["wall_s"]),
                "active_magnet_mask": [bool(v) for v in active],
                "previous_active_mask": [bool(v) for v in previous],
            }
        )
        np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))
        _plot_case(case_dir, oracle, outputs_for_plot, targets)

    p1_pass = sum(1 for r in rows if r["bundle"] == "p1_position" and r["success"])
    if all(r["passed"] for r in p0_rows) and p1_pass >= 6:
        for case in cases:
            case_dir = OUT_DIR / case.name
            engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
            oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
            params = oracle["params"]
            targets = _targets_from_oracle(oracle, engine)
            y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params, "segment_wrench")
            active = _active_magnet_mask(case)
            previous = _previous_active_mask(active)
            masks = {"position": active, "axis": previous, "so3": np.zeros_like(active, dtype=bool)}
            residual_jit, jacobian_jit, output_jit = _make_functions(params, targets, masks)
            final_scales = _stages(p2_bundle)[-1]
            outputs_for_plot = {}
            for init_mode, y0 in {"oracle_y0": y_oracle, "zero_y0": np.zeros_like(y_oracle)}.items():
                compile_profile = _time_compile(residual_jit, jacobian_jit, output_jit, y0, final_scales)
                result = _lm_solve(
                    y0=y0,
                    residual_jit=residual_jit,
                    jacobian_jit=jacobian_jit,
                    output_jit=output_jit,
                    targets=targets,
                    masks=masks,
                    bundle=p2_bundle,
                )
                metrics = _metrics(result.out, targets, masks)
                row = _row_from_result(
                    case=case,
                    bundle=p2_bundle,
                    init_mode=init_mode,
                    result=result,
                    compile_profile=compile_profile,
                    metrics=metrics,
                    masks=masks,
                    oracle_wall_s=float(oracle["wall_s"]),
                )
                rows.append(row)
                sub_dir = case_dir / p2_bundle.name / init_mode
                sub_dir.mkdir(parents=True, exist_ok=True)
                np.save(sub_dir / "state_y.npy", result.y)
                np.save(sub_dir / "centerline.npy", result.out["centerline"])
                np.save(sub_dir / "singular_values.npy", np.asarray(result.singular_values, dtype=float))
                (sub_dir / "history.json").write_text(json.dumps(_jsonable(result.history), ensure_ascii=False, indent=2), encoding="utf-8")
                (sub_dir / "summary.json").write_text(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), encoding="utf-8")
                outputs_for_plot[f"{p2_bundle.name}_{init_mode}"] = result.out
            _plot_case(case_dir / p2_bundle.name, oracle, outputs_for_plot, targets)

    p2_pass = sum(1 for r in rows if r["bundle"] == "p2_position_axis" and r["success"])
    if p2_pass >= 6:
        stress_cases = [case for case in cases if case.name in {"n3_mixed_base_i4", "n3_mixed_rand20"}]
        for case in stress_cases:
            case_dir = OUT_DIR / case.name
            engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
            oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
            params = oracle["params"]
            targets = _targets_from_oracle(oracle, engine)
            y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params, "segment_wrench")
            active = _active_magnet_mask(case)
            previous = _previous_active_mask(active)
            masks = {"position": active, "axis": np.zeros_like(active, dtype=bool), "so3": previous}
            residual_jit, jacobian_jit, output_jit = _make_functions(params, targets, masks)
            final_scales = _stages(p3_bundle)[-1]
            for init_mode, y0 in {"oracle_y0": y_oracle, "zero_y0": np.zeros_like(y_oracle)}.items():
                compile_profile = _time_compile(residual_jit, jacobian_jit, output_jit, y0, final_scales)
                result = _lm_solve(
                    y0=y0,
                    residual_jit=residual_jit,
                    jacobian_jit=jacobian_jit,
                    output_jit=output_jit,
                    targets=targets,
                    masks=masks,
                    bundle=p3_bundle,
                )
                metrics = _metrics(result.out, targets, masks)
                row = _row_from_result(
                    case=case,
                    bundle=p3_bundle,
                    init_mode=init_mode,
                    result=result,
                    compile_profile=compile_profile,
                    metrics=metrics,
                    masks=masks,
                    oracle_wall_s=float(oracle["wall_s"]),
                )
                rows.append(row)
                sub_dir = case_dir / p3_bundle.name / init_mode
                sub_dir.mkdir(parents=True, exist_ok=True)
                np.save(sub_dir / "state_y.npy", result.y)
                np.save(sub_dir / "centerline.npy", result.out["centerline"])
                np.save(sub_dir / "singular_values.npy", np.asarray(result.singular_values, dtype=float))
                (sub_dir / "history.json").write_text(json.dumps(_jsonable(result.history), ensure_ascii=False, indent=2), encoding="utf-8")
                (sub_dir / "summary.json").write_text(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_summary(rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "p0_site_gate.csv", p0_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "supervised_residuals": ["terminal_pose", "active_magnet_mid_position", "previous_magnet_axis_or_so3", "boundary"],
            "excluded_residuals": ["centerline_target", "smoothness", "rigid_internal_sample_proxy"],
        },
        "p0_site_gate": p0_rows,
        "cases": case_payloads,
        "rows": rows,
        "p1_pass_count": p1_pass,
        "p2_pass_count": p2_pass,
        "classification": _classification(rows, p0_rows),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


def _classification(rows: list[dict[str, Any]], p0_rows: list[dict[str, Any]]) -> str:
    if not p0_rows or not all(row.get("passed", False) for row in p0_rows):
        return "failed-branch: site convention gate failed"
    p1 = [row for row in rows if row.get("bundle") == "p1_position"]
    p1_pass = sum(1 for row in p1 if row.get("success"))
    if p1_pass < 6:
        return "failed-branch: multipoint-position did not reach support threshold"
    p2 = [row for row in rows if row.get("bundle") == "p2_position_axis"]
    p2_pass = sum(1 for row in p2 if row.get("success"))
    zero_rows = [row for row in rows if row.get("init_mode") == "zero_y0" and row.get("bundle") in {"p1_position", "p2_position_axis"}]
    zero_all_pass = bool(zero_rows) and all(row.get("success") for row in zero_rows)
    if p2 and p2_pass >= 6 and zero_all_pass:
        return "production-candidate: position+axis multipoint shooting"
    if p2 and p2_pass >= 6:
        return "diagnostic-only: position+axis feasible with zero-y0 basin failures"
    return "diagnostic-only: position multipoint shooting"


if __name__ == "__main__":
    main()
