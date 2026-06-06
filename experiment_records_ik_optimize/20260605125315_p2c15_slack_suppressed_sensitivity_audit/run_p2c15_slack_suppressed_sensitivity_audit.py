"""Audit re-optimized sensitivity for a slack-suppressed shooting candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C10X_DIR, P2C13_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c10w_task_reduction_phase_map as p2c10w
import run_p2c10x_position_alpha_linearity_audit as p2c10x
import run_p2c13_rigid_load_propagation_audit as p2c13

configure_matplotlib_cjk()
jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_seed_y(alpha: float) -> np.ndarray:
    path = P2C10X_DIR / f"n3_mixed_rand20_alpha_{alpha:.2f}_position_linearity.npz"
    return np.asarray(np.load(path)["best_y"], dtype=float)


def _make_functions(params_base: Any, target_p: np.ndarray, position_scale: float, boundary_scale: float, steps_per_flex: int):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c10w.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray, slack_scale: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return jnp.concatenate(
            [
                float(position_scale) * (out.tip_p - target_p_j),
                float(boundary_scale) * out.boundary,
                slack_scale * y[6:],
            ],
            axis=0,
        )

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _solve(
    *,
    stage: str,
    y0: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
    target_p: np.ndarray,
    slack_scale: float,
    max_iter: int,
) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j, s_j).block_until_ready()
    t0 = time.perf_counter()
    opt = scipy.optimize.least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j, s_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy), I_j, s_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=max_iter,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j, s_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j, s_j)), dtype=float)
    out = p2c10w.output_to_numpy(funcs["output"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    phi = np.asarray(out["tip_p"], dtype=float).reshape(3)
    return {
        "stage": stage,
        "y": y,
        "out": out,
        "phi": phi,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "message": str(opt.message),
        "steady_s": float(steady_s),
        "position_error_m": float(np.linalg.norm(phi - target_p)),
        "boundary_norm": float(np.linalg.norm(np.asarray(out["boundary"], dtype=float))),
        "slack_norm": float(np.linalg.norm(y[6:])),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "residual_norm": float(np.linalg.norm(r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
    }


def _audit_load(params_base: Any, I: np.ndarray, y: np.ndarray) -> dict[str, float]:
    params = with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "max_slack_force_over_rigid_force": max((float(row["slack_force_over_rigid_force"]) for row in rows), default=float("nan")),
        "max_slack_torque_over_rigid_torque": max((float(row["slack_torque_over_rigid_torque"]) for row in rows), default=float("nan")),
        "min_slack_force_cosine_rigid_force": min((float(row["slack_force_cosine_rigid_force"]) for row in rows), default=float("nan")),
        "min_slack_torque_cosine_rigid_torque": min((float(row["slack_torque_cosine_rigid_torque"]) for row in rows), default=float("nan")),
        "trace_boundary_norm": float(np.linalg.norm(trace["boundary"])),
    }


def _branch_rms(centerlines: dict[str, np.ndarray]) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for a, b in combinations(sorted(centerlines), 2):
        ca = centerlines[a]
        cb = centerlines[b]
        rows.append(
            {
                "branch_a": a,
                "branch_b": b,
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "tip_distance": float(np.linalg.norm(ca[-1] - cb[-1])),
            }
        )
    return max((row["centerline_rms"] for row in rows), default=0.0), rows


def _stage_row(alpha: float, eps: float | None, solve: dict[str, Any], load: dict[str, float]) -> dict[str, Any]:
    return {
        "alpha": alpha,
        "eps_alpha": eps,
        "stage": solve["stage"],
        "success": solve["success"],
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "position_error_m": solve["position_error_m"],
        "boundary_norm": solve["boundary_norm"],
        "slack_norm": solve["slack_norm"],
        "stationarity_norm": solve["stationarity_norm"],
        "residual_norm": solve["residual_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        **load,
    }


def _plot_summary(rows: list[dict[str, Any]]) -> None:
    labels = [f"eps={row['eps_alpha']:.3f}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    specs = [
        ("max_stage_stationarity", "最大一阶平稳性残差", 1.0e-5),
        ("max_linearization_error", "最大局部线性化误差", 5.0e-2),
        ("branch_rms_max", "中心线分支均方根差异 m", 1.0e-3),
        ("max_slack_force_over_rigid_force", "最大界面力松弛/刚段外力", 2.0),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [row[key] for row in rows])
        ax.axhline(gate, color="r", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_gate_summary.png", dpi=180)
    plt.close(fig)


def _plot_centerlines(eps: float, centerlines: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"base": "black", "plus": "red", "minus": "blue"}
    for key, pts in centerlines.items():
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(key, None), label=key)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=colors.get(key, None), s=35)
    ax.scatter(centerlines["base"][0, 0], centerlines["base"][0, 1], centerlines["base"][0, 2], color="green", s=45, label="base point")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"扰动重求解中心线对比 eps={eps:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"centerline_reoptimized_eps_{eps:.3f}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.40)
    parser.add_argument("--eps_values", default="0.005,0.01,0.02")
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=240)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()

    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": args.alpha})
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", args.alpha)
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    I_base = p2c10x._base_currents("n3_mixed_rand20")
    funcs = _make_functions(params, target_p, args.position_scale, args.boundary_scale, args.steps_per_flex)
    seed_y = _load_seed_y(args.alpha)

    base = _solve(
        stage="base",
        y0=seed_y,
        I=I0,
        funcs=funcs,
        target_p=target_p,
        slack_scale=args.slack_scale,
        max_iter=args.max_iter,
    )
    base_load = _audit_load(params, I0, base["y"])
    stage_rows = [_stage_row(args.alpha, None, base, base_load)]
    summaries = []
    branch_rows_all = []

    for eps in _parse_floats(args.eps_values):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "eps_start", "alpha": args.alpha, "eps_alpha": eps})
        solves = {"base": base}
        load_by_stage = {"base": base_load}
        for stage, sign in [("minus", -1.0), ("plus", 1.0)]:
            I = I0 + sign * eps * I_base
            solves[stage] = _solve(
                stage=stage,
                y0=base["y"],
                I=I,
                funcs=funcs,
                target_p=target_p,
                slack_scale=args.slack_scale,
                max_iter=args.max_iter,
            )
            load_by_stage[stage] = _audit_load(params, I, solves[stage]["y"])
            stage_rows.append(_stage_row(args.alpha, eps, solves[stage], load_by_stage[stage]))

        phi0 = solves["base"]["phi"]
        phi_plus = solves["plus"]["phi"]
        phi_minus = solves["minus"]["phi"]
        sensitivity = (phi_plus - phi_minus) / (2.0 * eps)
        scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
        lin_plus = float(np.linalg.norm(phi_plus - (phi0 + sensitivity * eps)) / scale)
        lin_minus = float(np.linalg.norm(phi_minus - (phi0 - sensitivity * eps)) / scale)
        lin_sym = float(np.linalg.norm((phi_plus + phi_minus) * 0.5 - phi0) / scale)
        centerlines = {stage: np.asarray(solve["out"]["centerline"], dtype=float) for stage, solve in solves.items()}
        branch_rms_max, branch_rows = _branch_rms(centerlines)
        for row in branch_rows:
            row.update({"alpha": args.alpha, "eps_alpha": eps})
        branch_rows_all.extend(branch_rows)
        _plot_centerlines(eps, centerlines)

        max_stage_stationarity = max(solve["stationarity_norm"] for solve in solves.values())
        max_stage_condition = max(solve["jacobian_condition"] for solve in solves.values())
        max_slack_force_ratio = max(load["max_slack_force_over_rigid_force"] for load in load_by_stage.values())
        max_linearization_error = max(lin_plus, lin_minus, lin_sym)
        publishable_gate = bool(
            all(solve["success"] for solve in solves.values())
            and max(solve["position_error_m"] for solve in solves.values()) <= 0.05 * 0.11
            and max_stage_stationarity <= 1.0e-5
            and max_stage_condition <= 1.0e10
            and max_linearization_error <= 0.05
            and branch_rms_max <= 1.0e-3
            and max_slack_force_ratio <= 2.0
        )
        summary = {
            "case": "n3_mixed_rand20",
            "alpha": args.alpha,
            "eps_alpha": eps,
            "publishable_gate": publishable_gate,
            "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sensitivity)),
            "sensitivity_vector": sensitivity.tolist(),
            "linearization_error_plus": lin_plus,
            "linearization_error_minus": lin_minus,
            "linearization_error_symmetry": lin_sym,
            "max_linearization_error": max_linearization_error,
            "branch_rms_max": branch_rms_max,
            "max_stage_stationarity": max_stage_stationarity,
            "max_stage_condition": max_stage_condition,
            "max_slack_force_over_rigid_force": max_slack_force_ratio,
            "max_position_error_m": max(solve["position_error_m"] for solve in solves.values()),
            "total_steady_s": sum(solve["steady_s"] for solve in solves.values()),
        }
        if not publishable_gate:
            reasons = []
            if max_stage_stationarity > 1.0e-5:
                reasons.append("一阶平稳性残差超过门限")
            if max_stage_condition > 1.0e10:
                reasons.append("局部条件数超过门限")
            if max_linearization_error > 0.05:
                reasons.append("局部线性化误差超过门限")
            if branch_rms_max > 1.0e-3:
                reasons.append("扰动重求解中心线分支漂移超过门限")
            if max_slack_force_ratio > 2.0:
                reasons.append("界面力松弛与刚段外力比例超过门限")
            _append_jsonl(failures, {"ts": time.time(), "failure": "sensitivity_gate_failed", "reasons": reasons, "evidence": summary})
        np.savez_compressed(
            OUT_DIR / f"alpha_{args.alpha:.2f}_eps_{eps:.3f}_sensitivity_audit.npz",
            phi0=phi0,
            phi_plus=phi_plus,
            phi_minus=phi_minus,
            sensitivity=sensitivity,
            base_centerline=centerlines["base"],
            plus_centerline=centerlines["plus"],
            minus_centerline=centerlines["minus"],
            base_y=solves["base"]["y"],
            plus_y=solves["plus"]["y"],
            minus_y=solves["minus"]["y"],
        )
        summaries.append(summary)

    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "solve_rows.csv", stage_rows)
    _write_csv(OUT_DIR / "branch_rows.csv", branch_rows_all)
    _plot_summary(summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "overall": {
            "eps_count": len(summaries),
            "publishable_count": sum(row["publishable_gate"] for row in summaries),
            "linearity_gate_count": sum(row["max_linearization_error"] <= 0.05 for row in summaries),
            "branch_gate_count": sum(row["branch_rms_max"] <= 1.0e-3 for row in summaries),
            "load_gate_count": sum(row["max_slack_force_over_rigid_force"] <= 2.0 for row in summaries),
            "stationarity_gate_count": sum(row["max_stage_stationarity"] <= 1.0e-5 for row in summaries),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
