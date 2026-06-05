"""Refine candidate position-task branches and audit step-size consistency."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, FK_OPT_DIR, P2C10X_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10x_position_alpha_linearity_audit as p2c10x

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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


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


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_seed_y(alpha: float) -> np.ndarray:
    p = P2C10X_DIR / f"n3_mixed_rand20_alpha_{alpha:.2f}_position_linearity.npz"
    z = np.load(p)
    return np.asarray(z["best_y"], dtype=float)


def _run_combo(
    *,
    alpha: float,
    eps_alpha: float,
    max_iter: int,
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_name = args.case
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": alpha, "eps_alpha": eps_alpha, "max_iter": max_iter})
    params, I0, target_p, target_q, _y_ref, _oracle = p2c10x.p2c10w._build_case(case_name, alpha)
    I_base = p2c10x._base_currents(case_name)
    funcs = p2c10x.p2c10w._make_functions(
        params_base=params,
        target_p=target_p,
        target_q=target_q,
        task_kind="position",
        position_scale=args.position_scale,
        axis_scale=args.axis_scale,
        so3_scale=args.so3_scale,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
        steps_per_flex=args.steps_per_flex,
    )
    seed_y = _load_seed_y(alpha)
    _ = funcs["residual_jit"](jnp.asarray(seed_y), jnp.asarray(I0)).block_until_ready()
    baseline = p2c10x._solve_at_current(y0=seed_y, I=I0, funcs=funcs, max_iter=max_iter)
    phi0 = p2c10x._position_task(baseline["out"])
    pos_err = float(np.linalg.norm(phi0 - target_p))
    task_gate = bool(pos_err <= 0.05 * 0.11)
    branch_centers = {"base": np.asarray(baseline["out"]["centerline"], dtype=float)}
    phi_by_sign = {}
    solve_rows = [
        {
            "case": case_name,
            "alpha": alpha,
            "eps_alpha": eps_alpha,
            "max_iter": max_iter,
            "stage": "base",
            "success": baseline["success"],
            "nfev": baseline["nfev"],
            "steady_s": baseline["steady_s"],
            "position_error": pos_err,
            "boundary_norm": baseline["boundary_norm"],
            "stationarity_norm": baseline["stationarity_norm"],
            "jacobian_condition": baseline["jacobian_condition"],
            "state_distance_from_seed": float(np.linalg.norm(baseline["y"] - seed_y)),
        }
    ]
    for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
        I_p = I0 + sign * eps_alpha * I_base
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": sign_name, "alpha": alpha, "eps_alpha": eps_alpha, "max_iter": max_iter})
        solve = p2c10x._solve_at_current(y0=baseline["y"], I=I_p, funcs=funcs, max_iter=max_iter)
        phi = p2c10x._position_task(solve["out"])
        phi_by_sign[sign_name] = phi
        branch_centers[sign_name] = np.asarray(solve["out"]["centerline"], dtype=float)
        solve_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "eps_alpha": eps_alpha,
                "max_iter": max_iter,
                "stage": sign_name,
                "success": solve["success"],
                "nfev": solve["nfev"],
                "steady_s": solve["steady_s"],
                "position_error": float(np.linalg.norm(phi - target_p)),
                "boundary_norm": solve["boundary_norm"],
                "stationarity_norm": solve["stationarity_norm"],
                "jacobian_condition": solve["jacobian_condition"],
                "state_distance_from_seed": float(np.linalg.norm(solve["y"] - seed_y)),
            }
        )
    sens = (phi_by_sign["plus"] - phi_by_sign["minus"]) / (2.0 * eps_alpha)
    scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
    lin_plus = float(np.linalg.norm(phi_by_sign["plus"] - (phi0 + sens * eps_alpha)) / scale)
    lin_minus = float(np.linalg.norm(phi_by_sign["minus"] - (phi0 - sens * eps_alpha)) / scale)
    lin_sym = float(np.linalg.norm((phi_by_sign["plus"] + phi_by_sign["minus"]) * 0.5 - phi0) / scale)
    branch_rms = []
    keys = sorted(branch_centers)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ca = branch_centers[a]
            cb = branch_centers[b]
            branch_rms.append(float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))))
    summary = {
        "case": case_name,
        "alpha": alpha,
        "eps_alpha": eps_alpha,
        "max_iter": max_iter,
        "publishable_gate": bool(
            baseline["success"]
            and task_gate
            and baseline["boundary_norm"] <= 0.3
            and baseline["stationarity_norm"] <= 1.0e-5
            and baseline["jacobian_condition"] <= 1.0e10
            and max(row["stationarity_norm"] for row in solve_rows) <= 1.0e-5
            and max(lin_plus, lin_minus, lin_sym) <= 0.05
            and max(branch_rms) <= 1.0e-3
        ),
        "baseline_success": baseline["success"],
        "baseline_task_gate": task_gate,
        "baseline_position_error": pos_err,
        "baseline_boundary_norm": baseline["boundary_norm"],
        "baseline_stationarity_norm": baseline["stationarity_norm"],
        "baseline_jacobian_condition": baseline["jacobian_condition"],
        "max_stage_stationarity": max(row["stationarity_norm"] for row in solve_rows),
        "max_stage_boundary": max(row["boundary_norm"] for row in solve_rows),
        "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sens)),
        "sensitivity_vector": sens.tolist(),
        "linearization_error_plus": lin_plus,
        "linearization_error_minus": lin_minus,
        "linearization_error_symmetry": lin_sym,
        "branch_rms_max": max(branch_rms),
    }
    if not summary["publishable_gate"]:
        reasons = []
        if not baseline["success"]:
            reasons.append("基线求解器未成功")
        if baseline["stationarity_norm"] > 1.0e-5:
            reasons.append("基线一阶平稳性不足")
        if max(row["stationarity_norm"] for row in solve_rows) > 1.0e-5:
            reasons.append("扰动重求解一阶平稳性不足")
        if max(branch_rms) > 1.0e-3:
            reasons.append("分支漂移过大")
        if max(lin_plus, lin_minus, lin_sym) > 0.05:
            reasons.append("局部线性化误差过大")
        _append_jsonl(failures, {"ts": time.time(), "failure": "refinement_gate_failed", "reasons": reasons, "evidence": summary})
    np.savez_compressed(
        OUT_DIR / f"alpha_{alpha:.2f}_eps_{eps_alpha:.3f}_iter_{max_iter}.npz",
        phi0=phi0,
        phi_plus=phi_by_sign["plus"],
        phi_minus=phi_by_sign["minus"],
        sensitivity=sens,
        base_centerline=branch_centers["base"],
        plus_centerline=branch_centers["plus"],
        minus_centerline=branch_centers["minus"],
        baseline_y=baseline["y"],
    )
    return summary, solve_rows


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"a={r['alpha']:.2f}\neps={r['eps_alpha']:.3f}\nit={r['max_iter']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(max(12, 0.9 * len(rows)), 11), sharex=True)
    for ax, key, ylabel, log in [
        (axes[0], "baseline_stationarity_norm", "baseline stationarity", True),
        (axes[1], "max_stage_stationarity", "max stage stationarity", True),
        (axes[2], "linearization_error_symmetry", "symmetry linearity error", True),
        (axes[3], "branch_rms_max", "branch RMS", True),
    ]:
        ax.bar(x, [r[key] for r in rows])
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(ylabel)
    axes[0].axhline(1.0e-5, color="r", ls="--", lw=1.0)
    axes[1].axhline(1.0e-5, color="r", ls="--", lw=1.0)
    axes[2].axhline(0.05, color="r", ls="--", lw=1.0)
    axes[3].axhline(1.0e-3, color="r", ls="--", lw=1.0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "refinement_step_audit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.40,0.45")
    parser.add_argument("--eps_values", default="0.005,0.01,0.02")
    parser.add_argument("--max_iters", default="100,250,500")
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=120.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=0.05)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries = []
    solve_rows = []
    for alpha in _parse_floats(args.alphas):
        for eps in _parse_floats(args.eps_values):
            for max_iter in _parse_ints(args.max_iters):
                summary, rows = _run_combo(
                    alpha=alpha,
                    eps_alpha=eps,
                    max_iter=max_iter,
                    args=args,
                    heartbeat=heartbeat,
                    failures=failures,
                )
                summaries.append(summary)
                solve_rows.extend(rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "solve_rows.csv", solve_rows)
    _plot(summaries)
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
            "combo_count": len(summaries),
            "publishable_count": sum(row["publishable_gate"] for row in summaries),
            "stationarity_gate_count": sum(row["max_stage_stationarity"] <= 1.0e-5 for row in summaries),
            "branch_gate_count": sum(row["branch_rms_max"] <= 1.0e-3 for row in summaries),
            "linearity_gate_count": sum(
                max(row["linearization_error_plus"], row["linearization_error_minus"], row["linearization_error_symmetry"]) <= 0.05
                for row in summaries
            ),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
