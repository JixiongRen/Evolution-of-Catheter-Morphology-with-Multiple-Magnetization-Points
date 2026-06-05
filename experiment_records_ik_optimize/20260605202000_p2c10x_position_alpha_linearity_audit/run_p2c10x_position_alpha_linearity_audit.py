"""Audit low-dimensional position-task linearity along current loading."""

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
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10w_task_reduction_phase_map as p2c10w

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


def _alpha_values(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _base_currents(case_name: str) -> np.ndarray:
    case = p2c10w.p2c10t.p2c10i._case_by_name(case_name)
    return np.asarray(case.currents, dtype=float)


def _solve_at_current(
    *,
    y0: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
    max_iter: int,
) -> dict[str, Any]:
    return p2c10w._solve(y0, I, funcs, max_iter)


def _position_task(out: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(out["tip_p"], dtype=float).reshape(3)


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ok = [row for row in rows if row["task_gate"]]
    pool = task_ok if task_ok else rows
    return min(pool, key=lambda row: (float(row["stationarity_norm"]), float(row["boundary_norm"])))


def _run_alpha(
    *,
    case_name: str,
    alpha: float,
    eps_alpha: float,
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha})
    params, I0, target_p, target_q, y_ref, oracle = p2c10w._build_case(case_name, alpha)
    I_base = _base_currents(case_name)
    funcs = p2c10w._make_functions(
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
    _ = funcs["residual_jit"](jnp.asarray(p2c10w._initial_vector(y_ref, "oracle", params)), jnp.asarray(I0)).block_until_ready()

    baseline_rows: list[dict[str, Any]] = []
    baseline_solves: dict[str, dict[str, Any]] = {}
    for init_name in ["oracle", "half_oracle", "zero"]:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "baseline", "case": case_name, "alpha": alpha, "init": init_name})
        y0 = p2c10w._initial_vector(y_ref, init_name, params)
        solve = _solve_at_current(y0=y0, I=I0, funcs=funcs, max_iter=args.max_iter)
        baseline_solves[init_name] = solve
        pos_err = float(np.linalg.norm(_position_task(solve["out"]) - target_p))
        task_gate = bool(pos_err <= 0.05 * 0.11)
        row = {
            "case": case_name,
            "alpha": alpha,
            "init_name": init_name,
            "success": solve["success"],
            "task_gate": task_gate,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "position_error": pos_err,
            "boundary_norm": solve["boundary_norm"],
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "slack_norm": float(np.linalg.norm(solve["y"][6:])),
        }
        baseline_rows.append(row)
    best = _select_best(baseline_rows)
    best_solve = baseline_solves[best["init_name"]]
    phi0 = _position_task(best_solve["out"])

    perturb_rows: list[dict[str, Any]] = []
    branch_centers: dict[str, np.ndarray] = {"base": np.asarray(best_solve["out"]["centerline"], dtype=float)}
    phi_by_sign: dict[str, np.ndarray] = {}
    solve_by_sign: dict[str, dict[str, Any]] = {}
    for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
        alpha_p = alpha + sign * eps_alpha
        I_p = I0 + sign * eps_alpha * I_base
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": sign_name, "case": case_name, "alpha": alpha, "alpha_perturbed": alpha_p})
        solve = _solve_at_current(y0=best_solve["y"], I=I_p, funcs=funcs, max_iter=args.max_iter)
        solve_by_sign[sign_name] = solve
        phi = _position_task(solve["out"])
        phi_by_sign[sign_name] = phi
        branch_centers[sign_name] = np.asarray(solve["out"]["centerline"], dtype=float)
        perturb_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "perturbation": sign_name,
                "alpha_perturbed": alpha_p,
                "success": solve["success"],
                "nfev": solve["nfev"],
                "steady_s": solve["steady_s"],
                "delta_position_norm": float(np.linalg.norm(phi - phi0)),
                "position_error_to_target": float(np.linalg.norm(phi - target_p)),
                "boundary_norm": solve["boundary_norm"],
                "stationarity_norm": solve["stationarity_norm"],
                "jacobian_condition": solve["jacobian_condition"],
                "state_distance": float(np.linalg.norm(solve["y"] - best_solve["y"])),
            }
        )

    sens = (phi_by_sign["plus"] - phi_by_sign["minus"]) / (2.0 * eps_alpha)
    pred_plus = phi0 + sens * eps_alpha
    pred_minus = phi0 - sens * eps_alpha
    scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
    lin_err_plus = float(np.linalg.norm(phi_by_sign["plus"] - pred_plus) / scale)
    lin_err_minus = float(np.linalg.norm(phi_by_sign["minus"] - pred_minus) / scale)
    lin_err_sym = float(np.linalg.norm((phi_by_sign["plus"] + phi_by_sign["minus"]) * 0.5 - phi0) / scale)
    pair_rows = []
    for a, b in combinations(sorted(branch_centers), 2):
        ca = branch_centers[a]
        cb = branch_centers[b]
        pair_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "branch_a": a,
                "branch_b": b,
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "tip_distance": float(np.linalg.norm(ca[-1] - cb[-1])),
            }
        )
    branch_rms_max = max(row["centerline_rms"] for row in pair_rows)
    gate = bool(
        best["success"]
        and best["task_gate"]
        and best["boundary_norm"] <= 0.3
        and best["stationarity_norm"] <= 1.0e-5
        and best["jacobian_condition"] <= 1.0e10
        and all(row["success"] for row in perturb_rows)
        and max(row["stationarity_norm"] for row in perturb_rows) <= 1.0e-5
        and max(lin_err_plus, lin_err_minus, lin_err_sym) <= 0.05
        and branch_rms_max <= 1.0e-3
    )
    summary = {
        "case": case_name,
        "alpha": alpha,
        "eps_alpha": eps_alpha,
        "best_init": best["init_name"],
        "publishable_gate": gate,
        "baseline_success": best["success"],
        "baseline_task_gate": best["task_gate"],
        "baseline_position_error": best["position_error"],
        "baseline_boundary_norm": best["boundary_norm"],
        "baseline_stationarity_norm": best["stationarity_norm"],
        "baseline_jacobian_condition": best["jacobian_condition"],
        "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sens)),
        "sensitivity_vector": sens.tolist(),
        "linearization_error_plus": lin_err_plus,
        "linearization_error_minus": lin_err_minus,
        "linearization_error_symmetry": lin_err_sym,
        "max_perturb_stationarity": max(row["stationarity_norm"] for row in perturb_rows),
        "max_perturb_boundary": max(row["boundary_norm"] for row in perturb_rows),
        "branch_rms_max": branch_rms_max,
        "branch_tip_distance_max": max(row["tip_distance"] for row in pair_rows),
    }
    if not gate:
        reasons = []
        if not best["success"]:
            reasons.append("基线求解器未成功")
        if not best["task_gate"]:
            reasons.append("基线任务误差未通过")
        if best["stationarity_norm"] > 1.0e-5:
            reasons.append("基线一阶平稳性不足")
        if best["jacobian_condition"] > 1.0e10:
            reasons.append("局部条件数过高")
        if max(row["stationarity_norm"] for row in perturb_rows) > 1.0e-5:
            reasons.append("扰动重求解一阶平稳性不足")
        if max(lin_err_plus, lin_err_minus, lin_err_sym) > 0.05:
            reasons.append("局部线性化误差过大")
        if branch_rms_max > 1.0e-3:
            reasons.append("扰动分支漂移过大")
        _append_jsonl(
            failures,
            {
                "ts": time.time(),
                "failure": "position_alpha_sensitivity_not_publishable",
                "case": case_name,
                "alpha": alpha,
                "reasons": reasons,
                "evidence": summary,
            },
        )
    np.savez_compressed(
        OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_position_linearity.npz",
        target_p=target_p,
        phi0=phi0,
        phi_plus=phi_by_sign["plus"],
        phi_minus=phi_by_sign["minus"],
        sensitivity=sens,
        base_centerline=branch_centers["base"],
        plus_centerline=branch_centers["plus"],
        minus_centerline=branch_centers["minus"],
        best_y=best_solve["y"],
        plus_y=solve_by_sign["plus"]["y"],
        minus_y=solve_by_sign["minus"]["y"],
    )
    return summary, baseline_rows, perturb_rows, pair_rows


def _plot_summary(rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows))
    labels = [f"{row['alpha']:.2f}" for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ax, key, ylabel, log in [
        (axes[0], "baseline_stationarity_norm", "baseline stationarity", True),
        (axes[1], "baseline_jacobian_condition", "condition", True),
        (axes[2], "sensitivity_norm_m_per_alpha", "sensitivity norm", False),
        (axes[3], "branch_rms_max", "branch RMS", True),
    ]:
        ax.bar(x, [row[key] for row in rows])
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("alpha")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_linearity_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([row["alpha"] for row in rows], [row["linearization_error_plus"] for row in rows], "o-", label="plus")
    ax.plot([row["alpha"] for row in rows], [row["linearization_error_minus"] for row in rows], "s-", label="minus")
    ax.plot([row["alpha"] for row in rows], [row["linearization_error_symmetry"] for row in rows], "^-", label="symmetry")
    ax.axhline(0.05, color="r", ls="--", lw=1.0, label="gate")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("relative linearization error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "linearization_error_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--eps_alpha", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=100)
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
    baseline_rows = []
    perturb_rows = []
    pair_rows = []
    for alpha in _alpha_values(args.alphas):
        summary, base_rows, pert_rows, pairs = _run_alpha(
            case_name=args.case,
            alpha=alpha,
            eps_alpha=args.eps_alpha,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        summaries.append(summary)
        baseline_rows.extend(base_rows)
        perturb_rows.extend(pert_rows)
        pair_rows.extend(pairs)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "baseline_rows.csv", baseline_rows)
    _write_csv(OUT_DIR / "perturb_rows.csv", perturb_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", pair_rows)
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
            "alpha_count": len(summaries),
            "publishable_count": sum(row["publishable_gate"] for row in summaries),
            "task_gate_count": sum(row["baseline_task_gate"] for row in summaries),
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
