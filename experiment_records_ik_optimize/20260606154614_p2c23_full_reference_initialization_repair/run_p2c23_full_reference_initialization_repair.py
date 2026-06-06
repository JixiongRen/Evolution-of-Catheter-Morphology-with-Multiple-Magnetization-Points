"""Audit full-reference initialization for cross-condition shooting sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C13_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

import run_p2c10w_task_reduction_phase_map as p2c10w
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


def _parse_candidates(text: str) -> list[tuple[str, float]]:
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        case, alpha = item.split(":", 1)
        specs.append((case.strip(), float(alpha)))
    return specs


def _base_currents(case_name: str) -> np.ndarray:
    case = p2c10w.p2c10t.p2c10i._case_by_name(case_name)
    return np.asarray(case.currents, dtype=float)


def _magnet_count(case_name: str) -> int:
    if case_name.startswith("n1_"):
        return 1
    if case_name.startswith("n2_"):
        return 2
    if case_name.startswith("n3_"):
        return 3
    return -1


def _seed_vectors(y_ref: np.ndarray, params: Any) -> list[tuple[str, str, np.ndarray]]:
    y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
    zero = p2c10w._initial_vector(y_ref, "zero", params)
    truncated = p2c10w._initial_vector(y_ref, "oracle", params)
    interface_only = zero.copy()
    interface_only[6:] = y_ref[6:]
    return [
        ("truncated_oracle", "baseline_truncated", truncated),
        ("full_reference", "full_reference", y_ref.copy()),
        ("half_full_reference", "scaled_reference", 0.5 * y_ref),
        ("interface_only", "interface_reference", interface_only),
        ("zero", "zero", zero),
    ]


def _audit_load(params_base: Any, I: np.ndarray, y: np.ndarray) -> dict[str, float]:
    params = with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "max_slack_force_over_rigid_force": max((float(row["slack_force_over_rigid_force"]) for row in rows), default=float("nan")),
        "max_slack_torque_over_rigid_torque": max((float(row["slack_torque_over_rigid_torque"]) for row in rows), default=float("nan")),
        "trace_boundary_norm": float(np.linalg.norm(trace["boundary"])),
    }


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes = []
    if not row["success"]:
        modes.append("求解器达到迭代上限")
    if row["position_error_m"] > 0.05 * 0.11:
        modes.append("可达域或末端约束失配")
    if row["stationarity_norm"] > 1.0e-5:
        modes.append("一阶最优性退化")
    if row["jacobian_condition"] > 1.0e10:
        modes.append("数值病态或局部不可辨识")
    if row.get("max_slack_force_over_rigid_force", 0.0) > 2.0:
        modes.append("刚性磁体段载荷传播失配")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    priority = [
        "一阶最优性退化",
        "刚性磁体段载荷传播失配",
        "数值病态或局部不可辨识",
        "可达域或末端约束失配",
        "求解器达到迭代上限",
    ]
    for item in priority:
        if item in modes:
            return item
    return modes[0]


def _solve_case(case_name: str, alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha})
    params, I0, target_p, target_q, y_ref, oracle = p2c10w._build_case(case_name, alpha)
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
    _ = funcs["residual_jit"](jnp.asarray(np.asarray(y_ref, dtype=float)), jnp.asarray(I0)).block_until_ready()

    seed_rows = []
    solves = {}
    for seed_name, seed_class, y0 in _seed_vectors(y_ref, params):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "seed", "case": case_name, "alpha": alpha, "seed": seed_name})
        solve = p2c10w._solve(y0, I0, funcs, args.max_iter)
        load = _audit_load(params, I0, solve["y"])
        pos_err = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
        row = {
            "case": case_name,
            "alpha": alpha,
            "magnet_count": _magnet_count(case_name),
            "seed_name": seed_name,
            "seed_class": seed_class,
            "success": solve["success"],
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "position_error_m": pos_err,
            "boundary_norm": solve["boundary_norm"],
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "slack_norm": float(np.linalg.norm(solve["y"][6:])),
            "residual_norm": solve["residual_norm"],
            **load,
        }
        row["trusted_baseline_gate"] = bool(
            row["success"]
            and row["position_error_m"] <= 0.05 * 0.11
            and row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] <= 1.0e10
            and row["max_slack_force_over_rigid_force"] <= 2.0
        )
        modes = [] if row["trusted_baseline_gate"] else _failure_modes(row)
        row["failure_modes"] = ";".join(modes)
        row["primary_failure_mode"] = "" if row["trusted_baseline_gate"] else _primary_mode(modes)
        seed_rows.append(row)
        solves[seed_name] = solve
        if not row["trusted_baseline_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "seed_baseline_gate_failed", "evidence": row})

    pair_rows = []
    for a, b in combinations(sorted(solves), 2):
        ca = np.asarray(solves[a]["out"]["centerline"], dtype=float)
        cb = np.asarray(solves[b]["out"]["centerline"], dtype=float)
        pair_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "seed_a": a,
                "seed_b": b,
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "tip_distance": float(np.linalg.norm(ca[-1] - cb[-1])),
                "state_distance": float(np.linalg.norm(solves[a]["y"] - solves[b]["y"])),
            }
        )
    best = min(seed_rows, key=lambda r: (not r["trusted_baseline_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"], r["position_error_m"]))
    summary = {
        "case": case_name,
        "alpha": alpha,
        "magnet_count": _magnet_count(case_name),
        "best_seed": best["seed_name"],
        "best_seed_class": best["seed_class"],
        "trusted_baseline_gate": best["trusted_baseline_gate"],
        "best_stationarity_norm": best["stationarity_norm"],
        "best_position_error_m": best["position_error_m"],
        "best_jacobian_condition": best["jacobian_condition"],
        "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
        "best_boundary_norm": best["boundary_norm"],
        "best_success": best["success"],
        "trusted_seed_count": sum(row["trusted_baseline_gate"] for row in seed_rows),
        "max_pair_centerline_rms": max((row["centerline_rms"] for row in pair_rows), default=0.0),
        "max_pair_tip_distance": max((row["tip_distance"] for row in pair_rows), default=0.0),
        "total_steady_s": sum(row["steady_s"] for row in seed_rows),
        "primary_failure_mode": "" if best["trusted_baseline_gate"] else best["primary_failure_mode"],
        "failure_modes": "" if best["trusted_baseline_gate"] else best["failure_modes"],
    }
    return summary, seed_rows, pair_rows


def _plot_seed_metrics(seed_rows: list[dict[str, Any]]) -> None:
    rows = seed_rows
    labels = [f"{r['case']}\n{r['seed_name']}" for r in rows]
    colors = ["green" if r["trusted_baseline_gate"] else "orange" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(max(12, 0.55 * len(rows)), 13), sharex=True)
    specs = [
        ("stationarity_norm", "一阶最优性", 1.0e-5),
        ("position_error_m", "末端位置误差 m", 0.05 * 0.11),
        ("max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
        ("jacobian_condition", "局部条件数", 1.0e10),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [float(r[key]) for r in rows], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_reference_seed_gate_metrics.png", dpi=180)
    plt.close(fig)


def _plot_summary(summaries: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}\nalpha={row['alpha']:.2f}" for row in summaries]
    x = np.arange(len(labels))
    colors = ["green" if row["trusted_baseline_gate"] else "orange" for row in summaries]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    specs = [
        ("best_stationarity_norm", "最佳一阶最优性", 1.0e-5),
        ("best_slack_force_ratio", "最佳界面力松弛/刚段外力", 2.0),
        ("max_pair_centerline_rms", "多初值中心线最大差异 m", 1.0e-3),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [float(row[key]) for row in summaries], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_reference_repair_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_base_i4:1.00,n3_mixed_rand20:0.40",
    )
    parser.add_argument("--max_iter", type=int, default=220)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for case_name, alpha in _parse_candidates(args.candidates):
        summary, seeds, pairs = _solve_case(case_name, alpha, args, heartbeat, failures)
        summaries.append(summary)
        seed_rows.extend(seeds)
        pair_rows.extend(pairs)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "seed_rows.csv", seed_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", pair_rows)
    _plot_seed_metrics(seed_rows)
    _plot_summary(summaries)
    counter = Counter(row["primary_failure_mode"] for row in summaries if row["primary_failure_mode"])
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
            "case_count": len(summaries),
            "trusted_case_count": sum(row["trusted_baseline_gate"] for row in summaries),
            "diagnostic_case_count": sum(not row["trusted_baseline_gate"] for row in summaries),
            "trusted_seed_count": sum(row["trusted_baseline_gate"] for row in seed_rows),
            "failure_mode_count": dict(counter),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
