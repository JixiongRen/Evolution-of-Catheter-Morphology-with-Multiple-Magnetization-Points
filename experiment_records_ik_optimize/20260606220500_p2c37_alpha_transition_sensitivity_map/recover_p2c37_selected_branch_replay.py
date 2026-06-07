"""Recover P2c37 by replaying selected saved baseline branches."""

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
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mcrs_mpl_cache")

ROOT = Path(__file__).resolve().parents[2]
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C25_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170041_p2c25_scaled_stationarity_and_interface_audit"
P2C29_DIR = ROOT / "experiment_records_ik_optimize" / "20260606175600_p2c29_rand20_load_continuation_refine"
for path in (ROOT, P2C10W_DIR, P2C24_DIR, P2C25_DIR, P2C29_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c10w_task_reduction_phase_map as p2c10w
import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c25_scaled_stationarity_and_interface_audit as p2c25
import run_p2c29_rand20_load_continuation_refine as p2c29


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
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _make_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        strategies=args.strategies,
        max_iter=args.max_iter,
        restarts=args.restarts,
        tol=args.tol,
        steps_per_flex=args.steps_per_flex,
        position_scale=args.position_scale,
        axis_scale=args.axis_scale,
        so3_scale=args.so3_scale,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
    )


def _selected_specs() -> list[tuple[float, str]]:
    return [
        (0.405, "trusted_alpha_0.425_seed"),
        (0.410, "reverse_alpha_0.425_seed"),
        (0.415, "trusted_alpha_0.425_seed"),
        (0.420, "trusted_alpha_0.400_seed"),
    ]


def _baseline_metrics(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray, target_p: np.ndarray, params: Any, args: argparse.Namespace) -> dict[str, Any]:
    metrics = p2c25._physical_metrics(funcs, y, I, args.boundary_scale, args.slack_scale)
    load = p2c24._audit_load(params, I, y)
    out = metrics["out"]
    row = {
        "position_error_m": float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float) - target_p)),
        "residual_norm": metrics["residual_norm"],
        "stationarity_norm": metrics["stationarity_norm"],
        "boundary_norm": metrics["boundary_norm"],
        "jacobian_condition": metrics["jacobian_condition"],
        "jacobian_rank": metrics["jacobian_rank"],
        **load,
    }
    row["static_gate"] = bool(
        row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
    )
    row["tip_p"] = np.asarray(out["tip_p"], dtype=float).tolist()
    return row


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.5), sharex=True)
    for eps in sorted({row["eps_alpha"] for row in rows}):
        sub = sorted([row for row in rows if row["eps_alpha"] == eps], key=lambda row: row["alpha"])
        alphas = [row["alpha"] for row in sub]
        axes[0].plot(alphas, [row["branch_rms_max"] for row in sub], marker="o", label=f"eps={eps:g}")
        axes[1].plot(alphas, [row["linearization_error_max"] for row in sub], marker="o", label=f"eps={eps:g}")
        axes[2].plot(alphas, [1 if row["overall_publishable"] else 0 for row in sub], marker="o", label=f"eps={eps:g}")
    axes[0].axhline(1.0e-3, color="black", ls="--", lw=1)
    axes[0].set_ylabel("分支漂移 / m")
    axes[0].set_yscale("log")
    axes[1].axhline(0.05, color="black", ls="--", lw=1)
    axes[1].set_ylabel("线性化误差")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("是否可发布")
    axes[2].set_xlabel("加载比例 alpha")
    axes[2].set_ylim(-0.1, 1.1)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("P2c37 过渡区恢复复核")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_transition_recovered_map.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps_values", default="0.010,0.0125")
    parser.add_argument("--strategies", default="trf_strict_restart,lm_refine,trf_then_lm")
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0e-12)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    args = parser.parse_args()
    eps_values = [float(x.strip()) for x in args.eps_values.split(",") if x.strip()]
    heartbeat = OUT_DIR / "recovery_heartbeat.jsonl"
    failures = OUT_DIR / "recovery_failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    baseline_rows: list[dict[str, Any]] = []
    eps_rows: list[dict[str, Any]] = []
    for alpha, seed_name in _selected_specs():
        params, I_target, target_p, target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
        I_target = np.asarray(I_target, dtype=float).reshape(8)
        target_p = np.asarray(target_p, dtype=float).reshape(3)
        target_q = np.asarray(target_q, dtype=float).reshape(4)
        funcs = p2c24._make_functions(params, target_p, target_q, _make_args(args))
        branch_file = OUT_DIR / f"alpha_{alpha:.3f}_{seed_name}_baseline_branch.npz"
        if not branch_file.exists():
            _append_jsonl(failures, {"failure": "missing_selected_branch", "alpha": alpha, "seed_name": seed_name, "branch_file": str(branch_file)})
            continue
        y_base = np.asarray(np.load(branch_file)["y"], dtype=float).reshape(-1)
        baseline = _baseline_metrics(funcs, y_base, I_target, target_p, params, args)
        baseline.update({"case": "n3_mixed_rand20", "alpha": alpha, "seed_name": seed_name, "branch_file": str(branch_file)})
        baseline_rows.append(baseline)
        base_currents = I_target / alpha
        for eps in eps_values:
            row = p2c29._sensitivity_gate(
                y_base=y_base,
                alpha=alpha,
                eps=eps,
                base_currents=base_currents,
                funcs=funcs,
                target_p=target_p,
                params=params,
                args=_make_args(args),
                heartbeat=heartbeat,
            )
            row.update(
                {
                    "case": "n3_mixed_rand20",
                    "alpha": alpha,
                    "seed_name": seed_name,
                    "baseline_static_gate": baseline["static_gate"],
                    "baseline_stationarity_norm": baseline["stationarity_norm"],
                    "baseline_slack_force_ratio": baseline["max_slack_force_over_rigid_force"],
                    "overall_publishable": bool(baseline["static_gate"] and row["publishable_gate"]),
                }
            )
            eps_rows.append(row)
            if not row["overall_publishable"]:
                modes = []
                if row["branch_rms_max"] > 1.0e-3:
                    modes.append("分支漂移")
                if row["max_perturb_stationarity"] > 1.0e-5:
                    modes.append("扰动重求解一阶最优性退化")
                if row["max_perturb_slack_force_ratio"] > 2.0:
                    modes.append("扰动后载荷传播失配")
                _append_jsonl(failures, {"failure": "transition_recovery_sensitivity_failed", "alpha": alpha, "eps": eps, "modes": modes or ["证据不足"], "row": row})
    _write_csv(OUT_DIR / "transition_recovered_baselines.csv", baseline_rows)
    _write_csv(OUT_DIR / "transition_recovered_eps_summary.csv", eps_rows)
    _plot(eps_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "recovery_wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "overall": {
            "baseline_count": len(baseline_rows),
            "baseline_static_pass_count": sum(row["static_gate"] for row in baseline_rows),
            "eps_count": len(eps_rows),
            "publishable_eps_count": sum(row["overall_publishable"] for row in eps_rows),
            "diagnostic_eps_count": sum(not row["overall_publishable"] for row in eps_rows),
            "max_branch_rms": max(row["branch_rms_max"] for row in eps_rows) if eps_rows else None,
        },
        "baselines": baseline_rows,
        "eps_rows": eps_rows,
    }
    (OUT_DIR / "recovery_summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
