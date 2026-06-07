"""Replay sensitivity gates from trusted physical branches."""

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
P2C35_DIR = ROOT / "experiment_records_ik_optimize" / "20260606213500_p2c35_branch_reproduction_repair"
for path in (ROOT, P2C10W_DIR, P2C24_DIR, P2C25_DIR, P2C29_DIR, P2C35_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
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


def _trusted_branch_specs() -> list[dict[str, Any]]:
    return [
        {
            "alpha": 0.4,
            "seed_name": "p2c22_saved_trusted_seed",
            "branch_file": P2C35_DIR / "alpha_0.400_p2c22_saved_trusted_seed_branch.npz",
        },
        {
            "alpha": 0.425,
            "seed_name": "p2c16_historical_base_seed",
            "branch_file": P2C35_DIR / "alpha_0.425_p2c16_historical_base_seed_branch.npz",
        },
    ]


def _baseline_row(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray, target_p: np.ndarray, params: Any, args: argparse.Namespace) -> dict[str, Any]:
    metrics = p2c25._physical_metrics(funcs, y, I, args.boundary_scale, args.slack_scale)
    load = p2c24._audit_load(params, I, y)
    out = metrics["out"]
    pos_err = float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float) - target_p))
    row = {
        "position_error_m": pos_err,
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
    row["centerline"] = np.asarray(out["centerline"], dtype=float)
    return row


def _run_branch(spec: dict[str, Any], eps_values: list[float], args: argparse.Namespace, heartbeat: Path, failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alpha = float(spec["alpha"])
    params, I_target, target_p, target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    I_target = np.asarray(I_target, dtype=float).reshape(8)
    base_currents = I_target / alpha
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    target_q = np.asarray(target_q, dtype=float).reshape(4)
    funcs = p2c24._make_functions(params, target_p, target_q, _make_args(args))
    branch = np.load(spec["branch_file"])
    y_base = np.asarray(branch["y"], dtype=float).reshape(-1)
    baseline = _baseline_row(funcs, y_base, I_target, target_p, params, args)
    baseline.pop("centerline")
    baseline.update(
        {
            "case": "n3_mixed_rand20",
            "alpha": alpha,
            "seed_name": spec["seed_name"],
            "branch_file": str(spec["branch_file"]),
        }
    )

    eps_rows: list[dict[str, Any]] = []
    for eps in eps_values:
        _append_jsonl(heartbeat, {"phase": "eps_start", "alpha": alpha, "eps": eps, "ts": time.time()})
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
                "seed_name": spec["seed_name"],
                "baseline_static_gate": baseline["static_gate"],
                "baseline_stationarity_norm": baseline["stationarity_norm"],
                "baseline_slack_force_ratio": baseline["max_slack_force_over_rigid_force"],
                "overall_publishable": bool(baseline["static_gate"] and row["publishable_gate"]),
            }
        )
        eps_rows.append(row)
        np.savez_compressed(
            OUT_DIR / f"alpha_{alpha:.3f}_eps_{eps:.4f}_trusted_branch_sensitivity.npz",
            sensitivity=np.asarray(row["sensitivity_vector"], dtype=float),
            baseline_y=y_base,
            target_p=target_p,
            target_q=target_q,
            I_target=I_target,
        )
        if not row["overall_publishable"]:
            modes = []
            if not baseline["static_gate"]:
                modes.append("基线静态门禁未通过")
            if row["max_perturb_stationarity"] > 1.0e-5:
                modes.append("扰动重求解一阶最优性退化")
            if row["linearization_error_max"] > 0.05:
                modes.append("局部线性化误差超限")
            if row["branch_rms_max"] > 1.0e-3:
                modes.append("分支漂移")
            if row["max_perturb_slack_force_ratio"] > 2.0:
                modes.append("扰动后载荷传播失配")
            _append_jsonl(failures, {"failure": "trusted_branch_sensitivity_gate_failed", "alpha": alpha, "eps": eps, "modes": modes or ["证据不足"], "row": row})
        _append_jsonl(heartbeat, {"phase": "eps_done", "alpha": alpha, "eps": eps, "publishable": row["overall_publishable"], "ts": time.time()})
    return baseline, eps_rows


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 11.0), sharex=False)
    alphas = sorted({float(row["alpha"]) for row in rows})
    for alpha in alphas:
        sub = [row for row in rows if abs(float(row["alpha"]) - alpha) < 1.0e-12]
        sub = sorted(sub, key=lambda r: r["eps_alpha"])
        eps = [r["eps_alpha"] for r in sub]
        label = f"alpha={alpha:.3f}"
        axes[0].plot(eps, [r["sensitivity_norm_m_per_alpha"] for r in sub], marker="o", label=label)
        axes[1].plot(eps, [r["linearization_error_max"] for r in sub], marker="o", label=label)
        axes[2].plot(eps, [r["branch_rms_max"] for r in sub], marker="o", label=label)
        axes[3].plot(eps, [1 if r["overall_publishable"] else 0 for r in sub], marker="o", label=label)
    axes[0].set_ylabel("敏感度模长 / (m/alpha)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("局部线性化误差")
    axes[1].set_yscale("log")
    axes[1].axhline(0.05, color="black", ls="--", lw=1)
    axes[2].set_ylabel("轴线分支漂移 / m")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1)
    axes[3].set_ylabel("是否可发布")
    axes[3].set_xlabel("扰动半径 eps")
    axes[3].set_ylim(-0.1, 1.1)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("可信分支上的严格敏感度复核")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trusted_branch_sensitivity_gates.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps_values", default="0.005,0.01,0.0125")
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

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    baseline_rows: list[dict[str, Any]] = []
    eps_rows: list[dict[str, Any]] = []
    for spec in _trusted_branch_specs():
        baseline, rows = _run_branch(spec, eps_values, args, heartbeat, failures)
        baseline_rows.append(baseline)
        eps_rows.extend(rows)
    _write_csv(OUT_DIR / "trusted_branch_baselines.csv", baseline_rows)
    _write_csv(OUT_DIR / "trusted_branch_eps_summary.csv", eps_rows)
    _plot(eps_rows)
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
            "baseline_count": len(baseline_rows),
            "baseline_static_pass_count": sum(row["static_gate"] for row in baseline_rows),
            "eps_count": len(eps_rows),
            "publishable_eps_count": sum(row["overall_publishable"] for row in eps_rows),
            "diagnostic_eps_count": sum(not row["overall_publishable"] for row in eps_rows),
            "max_branch_rms": max(row["branch_rms_max"] for row in eps_rows),
            "max_linearization_error": max(row["linearization_error_max"] for row in eps_rows),
        },
        "baselines": baseline_rows,
        "eps_rows": eps_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
