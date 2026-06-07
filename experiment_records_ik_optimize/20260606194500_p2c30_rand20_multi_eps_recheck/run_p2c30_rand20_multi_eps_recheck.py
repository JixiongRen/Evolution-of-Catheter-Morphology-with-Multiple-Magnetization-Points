"""Recheck n3 mixed random-current sensitivity over multiple perturbations."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C25_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170041_p2c25_scaled_stationarity_and_interface_audit"
P2C27_DIR = ROOT / "experiment_records_ik_optimize" / "20260606172200_p2c27_stationarity_refinement_gate"
P2C29_DIR = ROOT / "experiment_records_ik_optimize" / "20260606175600_p2c29_rand20_load_continuation_refine"
for path in (ROOT, P2C24_DIR, P2C25_DIR, P2C27_DIR, P2C29_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

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


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _plot(rows: list[dict[str, Any]]) -> None:
    eps = [r["eps_alpha"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    specs = [
        ("publishable_gate", "是否可发布", None),
        ("linearization_error_max", "最大线性化误差", 0.05),
        ("max_perturb_stationarity", "扰动后一阶最优性", 1.0e-5),
        ("branch_rms_max", "最大轴线分支漂移 m", 1.0e-3),
    ]
    axes[0].bar(eps, [1 if r["publishable_gate"] else 0 for r in rows], width=0.003)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel(specs[0][1])
    for ax, (key, ylabel, gate) in zip(axes[1:], specs[1:]):
        ax.plot(eps, [r[key] for r in rows], "o-")
        if gate is not None:
            ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("alpha 扰动幅值")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rand20_multi_eps_gate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--target_alpha", type=float, default=0.4)
    parser.add_argument("--eps_values", default="0.005,0.01,0.02")
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

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()

    params, _I_target = p2c24._params_from_saved_case(args.case, args.target_alpha)
    target_p, y_saved = p2c24._load_saved_target_and_seed(args.case, args.target_alpha)
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    funcs = p2c24._make_functions(params, target_p, target_q, args)
    base_currents = np.asarray(p2c24._case_spec(args.case).currents, dtype=float)

    baseline, y_base, _rows = p2c29._compete(
        label="baseline_direct_saved",
        alpha=args.target_alpha,
        funcs=funcs,
        y0=np.asarray(y_saved, dtype=float),
        I=base_currents * args.target_alpha,
        target_p=target_p,
        params=params,
        args=args,
        heartbeat=heartbeat,
    )

    eps_rows: list[dict[str, Any]] = []
    for eps in _parse_floats(args.eps_values):
        row = p2c29._sensitivity_gate(
            y_base=y_base,
            alpha=args.target_alpha,
            eps=eps,
            base_currents=base_currents,
            funcs=funcs,
            target_p=target_p,
            params=params,
            args=args,
            heartbeat=heartbeat,
        )
        row.update(
            {
                "case": args.case,
                "alpha": args.target_alpha,
                "baseline_static_gate": baseline["static_gate"],
                "baseline_stationarity_norm": baseline["stationarity_norm"],
                "baseline_slack_force_ratio": baseline["max_slack_force_over_rigid_force"],
                "overall_publishable": bool(baseline["static_gate"] and row["publishable_gate"]),
            }
        )
        eps_rows.append(row)
        np.savez_compressed(
            OUT_DIR / f"{args.case}_eps_{eps:.4f}_multi_eps.npz",
            sensitivity=np.asarray(row["sensitivity_vector"], dtype=float),
        )
        if not row["overall_publishable"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "multi_eps_gate_failed", "case": args.case, "alpha": args.target_alpha, "eps": eps, "evidence": row})

    _write_csv(OUT_DIR / "eps_summary.csv", eps_rows)
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
        "baseline": baseline,
        "overall": {
            "eps_count": len(eps_rows),
            "publishable_eps_count": sum(r["overall_publishable"] for r in eps_rows),
            "diagnostic_eps_count": sum(not r["overall_publishable"] for r in eps_rows),
            "max_linearization_error": max(r["linearization_error_max"] for r in eps_rows),
            "max_perturb_stationarity": max(r["max_perturb_stationarity"] for r in eps_rows),
            "max_branch_rms": max(r["branch_rms_max"] for r in eps_rows),
        },
        "eps_summaries": eps_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
