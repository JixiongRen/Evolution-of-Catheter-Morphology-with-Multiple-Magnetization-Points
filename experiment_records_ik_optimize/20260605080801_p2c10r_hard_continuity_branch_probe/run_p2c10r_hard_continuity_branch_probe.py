"""Probe whether hard interface continuity merges shooting branches."""

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
P2C9D_DIR = ROOT / "experiment_records_ik_optimize" / "20260604132850_p2c9d_hard_continuity_kkt_probe"
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
for path in (ROOT, P2C9D_DIR, P2C10I_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10i_boundary_candidate_strict_gate as p2c10i
import run_p2c9d_hard_continuity_kkt_probe as p2c9d

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


def _parse_candidates(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def _build_case(case_name: str, alpha: float) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_case = p2c10i._case_by_name(case_name)
    case = p2c10i._scaled_case(target_case, alpha)
    engine = p2c10i.p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c10i.p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c10i.p2c2._oracle_terminal_pose(oracle, engine)
    y_ref = p2c10i.p2c2.p2c1._y_guess_from_oracle(oracle["z"], oracle["params"], "segment_wrench")
    return oracle["params"], I0, np.asarray(target_p, dtype=float), np.asarray(target_q, dtype=float), np.asarray(y_ref, dtype=float)


def _solve_branches(
    *,
    case_name: str,
    alpha: float,
    anchor_mu: float,
    constraint_scale: float,
    max_iter: int,
    heartbeat: Path,
    failures: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha})
    params, I0, target_p, target_q, y_ref = _build_case(case_name, alpha)
    funcs = p2c9d._make_functions(
        params_base=params,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_ref,
        anchor_mu=anchor_mu,
        constraint_scale=constraint_scale,
        steps_per_flex=4,
    )
    init_map = {
        "oracle": y_ref,
        "half_oracle": 0.5 * y_ref,
        "zero": np.zeros_like(y_ref),
    }
    rows: list[dict[str, Any]] = []
    solves: dict[str, Any] = {}
    for name, y0 in init_map.items():
        v0 = np.concatenate([np.asarray(y0, dtype=float).reshape(-1), np.zeros(int(funcs["n_c"]), dtype=float)])
        _ = funcs["kkt_jit"](jnp.asarray(v0), jnp.asarray(I0)).block_until_ready()
        solve = p2c9d._solve_kkt(v0=v0, I=I0, funcs=funcs, max_iter=max_iter)
        solves[name] = solve
        F_v = np.asarray(jax.device_get(funcs["kkt_v_jit"](jnp.asarray(solve.v), jnp.asarray(I0))), dtype=float)
        svals = np.linalg.svd(F_v, compute_uv=False)
        row = {
            "case": case_name,
            "alpha": alpha,
            "init_name": name,
            "success": solve.success,
            "n_iter": solve.n_iter,
            "steady_s": solve.steady_s,
            "kkt_norm": solve.kkt_norm,
            "objective_norm": solve.objective_norm,
            "constraint_norm": solve.constraint_norm,
            "stationarity_norm": solve.stationarity_norm,
            "boundary_norm": solve.boundary_norm,
            "tip_p": np.asarray(solve.out["tip_p"], dtype=float).tolist(),
            "kkt_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
            "kkt_rank": int(np.linalg.matrix_rank(F_v)),
        }
        rows.append(row)
        if not solve.success:
            _append_jsonl(failures, {"ts": time.time(), "case": case_name, "alpha": alpha, "init_name": name, "failure": "hard_kkt_branch_failed", "evidence": row})
    pair_rows = []
    by_name = {row["init_name"]: row for row in rows}
    for a, b in combinations(sorted(by_name), 2):
        sa = solves[a]
        sb = solves[b]
        ca = np.asarray(sa.out["centerline"], dtype=float)
        cb = np.asarray(sb.out["centerline"], dtype=float)
        pair_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "branch_a": a,
                "branch_b": b,
                "tip_distance": float(np.linalg.norm(np.asarray(sa.out["tip_p"]) - np.asarray(sb.out["tip_p"]))),
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "state_distance": float(np.linalg.norm(sa.y - sb.y)),
                "lambda_distance": float(np.linalg.norm(sa.lam - sb.lam)),
                "both_success": bool(sa.success and sb.success),
            }
        )
    summary = {
        "case": case_name,
        "alpha": alpha,
        "success_count": sum(1 for row in rows if row["success"]),
        "branch_count": len(rows),
        "max_tip_distance": max([row["tip_distance"] for row in pair_rows], default=0.0),
        "max_centerline_rms": max([row["centerline_rms"] for row in pair_rows], default=0.0),
        "max_state_distance": max([row["state_distance"] for row in pair_rows], default=0.0),
        "max_kkt_norm": max(row["kkt_norm"] for row in rows),
        "max_constraint_norm": max(row["constraint_norm"] for row in rows),
        "max_boundary_norm": max(row["boundary_norm"] for row in rows),
        "max_kkt_condition": max(row["kkt_condition"] for row in rows),
        "branch_consistent": bool(
            all(row["success"] for row in rows)
            and max([row["tip_distance"] for row in pair_rows], default=0.0) <= 1.0e-4
            and max([row["centerline_rms"] for row in pair_rows], default=0.0) <= 1.0e-3
        ),
    }
    np.savez_compressed(
        OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_hard_kkt_branches.npz",
        **{f"{name}_v": solve.v for name, solve in solves.items()},
        **{f"{name}_y": solve.y for name, solve in solves.items()},
        **{f"{name}_centerline": np.asarray(solve.out["centerline"], dtype=float) for name, solve in solves.items()},
    )
    return rows, pair_rows, summary


def _plot_branch_metrics(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}\n{row['init_name']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.8 * len(rows)), 9), sharex=True)
    axes[0].bar(x, [row["constraint_norm"] for row in rows], color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("constraint")
    axes[1].bar(x, [row["boundary_norm"] for row in rows], color="#ff7f0e")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("boundary")
    axes[2].bar(x, [row["kkt_condition"] for row in rows], color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("KKT condition")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hard_kkt_branch_metrics.png", dpi=180)
    plt.close(fig)


def _plot_pairwise(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}\n{row['branch_a']} vs {row['branch_b']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.9 * len(rows)), 9), sharex=True)
    axes[0].bar(x, [row["tip_distance"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("tip distance")
    axes[1].bar(x, [row["centerline_rms"] for row in rows], color="#ff7f0e")
    axes[1].set_ylabel("centerline RMS")
    axes[2].bar(x, [row["state_distance"] for row in rows], color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("state distance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hard_kkt_pairwise_distances.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00")
    parser.add_argument("--anchor_mu", type=float, default=1.0e-6)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=80)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()
    all_branch_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case_name, alpha in _parse_candidates(args.candidates):
        branch_rows, pair_rows, case_summary = _solve_branches(
            case_name=case_name,
            alpha=alpha,
            anchor_mu=args.anchor_mu,
            constraint_scale=args.constraint_scale,
            max_iter=args.max_iter,
            heartbeat=heartbeat,
            failures=failures,
        )
        all_branch_rows.extend(branch_rows)
        all_pair_rows.extend(pair_rows)
        case_summaries.append(case_summary)
    _plot_branch_metrics(all_branch_rows)
    _plot_pairwise(all_pair_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "anchor_mu": args.anchor_mu,
            "constraint_scale": args.constraint_scale,
            "max_iter": args.max_iter,
        },
        "classification": "硬连续性KKT分支合并复测",
        "case_summaries": case_summaries,
        "branch_rows": all_branch_rows,
        "pair_rows": all_pair_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "hard_kkt_branch_summary.csv", all_branch_rows)
    _write_csv(OUT_DIR / "hard_kkt_pairwise_summary.csv", all_pair_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
