"""Audit branch-level wrench propagation in terminal shooting solutions."""

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
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
for path in (ROOT, P2C8_DIR, P2C10I_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10i_boundary_candidate_strict_gate as p2c10i
import run_p2c8_continuity_branch_sensitivity as p2c8

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


def _blocks(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape((-1, 6))


def _wrench_summary(y: np.ndarray) -> dict[str, Any]:
    blocks = _blocks(y)
    forces = blocks[:, 0:3]
    moments = blocks[:, 3:6]
    return {
        "block_force_norms": np.linalg.norm(forces, axis=1).tolist(),
        "block_moment_norms": np.linalg.norm(moments, axis=1).tolist(),
        "block_force_vectors": forces.tolist(),
        "block_moment_vectors": moments.tolist(),
        "block_force_total_norm": float(np.linalg.norm(forces)),
        "block_moment_total_norm": float(np.linalg.norm(moments)),
    }


def _continuity_summary(vec: np.ndarray) -> dict[str, Any]:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.size == 0:
        return {
            "continuity_norm": 0.0,
            "interface_force_jump_norms": [],
            "interface_moment_jump_norms": [],
            "interface_jump_vectors": [],
        }
    jumps = vec.reshape((-1, 6))
    return {
        "continuity_norm": float(np.linalg.norm(vec)),
        "interface_force_jump_norms": np.linalg.norm(jumps[:, 0:3], axis=1).tolist(),
        "interface_moment_jump_norms": np.linalg.norm(jumps[:, 3:6], axis=1).tolist(),
        "interface_jump_vectors": jumps.tolist(),
    }


def _pairwise_rows(case: str, alpha: float, branch_rows: list[dict[str, Any]], solves: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_name = {row["init_name"]: row for row in branch_rows}
    for a, b in combinations(sorted(by_name), 2):
        ra = by_name[a]
        rb = by_name[b]
        ya = solves[a].y
        yb = solves[b].y
        ca = np.asarray(solves[a].out["centerline"], dtype=float)
        cb = np.asarray(solves[b].out["centerline"], dtype=float)
        rows.append(
            {
                "case": case,
                "alpha": alpha,
                "branch_a": a,
                "branch_b": b,
                "tip_distance": float(np.linalg.norm(np.asarray(ra["tip_p"]) - np.asarray(rb["tip_p"]))),
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "state_distance": float(np.linalg.norm(ya - yb)),
                "force_block_distance": float(np.linalg.norm(_blocks(ya)[:, 0:3] - _blocks(yb)[:, 0:3])),
                "moment_block_distance": float(np.linalg.norm(_blocks(ya)[:, 3:6] - _blocks(yb)[:, 3:6])),
                "continuity_norm_delta": abs(float(ra["continuity_norm"]) - float(rb["continuity_norm"])),
                "boundary_norm_delta": abs(float(ra["boundary_norm"]) - float(rb["boundary_norm"])),
                "both_success": bool(ra["success"] and rb["success"]),
            }
        )
    return rows


def _plot_branch_scatter(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [f"{row['case']}@{row['alpha']:.2f}\n{row['init_name']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.7 * len(rows)), 10), sharex=True)
    axes[0].bar(x, [row["continuity_norm"] for row in rows], color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("continuity norm")
    axes[1].bar(x, [row["boundary_norm"] for row in rows], color="#ff7f0e")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("boundary norm")
    axes[2].bar(x, [row["stationarity_norm"] for row in rows], color="#2ca02c")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("stationarity")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "branch_residual_bars.png", dpi=180)
    plt.close(fig)


def _plot_pairwise(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [f"{row['case']}@{row['alpha']:.2f}\n{row['branch_a']} vs {row['branch_b']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.8 * len(rows)), 10), sharex=True)
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
    fig.savefig(OUT_DIR / "pairwise_branch_distances.png", dpi=180)
    plt.close(fig)


def _plot_centerlines(branch_rows: list[dict[str, Any]], solves: dict[str, Any], case_label: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for row in branch_rows:
        centerline = np.asarray(solves[row["init_name"]].out["centerline"], dtype=float)
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], label=row["init_name"])
        tip = centerline[-1]
        ax.scatter([tip[0]], [tip[1]], [tip[2]], s=30)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(case_label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{case_label.replace('@', '_').replace('.', 'p')}_centerlines.png", dpi=180)
    plt.close(fig)


def _audit_case(
    *,
    case_name: str,
    alpha: float,
    continuity_mu: float,
    reg_mu: float,
    max_iter: int,
    heartbeat: Path,
    failures: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target_case = p2c10i._case_by_name(case_name)
    case = p2c10i._scaled_case(target_case, alpha)
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
    bundle = p2c10i._build_case_functions(case, continuity_mu, reg_mu, max_iter)
    init_map = {
        "oracle": bundle["y_oracle"],
        "zero": np.zeros_like(bundle["y_oracle"]),
        "half_oracle": 0.5 * bundle["y_oracle"],
    }
    solves = {}
    branch_rows: list[dict[str, Any]] = []
    for init_name, y0 in init_map.items():
        solve = p2c10i.p2c7._solve_branch(
            y0=y0,
            I=bundle["I0"],
            residual_jit=bundle["residual_jit"],
            output_jit=bundle["output_jit"],
            Jy_jit=bundle["Jy_jit"],
            max_iter=max_iter,
        )
        solves[init_name] = solve
        cont = np.asarray(
            jax.device_get(bundle["continuity_jit"](jnp.asarray(solve.y), jnp.asarray(bundle["I0"]))),
            dtype=float,
        )
        cont_summary = _continuity_summary(cont)
        wrench = _wrench_summary(solve.y)
        row = {
            "case": case_name,
            "scaled_case": case.name,
            "alpha": alpha,
            "continuity_mu": continuity_mu,
            "reg_mu": reg_mu,
            "init_name": init_name,
            "success": solve.success,
            "residual_norm": solve.residual_norm,
            "stationarity_norm": solve.stationarity_norm,
            "boundary_norm": solve.boundary_norm,
            "steady_s": solve.steady_s,
            "tip_p": np.asarray(solve.out["tip_p"], dtype=float).tolist(),
            "boundary_vector": np.asarray(solve.out["boundary"], dtype=float).tolist(),
            **cont_summary,
            **wrench,
        }
        branch_rows.append(row)
        if not solve.success or cont_summary["continuity_norm"] > 1.0e-2:
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "init_name": init_name, "failure": "branch_or_continuity_defect", "evidence": row})
    pair_rows = _pairwise_rows(case_name, alpha, branch_rows, solves)
    _plot_centerlines(branch_rows, solves, f"{case_name}@{alpha:.2f}")
    np.savez_compressed(
        OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_branches.npz",
        **{f"{name}_y": solve.y for name, solve in solves.items()},
        **{f"{name}_centerline": np.asarray(solve.out["centerline"], dtype=float) for name, solve in solves.items()},
    )
    summary = {
        "case": case_name,
        "alpha": alpha,
        "continuity_mu": continuity_mu,
        "reg_mu": reg_mu,
        "branch_count": len(branch_rows),
        "success_count": sum(1 for row in branch_rows if row["success"]),
        "max_tip_distance": max([row["tip_distance"] for row in pair_rows], default=0.0),
        "max_centerline_rms": max([row["centerline_rms"] for row in pair_rows], default=0.0),
        "max_state_distance": max([row["state_distance"] for row in pair_rows], default=0.0),
        "min_continuity_norm": min(row["continuity_norm"] for row in branch_rows),
        "max_continuity_norm": max(row["continuity_norm"] for row in branch_rows),
        "min_boundary_norm": min(row["boundary_norm"] for row in branch_rows),
        "max_boundary_norm": max(row["boundary_norm"] for row in branch_rows),
        "all_success": all(row["success"] for row in branch_rows),
    }
    return branch_rows, pair_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=32)
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
        branch_rows, pair_rows, case_summary = _audit_case(
            case_name=case_name,
            alpha=alpha,
            continuity_mu=args.continuity_mu,
            reg_mu=args.reg_mu,
            max_iter=args.max_iter,
            heartbeat=heartbeat,
            failures=failures,
        )
        all_branch_rows.extend(branch_rows)
        all_pair_rows.extend(pair_rows)
        case_summaries.append(case_summary)

    total_wall_s = float(time.perf_counter() - t_all)
    _plot_branch_scatter(all_branch_rows)
    _plot_pairwise(all_pair_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": total_wall_s,
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
        },
        "classification": "段间载荷传播与多分支审计",
        "case_summaries": case_summaries,
        "branch_rows": all_branch_rows,
        "pair_rows": all_pair_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "branch_summary.csv", all_branch_rows)
    _write_csv(OUT_DIR / "pairwise_summary.csv", all_pair_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
