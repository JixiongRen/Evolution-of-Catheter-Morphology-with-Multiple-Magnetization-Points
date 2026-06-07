"""Reproduce trusted branches for problematic alpha points."""

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
P2C16_DIR = ROOT / "experiment_records_ik_optimize" / "20260605132600_p2c16_alpha_boundary_phase_map"
P2C22_DIR = ROOT / "experiment_records_ik_optimize" / "20260606152302_p2c22_cross_condition_reliability_pilot"
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C29_DIR = ROOT / "experiment_records_ik_optimize" / "20260606175600_p2c29_rand20_load_continuation_refine"
P2C34_DIR = ROOT / "experiment_records_ik_optimize" / "20260606210000_p2c34_arbitrary_alpha_canonical_state_generation"
for path in (ROOT, P2C10W_DIR, P2C16_DIR, P2C22_DIR, P2C24_DIR, P2C29_DIR, P2C34_DIR):
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


def _load_npz_seed(path: Path, key: str) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(np.load(path)[key], dtype=float).reshape(-1)


def _candidate_seeds(alpha: float, y_ref: np.ndarray) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = [
        ("oracle_seed", np.asarray(y_ref, dtype=float).reshape(-1)),
        ("zero_seed", np.zeros_like(np.asarray(y_ref, dtype=float).reshape(-1))),
    ]
    p34 = P2C34_DIR / f"n3_mixed_rand20_alpha_{alpha:.3f}_canonical_state.npz"
    y = _load_npz_seed(p34, "best_y")
    if y is not None:
        seeds.append(("p2c34_canonical_seed", y))
    if abs(alpha - 0.4) < 1.0e-12:
        y = _load_npz_seed(P2C22_DIR / "n3_mixed_rand20_alpha_0.40_position_linearity.npz", "best_y")
        if y is not None:
            seeds.append(("p2c22_saved_trusted_seed", y))
    if abs(alpha - 0.425) < 1.0e-12:
        y = _load_npz_seed(P2C16_DIR / "alpha_0.425_eps_0.010_phase_map.npz", "base_y")
        if y is not None:
            seeds.append(("p2c16_historical_base_seed", y))
        y = _load_npz_seed(P2C34_DIR / "n3_mixed_rand20_alpha_0.450_canonical_state.npz", "best_y")
        if y is not None:
            seeds.append(("reverse_from_alpha_0.450_seed", y))
        y = _load_npz_seed(P2C34_DIR / "n3_mixed_rand20_alpha_0.375_canonical_state.npz", "best_y")
        if y is not None:
            seeds.append(("forward_from_alpha_0.375_seed", y))
    unique: list[tuple[str, np.ndarray]] = []
    seen = set()
    for name, seed in seeds:
        key = (name, tuple(np.round(seed, 12)))
        if key not in seen:
            seen.add(key)
            unique.append((name, seed))
    return unique


def _solve_candidates(alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params, I_target, target_p, target_q, y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    I_target = np.asarray(I_target, dtype=float).reshape(8)
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    target_q = np.asarray(target_q, dtype=float).reshape(4)
    funcs = p2c24._make_functions(params, target_p, target_q, _make_args(args))
    rows: list[dict[str, Any]] = []
    strategy_rows_all: list[dict[str, Any]] = []
    centers: dict[str, np.ndarray] = {}
    for seed_name, y0 in _candidate_seeds(alpha, y_ref):
        _append_jsonl(heartbeat, {"phase": "candidate_start", "alpha": alpha, "seed": seed_name, "ts": time.time()})
        best, y, strategy_rows = p2c29._compete(
            label=seed_name,
            alpha=alpha,
            funcs=funcs,
            y0=y0,
            I=I_target,
            target_p=target_p,
            params=params,
            args=_make_args(args),
            heartbeat=heartbeat,
        )
        out = p2c24.p2c10w.output_to_numpy(funcs["output_jit"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I_target, dtype=jnp.float64)))
        center = np.asarray(out["centerline"], dtype=float)
        centers[seed_name] = center
        np.savez_compressed(
            OUT_DIR / f"alpha_{alpha:.3f}_{seed_name}_branch.npz",
            alpha=alpha,
            seed_name=seed_name,
            y=np.asarray(y, dtype=float),
            centerline=center,
            tip_p=np.asarray(out["tip_p"], dtype=float),
            tip_q=np.asarray(out["tip_q"], dtype=float),
            target_p=target_p,
            target_q=target_q,
            I_target=I_target,
        )
        row = dict(best)
        row.update(
            {
                "case": "n3_mixed_rand20",
                "alpha": alpha,
                "seed_name": seed_name,
                "branch_file": str(OUT_DIR / f"alpha_{alpha:.3f}_{seed_name}_branch.npz"),
                "tip_p": np.asarray(out["tip_p"], dtype=float).tolist(),
                "target_p": target_p.tolist(),
                "candidate_publishable_baseline": bool(best["static_gate"]),
            }
        )
        rows.append(row)
        for srow in strategy_rows:
            rec = dict(srow)
            rec.update({"case": "n3_mixed_rand20", "alpha": alpha, "seed_name": seed_name})
            strategy_rows_all.append(rec)
        if not row["static_gate"]:
            modes = []
            if row["stationarity_norm"] > 1.0e-5:
                modes.append("一阶最优性退化")
            if row["max_slack_force_over_rigid_force"] > 2.0:
                modes.append("刚性磁体段载荷传播失配")
            if row["position_error_m"] > 0.05 * 0.11:
                modes.append("末端位置误差超限")
            _append_jsonl(failures, {"failure": "branch_candidate_not_static", "alpha": alpha, "seed": seed_name, "modes": modes or ["证据不足"], "row": row})
        _append_jsonl(heartbeat, {"phase": "candidate_done", "alpha": alpha, "seed": seed_name, "static_gate": row["static_gate"], "ts": time.time()})
    for row in rows:
        diffs = []
        for other in rows:
            if row["seed_name"] == other["seed_name"]:
                continue
            a = centers[row["seed_name"]]
            b = centers[other["seed_name"]]
            diffs.append(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))))
        row["max_centerline_rms_to_other_candidates"] = max(diffs) if diffs else 0.0
    return rows, strategy_rows_all


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = [f"{r['alpha']:.3f}\n{r['seed_name']}" for r in rows]
    x = np.arange(len(rows))
    colors = ["#2ca02c" if r["static_gate"] else "#d62728" for r in rows]
    specs = [
        ("stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("max_slack_force_over_rigid_force", "界面松弛力/刚段外力", 2.0),
        ("max_centerline_rms_to_other_candidates", "候选分支最大中心线差异 / m", 1.0e-3),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [r[key] for r in rows], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle("P2c35 多初值分支复现门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "branch_reproduction_gates.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for row in rows:
        data = np.load(row["branch_file"])
        pts = np.asarray(data["centerline"], dtype=float)
        label = f"a={row['alpha']:.3f} {row['seed_name']} {'通过' if row['static_gate'] else '未过'}"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=20)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("多候选分支中心线")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "branch_reproduction_centerlines.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.4,0.425")
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
    rows_all: list[dict[str, Any]] = []
    strategy_rows_all: list[dict[str, Any]] = []
    for alpha_text in args.alphas.split(","):
        alpha = float(alpha_text.strip())
        rows, strategy_rows = _solve_candidates(alpha, args, heartbeat, failures)
        rows_all.extend(rows)
        strategy_rows_all.extend(strategy_rows)
    _write_csv(OUT_DIR / "branch_candidates.csv", rows_all)
    _write_csv(OUT_DIR / "branch_strategy_rows.csv", strategy_rows_all)
    _plot(rows_all)
    by_alpha: dict[str, dict[str, Any]] = {}
    for alpha in sorted({float(row["alpha"]) for row in rows_all}):
        alpha_rows = [row for row in rows_all if abs(float(row["alpha"]) - alpha) < 1.0e-12]
        pass_rows = [row for row in alpha_rows if row["static_gate"]]
        best = min(alpha_rows, key=lambda r: (not r["static_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"]))
        by_alpha[f"{alpha:.3f}"] = {
            "candidate_count": len(alpha_rows),
            "static_pass_count": len(pass_rows),
            "best_seed": best["seed_name"],
            "best_static_gate": best["static_gate"],
            "best_stationarity_norm": best["stationarity_norm"],
            "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
            "best_centerline_rms_to_other_candidates": best["max_centerline_rms_to_other_candidates"],
        }
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
            "candidate_count": len(rows_all),
            "static_pass_count": sum(row["static_gate"] for row in rows_all),
            "alpha_summary": by_alpha,
        },
        "rows": rows_all,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
