"""Generate canonical arbitrary-alpha states for strict sensitivity replay."""

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
P2C29_DIR = ROOT / "experiment_records_ik_optimize" / "20260606175600_p2c29_rand20_load_continuation_refine"
for path in (ROOT, P2C10W_DIR, P2C24_DIR, P2C29_DIR):
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


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _zero_seed_like(y: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(y, dtype=float).reshape(-1))


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


def _solve_alpha(
    *,
    case: str,
    alpha: float,
    previous_y: np.ndarray | None,
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], np.ndarray | None]:
    params, I_target, target_p, target_q, y_ref, _oracle = p2c10w._build_case(case, alpha)
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    target_q = np.asarray(target_q, dtype=float).reshape(4)
    I_target = np.asarray(I_target, dtype=float).reshape(8)
    funcs = p2c24._make_functions(params, target_p, target_q, _make_args(args))

    seeds: list[tuple[str, np.ndarray]] = [
        ("oracle_seed", np.asarray(y_ref, dtype=float).reshape(-1)),
        ("zero_seed", _zero_seed_like(y_ref)),
    ]
    if previous_y is not None:
        seeds.append(("previous_alpha_continuation", np.asarray(previous_y, dtype=float).reshape(-1)))

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_y: np.ndarray | None = None
    for seed_name, y0 in seeds:
        row, y, strategy_rows = p2c29._compete(
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
        row = dict(row)
        row.update({"case": case, "alpha": alpha, "seed_name": seed_name, "strategy_count": len(strategy_rows)})
        rows.append(row)
        if best_row is None or (not row["static_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]) < (
            not best_row["static_gate"],
            best_row["stationarity_norm"],
            best_row["max_slack_force_over_rigid_force"],
        ):
            best_row = row
            best_y = y

    if best_row is None or best_y is None:
        raise RuntimeError(f"No candidate solved for {case} alpha={alpha}")

    out = p2c24.p2c10w.output_to_numpy(funcs["output_jit"](jnp.asarray(best_y, dtype=jnp.float64), jnp.asarray(I_target, dtype=jnp.float64)))
    state_path = OUT_DIR / f"{case}_alpha_{alpha:.3f}_canonical_state.npz"
    np.savez_compressed(
        state_path,
        case=case,
        alpha=float(alpha),
        target_p=target_p,
        target_q=target_q,
        I_target=I_target,
        best_y=np.asarray(best_y, dtype=float),
        centerline=np.asarray(out["centerline"], dtype=float),
        tip_p=np.asarray(out["tip_p"], dtype=float),
        tip_q=np.asarray(out["tip_q"], dtype=float),
    )
    row_out = dict(best_row)
    row_out.update(
        {
            "state_path": str(state_path),
            "candidate_count": len(rows),
            "target_p": target_p.tolist(),
            "tip_p": np.asarray(out["tip_p"], dtype=float).tolist(),
        }
    )
    if not row_out["static_gate"]:
        modes = []
        if row_out["stationarity_norm"] > 1.0e-5:
            modes.append("一阶最优性退化")
        if row_out["max_slack_force_over_rigid_force"] > 2.0:
            modes.append("刚性磁体段载荷传播失配")
        if row_out["position_error_m"] > 0.05 * 0.11:
            modes.append("末端位置误差超限")
        _append_jsonl(
            failures,
            {
                "failure": "canonical_alpha_state_not_static",
                "case": case,
                "alpha": alpha,
                "modes": modes or ["证据不足"],
                "best_row": row_out,
                "all_candidates": rows,
            },
        )
    return row_out, np.asarray(best_y, dtype=float)


def _plot(rows: list[dict[str, Any]]) -> None:
    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    colors = ["#2ca02c" if row["static_gate"] else "#d62728" for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 10.5), sharex=True)
    specs = [
        ("stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("max_slack_force_over_rigid_force", "界面松弛力/刚段外力", 2.0),
        ("position_error_m", "末端位置误差 / m", 0.05 * 0.11),
        ("jacobian_condition", "雅可比矩阵条件数", 1.0e10),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.scatter(alphas, [row[key] for row in rows], c=colors, s=70)
        ax.plot(alphas, [row[key] for row in rows], color="0.35", lw=1)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("加载比例 alpha")
    fig.suptitle("任意 alpha 统一目标与初值生成后的静态门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "canonical_alpha_state_gates.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for row in rows:
        data = np.load(row["state_path"])
        pts = np.asarray(data["centerline"], dtype=float)
        label = f"a={row['alpha']:.3f} {'通过' if row['static_gate'] else '未通过'}"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("统一口径基线中心线")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "canonical_alpha_centerlines.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.375,0.4,0.425,0.45")
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
    rows: list[dict[str, Any]] = []
    previous_y: np.ndarray | None = None
    for alpha in _parse_floats(args.alphas):
        _append_jsonl(heartbeat, {"phase": "alpha_start", "case": args.case, "alpha": alpha, "ts": time.time()})
        row, previous_y = _solve_alpha(
            case=args.case,
            alpha=alpha,
            previous_y=previous_y,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        rows.append(row)
        _append_jsonl(heartbeat, {"phase": "alpha_done", "case": args.case, "alpha": alpha, "static_gate": row["static_gate"], "ts": time.time()})
    _write_csv(OUT_DIR / "canonical_alpha_states.csv", rows)
    _plot(rows)
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
            "alpha_count": len(rows),
            "static_pass_count": sum(row["static_gate"] for row in rows),
            "static_fail_count": sum(not row["static_gate"] for row in rows),
        },
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
