"""Run clean terminal-pose shooting without oracle shape leakage."""

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
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(P2C2_DIR) not in sys.path:
    sys.path.insert(0, str(P2C2_DIR))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot_pose(case_dir: Path, oracle: dict[str, Any], engine: Any, out: dict[str, np.ndarray], title: str) -> None:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = p2c2.p2c1._arc_resample(out["centerline"], len(oracle_cl))
    p_target, _q_target = p2c2._oracle_terminal_pose(oracle, engine)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.0, label="FK oracle centerline")
    ax.plot(shoot_cl[:, 0], shoot_cl[:, 1], shoot_cl[:, 2], "r--", lw=2.0, label="clean shooting")
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=40, label="base")
    ax.scatter([p_target[0]], [p_target[1]], [p_target[2]], c="green", s=55, label="target terminal")
    ax.scatter([out["tip_p"][0]], [out["tip_p"][1]], [out["tip_p"][2]], c="red", s=45, label="shooting terminal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "clean_terminal_oracle_vs_shooting.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    diffs = np.linalg.norm(shoot_cl - oracle_cl, axis=1)
    ax.plot(np.arange(len(diffs)), diffs, marker=".")
    ax.set_title("Pointwise centerline drift")
    ax.set_xlabel("resampled point index")
    ax.set_ylabel("distance (m)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / "clean_centerline_pointwise_error.png", dpi=180)
    plt.close(fig)


def _plot_history(case_dir: Path, history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(len(history)), history, marker=".")
    ax.set_title("Clean terminal-pose residual history")
    ax.set_xlabel("residual evaluation")
    ax.set_ylabel("||residual||")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(case_dir / "clean_residual_history.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--candidate_max_nfev", type=int, default=16)
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = p2c2.p2c1.p2c._build_engine((1, 1, 1), max_iter=int(args.max_iter), tol=float(args.tol))
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    length = float(np.sum(engine.rigid_lengths) + sum(engine._compute_flex_lengths(L_protrude=0.11)))

    spec = p2c2.PoseCandidateSpec(
        "c3_clean_terminal_pose",
        max_nfev=int(args.candidate_max_nfev),
        centerline_weight=0.0,
        tip_weight=120.0,
        so3_weight=2.0,
        rigid_weight=0.0,
        boundary_weight=0.02,
    )
    result = p2c2._solve_pose_candidate(spec, oracle, engine, steps_per_flex=int(args.steps_per_flex))
    metrics = p2c2._terminal_metrics(oracle, engine, result.out)
    row = {
        "candidate": spec.name,
        "data_leakage": False,
        "uses_centerline_residual": False,
        "uses_rigid_midpoint_residual": False,
        "uses_terminal_position_residual": True,
        "uses_terminal_so3_residual": True,
        "uses_boundary_residual": True,
        "state_mode": "segment_wrench",
        "state_dim": int(result.y.size),
        "success": bool(result.success),
        "wall_s": float(result.wall_s),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "pose_stage": p2c2._pose_stage(metrics, length),
        **metrics,
    }

    case_dir = OUT_DIR / "M111"
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))
    np.save(case_dir / "clean_shooting_centerline.npy", result.out["centerline"])
    np.save(case_dir / "clean_shooting_state_y.npy", result.y)
    np.save(case_dir / "clean_residual_history.npy", np.asarray(result.residual_history, dtype=float))
    np.savez(
        case_dir / "clean_terminal_pose_data.npz",
        y=result.y,
        oracle_centerline=np.asarray(oracle["centerline"], dtype=float),
        shooting_centerline=result.out["centerline"],
        boundary=result.out["boundary"],
        tip_p=result.out["tip_p"],
        tip_q=result.out["tip_q"],
    )
    _plot_pose(case_dir, oracle, engine, result.out, "Clean terminal-pose shooting")
    _plot_history(case_dir, result.residual_history)

    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "M_list": [1, 1, 1],
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "steps_per_flex": int(args.steps_per_flex),
            "candidate_max_nfev": int(args.candidate_max_nfev),
            "terminal_convention": "distal end of the last rigid segment",
            "clean_residuals": ["terminal_position", "terminal_so3", "physical_boundary"],
            "excluded_residuals": ["centerline", "rigid_midpoint"],
            "catheter_length": length,
            "total_wall_s": float(time.perf_counter() - t0),
            "non_production": True,
        },
        "oracle": {
            "base_ok": bool(oracle["ok"]),
            "base_normE": float(oracle["stats"].normE),
            "base_wall_s": float(oracle["wall_s"]),
        },
        "rows": [row],
        "best_candidate": row,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", [row])
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
