"""Plot IK benchmark trajectories and convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_centerline(run_dir: Path, rel_path: str | None) -> np.ndarray | None:
    if not rel_path:
        return None
    path = run_dir / rel_path
    if not path.exists():
        return None
    return np.load(path)


def _sample_paths(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    idx = np.unique(np.linspace(0, len(paths) - 1, max_count).astype(int))
    return [paths[int(i)] for i in idx]


def _equal_axes_3d(ax, arrays: Iterable[np.ndarray | None]) -> None:
    pts = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        pts.append(arr.reshape(-1, 3))
    if not pts:
        return
    all_pts = np.vstack(pts)
    mins = np.nanmin(all_pts, axis=0)
    maxs = np.nanmax(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.nanmax(maxs - mins)
    if not np.isfinite(radius) or radius <= 0:
        radius = 0.01
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _history_arrays(history: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer = history.get("outer", [])
    k = np.asarray([int(row.get("iter", i)) for i, row in enumerate(outer)], dtype=int)
    p_tip = np.asarray([row.get("p_tip_m", [np.nan, np.nan, np.nan]) for row in outer], dtype=float)
    err_mm = np.asarray([row.get("err_mm", np.nan) for row in outer], dtype=float)
    norm_e = np.asarray([row.get("normE", np.nan) for row in outer], dtype=float)
    accepted = np.asarray([bool(row.get("accept", False)) for row in outer], dtype=bool)
    return k, p_tip, err_mm, norm_e, accepted


def plot_run(run_dir: Path, out_dir: Path, *, max_trial_lines: int) -> None:
    """Plot one IK benchmark run.

    Args:
        run_dir: Directory containing history.json, run_config.json, and poses.
        out_dir: Directory where PNG files are written.
        max_trial_lines: Maximum number of rejected trial centerlines to draw.
    """
    cfg = _read_json(run_dir / "run_config.json")
    result = _read_json(run_dir / "result.json")
    history = _read_json(run_dir / "history.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_des = np.asarray(cfg.get("p_des") or result.get("p_des"), dtype=float).reshape(3)
    k, p_tip, err_mm, norm_e, accepted = _history_arrays(history)

    init_cl = _load_centerline(run_dir, cfg.get("init", {}).get("centerline_npy"))
    final_cl = _load_centerline(run_dir, cfg.get("final", {}).get("centerline_npy"))
    trial_paths = _sample_paths(sorted((run_dir / "poses" / "trial").glob("centerline_iter_*.npy")), max_trial_lines)
    trial_centerlines = [np.load(path) for path in trial_paths]

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    all_pts: list[np.ndarray | None] = [p_des.reshape(1, 3), p_tip, init_cl, final_cl]
    for idx, cl in enumerate(trial_centerlines):
        if cl.size == 0:
            continue
        color = cmap(idx / max(len(trial_centerlines) - 1, 1))
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], color=color, alpha=0.28, linewidth=0.8)
        all_pts.append(cl)
    if init_cl is not None and init_cl.size:
        ax.plot(init_cl[:, 0], init_cl[:, 1], init_cl[:, 2], color="tab:red", linewidth=2.2, label="init accepted")
        ax.scatter(init_cl[-1, 0], init_cl[-1, 1], init_cl[-1, 2], color="tab:red", s=36)
    if final_cl is not None and final_cl.size:
        ax.plot(final_cl[:, 0], final_cl[:, 1], final_cl[:, 2], color="tab:green", linewidth=2.2, label="final accepted")
        ax.scatter(final_cl[-1, 0], final_cl[-1, 1], final_cl[-1, 2], color="tab:green", s=36)
    if p_tip.size:
        ax.plot(p_tip[:, 0], p_tip[:, 1], p_tip[:, 2], color="black", linewidth=1.3, label="tip trajectory")
        ax.scatter(p_tip[0, 0], p_tip[0, 1], p_tip[0, 2], color="tab:red", s=44, marker="o", label="start tip")
        ax.scatter(p_tip[-1, 0], p_tip[-1, 1], p_tip[-1, 2], color="black", s=48, marker="x", label="last trial tip")
    ax.scatter(p_des[0], p_des[1], p_des[2], marker="*", color="tab:blue", s=110, label="target")
    ax.set_title(f"{run_dir.name}: centerlines, tips, and target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    _equal_axes_3d(ax, all_pts)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "centerline_trials_tip_target_3d.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    projections = [(0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"), (1, 2, "y (m)", "z (m)")]
    for ax, (a, b, xlabel, ylabel) in zip(axs, projections):
        ax.plot(p_tip[:, a], p_tip[:, b], "-o", markersize=2.4, linewidth=1.1, color="black", alpha=0.8, label="tip")
        ax.scatter(p_tip[0, a], p_tip[0, b], color="tab:red", s=42, label="start")
        ax.scatter(p_tip[-1, a], p_tip[-1, b], color="tab:green", s=42, marker="x", label="last")
        ax.scatter(p_des[a], p_des[b], color="tab:blue", marker="*", s=90, label="target")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{run_dir.name}: tip trajectory projections")
    fig.tight_layout()
    fig.savefig(out_dir / "tip_trajectory_projections.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
    ax1.plot(k, err_mm, color="tab:red", label="tip error (mm)")
    if np.any(accepted):
        ax1.scatter(k[accepted], err_mm[accepted], color="tab:green", s=28, label="accepted")
    if np.any(~accepted):
        ax1.scatter(k[~accepted], err_mm[~accepted], color="tab:gray", s=9, alpha=0.5, label="rejected")
    ax1.set_xlabel("IK iteration")
    ax1.set_ylabel("tip error (mm)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(k, norm_e, color="tab:blue", alpha=0.75, label="normE")
    ax2.set_ylabel("equilibrium residual normE")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "ts_error_normE_acceptance.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comprehensive IK benchmark diagnostics.")
    parser.add_argument("--run_dir", action="append", required=True, help="IK run directory. Can be repeated.")
    parser.add_argument("--out_root", required=True, help="Root directory for generated plots.")
    parser.add_argument("--max_trial_lines", type=int, default=80)
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    for run_dir_text in args.run_dir:
        run_dir = Path(run_dir_text).resolve()
        plot_run(run_dir, out_root / run_dir.name, max_trial_lines=int(args.max_trial_lines))
    print(f"[plot_ik_benchmark_results] wrote plots to: {out_root}")


if __name__ == "__main__":
    main()
