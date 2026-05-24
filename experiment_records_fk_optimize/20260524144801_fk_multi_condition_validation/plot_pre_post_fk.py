"""Plot pre- and post-optimization FK centerline comparisons."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PREV_DIR = REPO_ROOT / "experiment_records_fk_optimize" / "20260524114754_fk_strict_solver_algorithm_opt"


SOURCES = [
    {
        "label": "old_cli_callback",
        "path": REPO_ROOT
        / "docs"
        / "efficiency_optimization_experiment"
        / "eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data"
        / "old_fk_tol1e-5"
        / "centerline_last_callback.npy",
        "color": "tab:red",
        "style": "--",
    },
    {
        "label": "pre_algorithm_baseline",
        "path": PREV_DIR / "fk_driver_lm03_solve_strict_baseline" / "centerline.npy",
        "color": "tab:blue",
        "style": "-.",
    },
    {
        "label": "post_optimized_candidate",
        "path": PREV_DIR / "fk_pos_lm01_shortstage_final1450_production" / "centerline.npy",
        "color": "tab:green",
        "style": "-",
    },
]


def load_sources() -> dict[str, np.ndarray]:
    data = {}
    for item in SOURCES:
        path = Path(item["path"])
        if path.exists():
            data[item["label"]] = np.load(path)
    return data


def pointwise_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float | int]:
    n = min(len(ref), len(cand))
    delta = np.linalg.norm(ref[:n] - cand[:n], axis=1)
    return {
        "point_count": int(n),
        "max_point_error_m": float(np.max(delta)),
        "rms_point_error_m": float(np.sqrt(np.mean(delta * delta))),
        "end_error_m": float(delta[-1]),
        "max_rel_to_L_ref": float(np.max(delta) / 0.11),
    }


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    data = load_sources()
    if "post_optimized_candidate" not in data:
        raise FileNotFoundError("Missing optimized candidate centerline")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    all_points = []
    for item in SOURCES:
        label = item["label"]
        if label not in data:
            continue
        pts = data[label]
        all_points.append(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], item["style"], color=item["color"], linewidth=2.2, label=label)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=item["color"], s=35)
    set_equal_axes(ax, np.concatenate(all_points, axis=0))
    ax.set_title("FK centerline comparison: pre vs post optimization")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pre_post_fk_centerline_overlay.png", dpi=180)
    plt.close(fig)

    metrics = []
    candidate = data["post_optimized_candidate"]
    for ref_name in ("pre_algorithm_baseline", "old_cli_callback"):
        if ref_name not in data:
            continue
        row = {"reference": ref_name, "candidate": "post_optimized_candidate"}
        row.update(pointwise_metrics(data[ref_name], candidate))
        metrics.append(row)

    with (OUT_DIR / "pre_post_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    (OUT_DIR / "pre_post_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    for ref_name in ("pre_algorithm_baseline", "old_cli_callback"):
        if ref_name not in data:
            continue
        n = min(len(data[ref_name]), len(candidate))
        delta = np.linalg.norm(data[ref_name][:n] - candidate[:n], axis=1)
        ax.plot(np.arange(n), delta, marker="o", linewidth=1.6, label=f"{ref_name} error")
    ax.set_title("Pointwise centerline error")
    ax.set_xlabel("point index")
    ax.set_ylabel("error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pre_post_pointwise_error.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
