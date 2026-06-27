"""Postprocess legacy fk.py gravity run into report-style evolution figures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from experiments.report_figures.generate_fk_321_report_figures import (
    en_font,
    plot_case_evolution,
    set_equal_3d_axes_mm,
    setup_plot_style,
    zh_font,
)


def initial_centerline(
    *,
    flex_lengths: tuple[float, ...] = (0.041, 0.03, 0.03),
    rigid_lengths: tuple[float, ...] = (0.003, 0.003, 0.003),
    m_list: tuple[int, ...] = (5, 5, 5),
    rigid_sample_count: int = 10,
) -> np.ndarray:
    points: list[np.ndarray] = []
    s = 0.0
    for lf, lr, m in zip(flex_lengths, rigid_lengths, m_list):
        flex_s = np.linspace(s, s + lf, int(m) + 1, dtype=np.float64)
        points.append(np.column_stack([flex_s, np.zeros_like(flex_s), np.zeros_like(flex_s)]))
        s = s + lf
        rigid_s = np.linspace(s, s + lr, int(rigid_sample_count) + 1, dtype=np.float64)[1:]
        points.append(np.column_stack([rigid_s, np.zeros_like(rigid_s), np.zeros_like(rigid_s)]))
        s = s + lr
    return np.concatenate(points, axis=0)


def centerline_from_z(
    z: np.ndarray,
    *,
    m_list: tuple[int, ...] = (5, 5, 5),
    l_ref: float = 0.11,
    rigid_sample_count: int = 10,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    idx = 0
    points: list[np.ndarray] = []
    for m in m_list:
        node_len = (int(m) + 1) * 13
        nodes = z[idx : idx + node_len].reshape((int(m) + 1, 13))
        idx += node_len
        idx += int(m) * 39
        x_rigid = z[idx : idx + 13]
        idx += 13

        p_flex = nodes[:, 0:3] * float(l_ref)
        points.append(p_flex)

        prox = p_flex[-1]
        dist = x_rigid[0:3] * float(l_ref)
        alphas = np.linspace(0.0, 1.0, int(rigid_sample_count) + 1, dtype=np.float64)[1:]
        points.append(prox[None, :] + alphas[:, None] * (dist - prox)[None, :])
    return np.concatenate(points, axis=0)


def parse_final_norm(log_text: str) -> float:
    matches = re.findall(r"\|\|E\|\|=([0-9.eE+-]+)", log_text)
    if matches:
        return float(matches[-1])
    return float("nan")


def build_case(run_dir: Path, case_dir: Path) -> dict[str, Any]:
    callback_dir = run_dir / "fk_py" / "callback_snaps"
    final_npz = run_dir / "fk_py" / "final_solution.npz"
    console_log = run_dir / "fk_py" / "console.log"
    if not final_npz.exists():
        raise FileNotFoundError(final_npz)
    if not callback_dir.exists():
        raise FileNotFoundError(callback_dir)

    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted(callback_dir.glob("callback_iter_*.npz"))
    checkpoints: list[tuple[int, float, np.ndarray]] = [(0, float("nan"), initial_centerline())]
    for path in checkpoint_paths:
        data = np.load(path)
        iteration = int(data["iter"])
        norm = float(data["normE"])
        checkpoints.append((iteration, norm, centerline_from_z(data["z"])))

    final_data = np.load(final_npz)
    log_text = console_log.read_text(encoding="utf-8", errors="replace") if console_log.exists() else ""
    final_iter = 20000
    iter_matches = re.findall(r"\[Multi-LM-JAX\]\s+iter=(\d+)", log_text)
    if iter_matches:
        final_iter = int(iter_matches[-1])
    final_norm = parse_final_norm(log_text)
    checkpoints.append((final_iter, final_norm, centerline_from_z(final_data["z"])))

    dedup: dict[int, tuple[float, np.ndarray]] = {}
    for iteration, norm, centerline in checkpoints:
        dedup[int(iteration)] = (float(norm), np.asarray(centerline, dtype=np.float64))

    iters = np.asarray(sorted(dedup), dtype=np.int32)
    normE = np.asarray([dedup[int(i)][0] for i in iters], dtype=np.float64)
    payload: dict[str, Any] = {"iters": iters, "normE": normE}
    for iteration in iters:
        payload[f"iter_{int(iteration):05d}"] = dedup[int(iteration)][1]
    np.savez_compressed(case_dir / "centerline_checkpoints_every20.npz", **payload)

    final_centerline = dedup[int(iters[-1])][1]
    np.save(case_dir / "centerline.npy", final_centerline)
    summary = {
        "case_name": case_dir.name,
        "condition_label": "gravity_only",
        "source_script": "forward_kinematics/fk.py",
        "tol": 1e-7,
        "max_iter": 20000,
        "ok": bool(final_data["ok"]) if "ok" in final_data.files else False,
        "n_iter": int(iters[-1]),
        "final_normE": float(normE[-1]),
        "checkpoint_count": int(len(iters)),
        "centerline_shape": list(final_centerline.shape),
        "centerline_end_m": final_centerline[-1].tolist(),
        "tip_sag_mm": float(-final_centerline[-1, 2] * 1000.0),
        "tip_x_mm": float(final_centerline[-1, 0] * 1000.0),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _magnet_points_and_directions(centerline_mm: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return three rigid magnet centers and local +z body directions.

    Centerline layout for M=(5,5,5), rigid_sample_count=10:
    flex1: 0..5, rigid1: 6..15; flex2: 16..21, rigid2: 22..31;
    flex3: 32..37, rigid3: 38..47.
    """
    rigid_pairs = [(5, 15), (21, 31), (37, 47)]
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for magnet_idx, (prox_idx, dist_idx) in enumerate(rigid_pairs, start=1):
        rigid_prox = centerline_mm[prox_idx]
        rigid_dist = centerline_mm[dist_idx]
        center = 0.5 * (rigid_prox + rigid_dist)
        direction = rigid_dist - rigid_prox
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm
        out.append((magnet_idx, center, direction))
    return out


def _gradient_3d_line(ax: Any, points: np.ndarray, values: np.ndarray, *, cmap: Any, lw: float, alpha: float) -> None:
    if len(points) < 2:
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(segments, cmap=cmap, linewidth=lw, alpha=alpha)
    collection.set_array(values[:-1])
    ax.add_collection3d(collection)


def plot_case_evolution_enhanced(case_dir: Path, out_path: Path) -> None:
    data = np.load(case_dir / "centerline_checkpoints_every20.npz")
    iters = np.asarray(data["iters"], dtype=np.int32)
    normE = np.asarray(data["normE"], dtype=np.float64)
    centerlines = [np.asarray(data[f"iter_{int(i):05d}"], dtype=np.float64) * 1000.0 for i in iters]
    all_points = np.concatenate(centerlines, axis=0)
    tips = np.asarray([line[-1] for line in centerlines], dtype=np.float64)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max()))
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    mid_count = len(centerlines) - 2
    keep_stride = max(1, int(np.ceil(max(mid_count, 1) / 55)))
    for idx, centerline in enumerate(centerlines):
        if idx == 0 or idx == len(centerlines) - 1:
            continue
        if idx % keep_stride != 0:
            continue
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            centerline[:, 2],
            color=cmap(norm(float(iters[idx]))),
            lw=0.62,
            alpha=0.16,
            zorder=1,
        )

    initial = centerlines[0]
    final = centerlines[-1]
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color="#3A3A3A", lw=2.2, label="初始位姿", zorder=4)
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color="#D55E00", lw=2.6, label="收敛位姿", zorder=5)

    _gradient_3d_line(ax, tips, iters.astype(np.float64), cmap=cmap, lw=2.1, alpha=0.98)
    key_ids = np.unique(np.linspace(0, len(tips) - 1, 8, dtype=int))
    ax.scatter(
        tips[key_ids, 0],
        tips[key_ids, 1],
        tips[key_ids, 2],
        s=18,
        color="black",
        edgecolors="white",
        linewidths=0.55,
        depthshade=False,
        label="末端轨迹",
        zorder=6,
    )

    mag_color = "#B2182B"
    text_offsets_initial = {
        1: np.array([1.8, -4.0, 5.5]),
        2: np.array([1.8, -4.0, 5.5]),
        3: np.array([1.8, -4.0, 5.5]),
    }
    text_offsets_final = {
        1: np.array([4.0, -4.0, 4.5]),
        2: np.array([4.0, -4.0, 4.5]),
        3: np.array([4.0, -4.0, 4.5]),
    }
    for line, state_name, offsets, alpha in [
        (initial, "initial", text_offsets_initial, 0.92),
        (final, "final", text_offsets_final, 1.0),
    ]:
        for magnet_idx, center, direction in _magnet_points_and_directions(line):
            label_point = "磁化点" if (state_name == "initial" and magnet_idx == 1) else None
            label_dir = "磁化方向" if (state_name == "initial" and magnet_idx == 1) else None
            ax.scatter(
                [center[0]],
                [center[1]],
                [center[2]],
                s=44,
                marker="o",
                facecolor="white",
                edgecolor=mag_color,
                linewidth=1.45,
                alpha=alpha,
                depthshade=False,
                label=label_point,
                zorder=8,
            )
            ax.quiver(
                center[0],
                center[1],
                center[2],
                direction[0],
                direction[1],
                direction[2],
                length=6.8,
                normalize=True,
                color=mag_color,
                alpha=alpha,
                linewidth=1.45,
                arrow_length_ratio=0.32,
                label=label_dir,
                zorder=9,
            )
            text_pos = center + offsets[magnet_idx]
            ax.text(
                text_pos[0],
                text_pos[1],
                text_pos[2],
                rf"$\mathbf{{m}}_{magnet_idx}$",
                color=mag_color,
                fontsize=8.6,
                fontproperties=en_font(8.6, "bold"),
                zorder=10,
            )

    ax.scatter(initial[0, 0], initial[0, 1], initial[0, 2], marker="^", s=40, color="black", depthshade=False)
    set_equal_3d_axes_mm(ax, all_points)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x (mm)", fontproperties=en_font(8.8))
    ax.set_ylabel("y (mm)", fontproperties=en_font(8.8))
    ax.set_zlabel("z (mm)", fontproperties=en_font(8.8))
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(en_font(7.8))
    ax.grid(True, color="0.86", linewidth=0.55)
    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.xaxis.pane.set_edgecolor("0.92")
    ax.yaxis.pane.set_edgecolor("0.92")
    ax.zaxis.pane.set_edgecolor("0.92")

    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        prop=zh_font(8.2),
        handlelength=2.3,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.045, aspect=22)
    cbar.set_label("累计迭代步", fontproperties=zh_font(8.2))
    cbar.outline.set_linewidth(0.55)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties=en_font(7.8)

    fig.text(
        0.14,
        0.035,
        f"最终残差 ||E||={float(normE[-1]):.2e}",
        fontproperties=zh_font(8.2),
        color="0.15",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--case_name", default="n1_tip_z_gravity_only_fk_py_tol1e7")
    args = parser.parse_args()
    setup_plot_style()
    run_dir = Path(args.run_dir)
    case_dir = run_dir / "cases" / args.case_name
    summary = build_case(run_dir, case_dir)
    plot_case_evolution(case_dir, case_dir / "fig_evolution", "n1_tip_z_gravity_only")
    plot_case_evolution_enhanced(case_dir, case_dir / "fig_evolution_enhanced")
    print(json.dumps({"case_dir": str(case_dir), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
