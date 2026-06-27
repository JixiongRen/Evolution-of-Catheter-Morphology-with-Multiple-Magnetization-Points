"""Plot FK evolution for true multi-magnet catheter configurations."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from experiments.report_figures.generate_fk_321_report_figures import (
    en_font,
    setup_plot_style,
    zh_font,
)


DEFAULT_BASE = REPO_ROOT / "experiment_records_fk_optimize" / "20260616_fk_true_multi_magnet_fkpy_tol1e6"
MAGNET_COLOR = "#B2182B"
INITIAL_COLOR = "#333333"
FINAL_COLOR = "#D55E00"
TIP_COLOR = "#111111"


def parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in str(text).split(",") if v.strip())


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(float(v.strip())) for v in str(text).split(",") if v.strip())


def parse_m_body_list(text: str) -> tuple[np.ndarray, ...]:
    rows: list[np.ndarray] = []
    for row in str(text).split(";"):
        row = row.strip()
        if row:
            rows.append(np.asarray(parse_csv_floats(row), dtype=np.float64))
    return tuple(rows)


def parse_final_norm(log_text: str) -> float:
    matches = re.findall(r"\|\|E\|\|=([0-9.eE+-]+)", log_text)
    return float(matches[-1]) if matches else float("nan")


def parse_final_iter(log_text: str, fallback: int) -> int:
    matches = re.findall(r"\[Multi-LM-JAX\]\s+iter=(\d+)", log_text)
    return int(matches[-1]) if matches else int(fallback)


def centerline_from_z(
    z: np.ndarray,
    *,
    m_list: tuple[int, ...],
    l_ref: float,
    rigid_sample_count: int = 14,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    idx = 0
    points: list[np.ndarray] = []
    rigid_pairs: list[tuple[int, int]] = []
    point_count = 0
    for m in m_list:
        node_len = (int(m) + 1) * 13
        nodes = z[idx : idx + node_len].reshape((int(m) + 1, 13))
        idx += node_len
        idx += int(m) * 39
        x_rigid = z[idx : idx + 13]
        idx += 13

        p_flex = nodes[:, 0:3] * float(l_ref)
        points.append(p_flex)
        prox_idx = point_count + len(p_flex) - 1
        point_count += len(p_flex)

        prox = p_flex[-1]
        dist = x_rigid[0:3] * float(l_ref)
        alphas = np.linspace(0.0, 1.0, int(rigid_sample_count) + 1, dtype=np.float64)[1:]
        rigid_points = prox[None, :] + alphas[:, None] * (dist - prox)[None, :]
        points.append(rigid_points)
        point_count += len(rigid_points)
        rigid_pairs.append((prox_idx, point_count - 1))
    return np.concatenate(points, axis=0), rigid_pairs


def initial_centerline_from_config(
    *,
    flex_lengths: tuple[float, ...],
    rigid_lengths: tuple[float, ...],
    m_list: tuple[int, ...],
    rigid_sample_count: int = 14,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    points: list[np.ndarray] = []
    rigid_pairs: list[tuple[int, int]] = []
    x0 = 0.0
    point_count = 0
    for lf, lr, m in zip(flex_lengths, rigid_lengths, m_list):
        flex_s = np.linspace(x0, x0 + float(lf), int(m) + 1, dtype=np.float64)
        p_flex = np.column_stack([flex_s, np.zeros_like(flex_s), np.zeros_like(flex_s)])
        points.append(p_flex)
        prox_idx = point_count + len(p_flex) - 1
        point_count += len(p_flex)
        x0 += float(lf)

        rigid_s = np.linspace(x0, x0 + float(lr), int(rigid_sample_count) + 1, dtype=np.float64)[1:]
        p_rigid = np.column_stack([rigid_s, np.zeros_like(rigid_s), np.zeros_like(rigid_s)])
        points.append(p_rigid)
        point_count += len(p_rigid)
        rigid_pairs.append((prox_idx, point_count - 1))
        x0 += float(lr)
    return np.concatenate(points, axis=0), rigid_pairs


def load_case(case_dir: Path) -> dict[str, Any]:
    config = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
    flex_lengths = parse_csv_floats(config["flex_lengths"])
    rigid_lengths = parse_csv_floats(config["rigid_lengths"])
    m_list = parse_csv_ints(config["M_list"])
    m_body_list = parse_m_body_list(config["m_body_list"])
    total_length = float(config["total_length_m"])

    callback_dir = case_dir / "callback_snaps"
    log_text = (case_dir / "console.log").read_text(encoding="utf-8", errors="replace")
    final_npz = np.load(case_dir / "final_solution.npz")

    checkpoints: list[tuple[int, float, np.ndarray, list[tuple[int, int]]]] = []
    initial_line, rigid_pairs = initial_centerline_from_config(
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
        m_list=m_list,
    )
    checkpoints.append((0, float("nan"), initial_line, rigid_pairs))

    for path in sorted(callback_dir.glob("callback_iter_*.npz")):
        data = np.load(path)
        line, pairs = centerline_from_z(data["z"], m_list=m_list, l_ref=total_length)
        checkpoints.append((int(data["iter"]), float(data["normE"]), line, pairs))

    final_iter = parse_final_iter(log_text, checkpoints[-1][0])
    final_norm = parse_final_norm(log_text)
    final_line, final_pairs = centerline_from_z(final_npz["z"], m_list=m_list, l_ref=total_length)
    checkpoints.append((final_iter, final_norm, final_line, final_pairs))

    dedup: dict[int, tuple[float, np.ndarray, list[tuple[int, int]]]] = {}
    for iteration, norm_e, line, pairs in checkpoints:
        dedup[int(iteration)] = (float(norm_e), np.asarray(line, dtype=np.float64), pairs)

    iters = np.asarray(sorted(dedup), dtype=np.int32)
    centerlines = [dedup[int(i)][1] for i in iters]
    norm_e = np.asarray([dedup[int(i)][0] for i in iters], dtype=np.float64)
    return {
        "case_name": case_dir.name,
        "label": str(config["label"]),
        "m_body_list": m_body_list,
        "iters": iters,
        "normE": norm_e,
        "centerlines_m": centerlines,
        "initial_pairs": dedup[int(iters[0])][2],
        "final_pairs": dedup[int(iters[-1])][2],
        "final_norm": float(norm_e[-1]),
        "converged": "Converged: True" in log_text,
    }


def set_equal_3d_axes(ax: Any, points_mm: np.ndarray, *, pad: float = 9.0) -> None:
    mins = np.min(points_mm, axis=0)
    maxs = np.max(points_mm, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) + float(pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def magnet_center_direction(line_mm: np.ndarray, pair: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    prox = line_mm[pair[0]]
    dist = line_mm[pair[1]]
    center = 0.5 * (prox + dist)
    direction = dist - prox
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return center, direction


def add_magnets(
    ax: Any,
    line_mm: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    annotate: bool,
    state: str,
) -> None:
    if state == "initial":
        offsets = [
            np.array([2.8, 2.2, 4.4]),
            np.array([2.8, 2.2, 4.4]),
            np.array([2.8, 2.2, 4.4]),
        ]
    else:
        offsets = [
            np.array([2.4, -5.2, -3.2]),
            np.array([2.4, -5.2, -3.2]),
            np.array([2.4, -5.2, -3.2]),
        ]
    for idx, pair in enumerate(pairs, start=1):
        center, direction = magnet_center_direction(line_mm, pair)
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            s=62,
            facecolor="white",
            edgecolor=MAGNET_COLOR,
            linewidth=1.6,
            depthshade=False,
            label="磁化点" if idx == 1 and annotate else None,
            zorder=9,
        )
        ax.quiver(
            center[0],
            center[1],
            center[2],
            direction[0],
            direction[1],
            direction[2],
            length=8.0,
            normalize=True,
            color=MAGNET_COLOR,
            linewidth=1.35,
            arrow_length_ratio=0.32,
            label="磁化方向" if idx == 1 and annotate else None,
            zorder=10,
        )
        if annotate:
            text_pos = center + offsets[min(idx - 1, len(offsets) - 1)]
            ax.text(
                text_pos[0],
                text_pos[1],
                text_pos[2],
                rf"$\mathbf{{m}}_{idx}$",
                color=MAGNET_COLOR,
                fontproperties=en_font(11.0, "bold"),
                zorder=11,
            )


def add_tip_path(ax: Any, centerlines_mm: list[np.ndarray], iters: np.ndarray) -> None:
    tips = np.asarray([line[-1] for line in centerlines_mm], dtype=np.float64)
    if len(tips) > 1:
        segments = np.stack([tips[:-1], tips[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=plt.get_cmap("viridis"), linewidth=2.0, alpha=0.98)
        lc.set_array(iters[:-1].astype(np.float64))
        ax.add_collection3d(lc)
    key_ids = np.unique(np.linspace(0, len(tips) - 1, min(9, len(tips)), dtype=int))
    ax.scatter(
        tips[key_ids, 0],
        tips[key_ids, 1],
        tips[key_ids, 2],
        s=18,
        color=TIP_COLOR,
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
        label="末端轨迹",
        zorder=8,
    )


def plot_case(case: dict[str, Any], out_base: Path) -> None:
    centerlines_mm = [line * 1000.0 for line in case["centerlines_m"]]
    iters = np.asarray(case["iters"], dtype=np.int32)
    all_points = np.concatenate(centerlines_mm, axis=0)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max()))

    fig = plt.figure(figsize=(7.2, 6.1))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    mid_count = len(centerlines_mm) - 2
    stride = max(1, int(np.ceil(max(mid_count, 1) / 65)))
    for idx, line in enumerate(centerlines_mm[1:-1], start=1):
        if idx % stride != 0:
            continue
        ax.plot(
            line[:, 0],
            line[:, 1],
            line[:, 2],
            color=cmap(norm(float(iters[idx]))),
            lw=0.65,
            alpha=0.16,
            zorder=1,
        )

    initial = centerlines_mm[0]
    final = centerlines_mm[-1]
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color=INITIAL_COLOR, lw=2.4, label="初始位姿", zorder=4)
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color=FINAL_COLOR, lw=3.0, label="收敛位姿", zorder=5)
    add_tip_path(ax, centerlines_mm, iters)
    add_magnets(ax, initial, case["initial_pairs"], annotate=True, state="initial")
    add_magnets(ax, final, case["final_pairs"], annotate=True, state="final")

    ax.scatter(initial[0, 0], initial[0, 1], initial[0, 2], marker="^", s=56, color="black", depthshade=False)
    set_equal_3d_axes(ax, all_points, pad=9.0)
    ax.view_init(elev=20, azim=-56)
    ax.set_xlabel("x (mm)", fontproperties=en_font(11.0))
    ax.set_ylabel("y (mm)", fontproperties=en_font(11.0))
    ax.set_zlabel("z (mm)", fontproperties=en_font(11.0))
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(en_font(9.2))
    ax.grid(True, color="0.84", linewidth=0.65)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        pane.set_edgecolor("0.90")

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
        prop=zh_font(10.0),
        handlelength=2.1,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.085, aspect=20)
    cbar.set_label("累计迭代步", fontproperties=zh_font(10.0))
    cbar.outline.set_linewidth(0.65)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(en_font(9.0))

    status = "收敛" if case["converged"] else "未达容差"
    fig.text(
        0.08,
        0.035,
        f"{case['label']}；{status}；最终残差 $||E||$={case['final_norm']:.2e}",
        fontproperties=zh_font(10.2),
        color="0.18",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=360)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    setup_plot_style()
    base_dir = args.base_dir.resolve()
    out_dir = (args.out_dir or (base_dir / "report_style_figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for case_name in ["n2_z_pair_rand_a", "n2_xz_pair_rand_b", "n3_xyz_rand_a"]:
        case = load_case(base_dir / case_name)
        plot_case(case, out_dir / f"{case_name}_evolution_3d")
        summary.append(
            {
                "case_name": case["case_name"],
                "label": case["label"],
                "final_norm": case["final_norm"],
                "converged": case["converged"],
                "output_base": str(out_dir / f"{case_name}_evolution_3d"),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "cases": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
