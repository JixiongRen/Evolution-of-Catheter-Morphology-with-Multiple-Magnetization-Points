"""Visualize single-terminal-magnet FK cases with calibrated coil poses."""

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
FK_DIR = REPO_ROOT / "forward_kinematics"
if str(FK_DIR) not in sys.path:
    sys.path.insert(0, str(FK_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from experiments.report_figures.generate_fk_321_report_figures import (
    en_font,
    setup_plot_style,
    zh_font,
)
from forward_kinematics.mas_nondim import MagneticActuationSystem
from supiee_auto_diff.actuation_interpolator_jax import interpolate_A_vmap


DEFAULT_BASE = (
    REPO_ROOT
    / "experiment_records_fk_optimize"
    / "20260615_fk_single_segment_L50mm_coils1_4_15A_tol1e6"
)
DEFAULT_CALIB = REPO_ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
DEFAULT_TABLE = (
    REPO_ROOT
    / "supiee_auto_diff"
    / "offline_interpolation_data"
    / "actuation_tables"
    / "actuation_table.pkl"
)

ACTIVE_RED = "#C7222A"
INACTIVE_BLUE = "#2767B1"
INITIAL_COLOR = "#343434"
FINAL_COLOR = "#D55E00"
FIELD_COLOR = "#009E73"
ZOOM_FIELD_COLOR = "#0B3D91"
TIP_COLOR = "#111111"


def _parse_currents(text: str) -> np.ndarray:
    return np.asarray([float(v.strip()) for v in text.split(",") if v.strip()], dtype=np.float64)


def _parse_final_norm(log_text: str) -> float:
    matches = re.findall(r"\|\|E\|\|=([0-9.eE+-]+)", log_text)
    return float(matches[-1]) if matches else float("nan")


def _parse_final_iter(log_text: str, fallback: int) -> int:
    matches = re.findall(r"\[Multi-LM-JAX\]\s+iter=(\d+)", log_text)
    return int(matches[-1]) if matches else int(fallback)


def centerline_from_z(
    z: np.ndarray,
    *,
    m_list: tuple[int, ...] = (12,),
    l_ref: float = 0.05,
    rigid_sample_count: int = 18,
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


def load_case(case_dir: Path) -> dict[str, Any]:
    final_npz = case_dir / "final_solution.npz"
    callback_dir = case_dir / "callback_snaps"
    config_path = case_dir / "case_config.json"
    console_log = case_dir / "console.log"
    if not final_npz.exists():
        raise FileNotFoundError(final_npz)
    if not callback_dir.exists():
        raise FileNotFoundError(callback_dir)

    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    currents = _parse_currents(config.get("coil_currents", "0,0,0,0,0,0,0,0"))
    active_idx = int(np.argmax(np.abs(currents))) if np.any(np.abs(currents) > 0) else -1
    log_text = console_log.read_text(encoding="utf-8", errors="replace") if console_log.exists() else ""

    checkpoints: list[tuple[int, float, np.ndarray]] = []
    for path in sorted(callback_dir.glob("callback_iter_*.npz")):
        data = np.load(path)
        checkpoints.append(
            (
                int(data["iter"]),
                float(data["normE"]),
                centerline_from_z(data["z"]),
            )
        )

    final_data = np.load(final_npz)
    final_iter = _parse_final_iter(log_text, checkpoints[-1][0] if checkpoints else 0)
    final_norm = _parse_final_norm(log_text)
    checkpoints.append((final_iter, final_norm, centerline_from_z(final_data["z"])))

    dedup: dict[int, tuple[float, np.ndarray]] = {}
    for iteration, norm_e, centerline in checkpoints:
        dedup[int(iteration)] = (float(norm_e), np.asarray(centerline, dtype=np.float64))

    iters = np.asarray(sorted(dedup), dtype=np.int32)
    centerlines = [dedup[int(i)][1] for i in iters]
    norms = np.asarray([dedup[int(i)][0] for i in iters], dtype=np.float64)
    return {
        "name": case_dir.name,
        "config": config,
        "currents": currents,
        "active_idx": active_idx,
        "iters": iters,
        "normE": norms,
        "centerlines_m": centerlines,
        "final_norm": float(norms[-1]),
    }


def load_coils(calib_file: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(calib_file.read_text(encoding="utf-8"))
    coils: list[dict[str, Any]] = []
    for idx, coil_name in enumerate(data["Coil_List"]):
        src_name = data[coil_name]["Source_List"][0]
        src = data[coil_name][src_name]
        pos = np.asarray(src["Source_Position"], dtype=np.float64)
        direction = np.asarray(src["Source_Direction"], dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        coils.append({"idx": idx, "name": coil_name, "position_m": pos, "direction": direction})
    return coils


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, ref))) > 0.88:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(axis, u)
    return u, v


def cylinder_mesh_mm(
    center_mm: np.ndarray,
    axis: np.ndarray,
    *,
    radius_mm: float = 18.0,
    length_mm: float = 92.0,
    n_theta: int = 28,
    n_len: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    u, v = _orthonormal_basis(axis)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    t = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_len)
    theta_grid, t_grid = np.meshgrid(theta, t)
    pts = (
        center_mm[None, None, :]
        + t_grid[..., None] * axis[None, None, :]
        + radius_mm
        * (
            np.cos(theta_grid)[..., None] * u[None, None, :]
            + np.sin(theta_grid)[..., None] * v[None, None, :]
        )
    )
    return pts[..., 0], pts[..., 1], pts[..., 2]


def magnet_center_direction(centerline_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prox = centerline_mm[12]
    dist = centerline_mm[-1]
    center = 0.5 * (prox + dist)
    direction = dist - prox
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return center, direction


def field_samples(mas: MagneticActuationSystem, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.006, 0.052, 4)
    ys = np.linspace(-0.018, 0.018, 3)
    zs = np.linspace(-0.020, 0.020, 3)
    pts = np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    vecs = []
    for p in pts:
        vecs.append(np.asarray(mas.b_field(p, currents), dtype=np.float64))
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1)
    keep = norms > np.percentile(norms, 18)
    pts = pts[keep]
    vecs = vecs[keep]
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return pts * 1000.0, vecs


def catheter_local_plane_basis(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = case["centerlines_m"][0] * 1000.0
    final = case["centerlines_m"][-1] * 1000.0
    origin = final[0].copy()
    e1 = initial[-1] - initial[0]
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)

    bend = final[-1] - final[0]
    e2 = bend - float(np.dot(bend, e1)) * e1
    if np.linalg.norm(e2) < 1e-9:
        centered = final - np.mean(final, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        e2 = vh[1] - float(np.dot(vh[1], e1)) * e1
    if np.linalg.norm(e2) < 1e-9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(ref, e1))) > 0.88:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e2 = ref - float(np.dot(ref, e1)) * e1
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-12)
    return origin, e1, e2


def local_plane_field_samples(
    mas: MagneticActuationSystem,
    case: dict[str, Any],
    *,
    grid_u: int = 6,
    grid_v: int = 5,
    pad_u: float = 4.0,
    pad_v: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    final = case["centerlines_m"][-1] * 1000.0
    origin, e1, e2 = catheter_local_plane_basis(case)
    rel = final - origin[None, :]
    u = rel @ e1
    v = rel @ e2
    us = np.linspace(float(u.min() - pad_u), float(u.max() + pad_u), int(grid_u))
    vs = np.linspace(float(v.min() - pad_v), float(v.max() + pad_v), int(grid_v))
    uu, vv = np.meshgrid(us, vs)
    pts_mm = origin[None, :] + uu.reshape(-1, 1) * e1[None, :] + vv.reshape(-1, 1) * e2[None, :]

    vecs = []
    for p_mm in pts_mm:
        b = np.asarray(mas.b_field(p_mm / 1000.0, case["currents"]), dtype=np.float64)
        b_plane = float(np.dot(b, e1)) * e1 + float(np.dot(b, e2)) * e2
        if np.linalg.norm(b_plane) < 1e-14:
            b_plane = b
        vecs.append(b_plane)
    vecs = np.asarray(vecs, dtype=np.float64)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return pts_mm, vecs


def local_plane_view_angles(case: dict[str, Any]) -> tuple[float, float]:
    _, e1, e2 = catheter_local_plane_basis(case)
    normal = np.cross(e1, e2)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    if normal[1] > 0.0:
        normal = -normal
    azim = float(np.degrees(np.arctan2(normal[1], normal[0])))
    elev = float(np.degrees(np.arcsin(np.clip(normal[2], -1.0, 1.0))))
    return elev, azim


def add_coils_3d(ax: Any, coils: list[dict[str, Any]], active_idx: int) -> None:
    for coil in coils:
        center_mm = coil["position_m"] * 1000.0
        color = ACTIVE_RED if coil["idx"] == active_idx else INACTIVE_BLUE
        x, y, z = cylinder_mesh_mm(center_mm, coil["direction"])
        ax.plot_surface(x, y, z, color=color, alpha=0.78, linewidth=0.0, shade=True, zorder=1)
        tip = center_mm + coil["direction"] * 58.0
        ax.plot(
            [center_mm[0], tip[0]],
            [center_mm[1], tip[1]],
            [center_mm[2], tip[2]],
            color="white",
            lw=1.0,
            alpha=0.82,
            zorder=2,
        )
        ax.text(
            center_mm[0],
            center_mm[1],
            center_mm[2] + 34.0,
            f"C{coil['idx'] + 1}",
            color=color,
            fontproperties=en_font(7.5, "bold"),
            ha="center",
            va="center",
            zorder=3,
        )


def add_catheter_3d(
    ax: Any,
    case: dict[str, Any],
    *,
    show_intermediate: bool = True,
    annotation_size: float = 7.6,
    main_lw_scale: float = 1.0,
) -> None:
    centerlines = [line * 1000.0 for line in case["centerlines_m"]]
    iters = np.asarray(case["iters"], dtype=np.float64)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max()))
    if show_intermediate:
        mid_count = len(centerlines) - 2
        stride = max(1, int(np.ceil(max(mid_count, 1) / 44)))
        for idx, line in enumerate(centerlines[1:-1], start=1):
            if idx % stride != 0:
                continue
            ax.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color=cmap(norm(float(iters[idx]))),
                lw=0.72 * main_lw_scale,
                alpha=0.18,
                zorder=4,
            )
    initial = centerlines[0]
    final = centerlines[-1]
    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], color=INITIAL_COLOR, lw=2.35 * main_lw_scale, label="初始位姿", zorder=5)
    ax.plot(final[:, 0], final[:, 1], final[:, 2], color=FINAL_COLOR, lw=2.8 * main_lw_scale, label="收敛位姿", zorder=6)

    tips = np.asarray([line[-1] for line in centerlines], dtype=np.float64)
    segments = np.stack([tips[:-1], tips[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, linewidth=2.0 * main_lw_scale, alpha=0.95)
    lc.set_array(iters[:-1])
    ax.add_collection3d(lc)
    key_ids = np.unique(np.linspace(0, len(tips) - 1, 7, dtype=int))
    ax.scatter(
        tips[key_ids, 0],
        tips[key_ids, 1],
        tips[key_ids, 2],
            s=16 * main_lw_scale,
        color=TIP_COLOR,
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
        label="末端轨迹",
        zorder=8,
    )

    for line, alpha, label_suffix in [(initial, 0.86, "初始"), (final, 1.0, "收敛")]:
        center, direction = magnet_center_direction(line)
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            s=54 * main_lw_scale,
            marker="o",
            facecolor="white",
            edgecolor="#B2182B",
            linewidth=1.45 * main_lw_scale,
            alpha=alpha,
            depthshade=False,
            zorder=9,
        )
        ax.quiver(
            center[0],
            center[1],
            center[2],
            direction[0],
            direction[1],
            direction[2],
            length=8.0 * main_lw_scale,
            normalize=True,
            color="#B2182B",
            linewidth=1.35 * main_lw_scale,
            arrow_length_ratio=0.32,
            alpha=alpha,
            zorder=10,
        )
        text_offset = np.array([4.2, 3.4, 4.2]) if label_suffix == "初始" else np.array([3.0, -5.2, 3.2])
        ax.text(
            center[0] + float(text_offset[0]),
            center[1] + float(text_offset[1]),
            center[2] + float(text_offset[2]),
            rf"$\mathbf{{m}}$({label_suffix})",
            color="#B2182B",
            fontproperties=zh_font(annotation_size),
            zorder=11,
        )


def add_field_3d(
    ax: Any,
    field_pts_mm: np.ndarray,
    field_dirs: np.ndarray,
    *,
    length: float = 23.0,
    color: str = FIELD_COLOR,
    alpha: float = 0.34,
    linewidth: float = 0.7,
    arrow_length_ratio: float = 0.25,
) -> None:
    ax.quiver(
        field_pts_mm[:, 0],
        field_pts_mm[:, 1],
        field_pts_mm[:, 2],
        field_dirs[:, 0],
        field_dirs[:, 1],
        field_dirs[:, 2],
        length=length,
        normalize=True,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        arrow_length_ratio=arrow_length_ratio,
        zorder=3,
    )


def set_equal_3d_limits(ax: Any, points_mm: np.ndarray, *, pad_mm: float = 80.0) -> None:
    mins = np.min(points_mm, axis=0)
    maxs = np.max(points_mm, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) + pad_mm
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def style_3d_axis(ax: Any, *, label_size: float = 8.2, tick_size: float = 7.2, grid_width: float = 0.55) -> None:
    ax.set_xlabel("x (mm)", fontproperties=en_font(label_size), labelpad=9)
    ax.set_ylabel("y (mm)", fontproperties=en_font(label_size), labelpad=9)
    ax.set_zlabel("z (mm)", fontproperties=en_font(label_size), labelpad=9)
    ax.grid(True, color="0.86", linewidth=0.55)
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(en_font(tick_size))
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        pane.set_edgecolor("0.92")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = grid_width


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=360)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))


def plot_3d_case(
    case: dict[str, Any],
    coils: list[dict[str, Any]],
    field_pts_mm: np.ndarray,
    field_dirs: np.ndarray,
    out_base: Path,
) -> None:
    all_catheter = np.concatenate([line * 1000.0 for line in case["centerlines_m"]], axis=0)
    coil_centers = np.asarray([coil["position_m"] * 1000.0 for coil in coils], dtype=np.float64)
    all_full = np.vstack([coil_centers, all_catheter, field_pts_mm])

    fig = plt.figure(figsize=(10.6, 5.2))
    ax_full = fig.add_subplot(1, 2, 1, projection="3d")
    ax_zoom = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, zoom in [(ax_full, False), (ax_zoom, True)]:
        if not zoom:
            add_coils_3d(ax, coils, int(case["active_idx"]))
        add_field_3d(ax, field_pts_mm, field_dirs, length=23.0 if not zoom else 8.0)
        add_catheter_3d(ax, case, show_intermediate=True)
        style_3d_axis(ax)
        ax.view_init(elev=24, azim=-52)
        if zoom:
            set_equal_3d_limits(ax, np.vstack([all_catheter, field_pts_mm]), pad_mm=12.0)
            ax.text2D(0.02, 0.96, "中心工作区放大", transform=ax.transAxes, fontproperties=zh_font(8.4, "bold"))
        else:
            set_equal_3d_limits(ax, all_full, pad_mm=70.0)
            ax.text2D(0.02, 0.96, "系统相对位姿", transform=ax.transAxes, fontproperties=zh_font(8.4, "bold"))

    active_label = f"C{int(case['active_idx']) + 1}"
    fig.text(
        0.062,
        0.025,
        f"{active_label}: 15 A；最终残差 $||E||$={case['final_norm']:.2e}",
        fontproperties=zh_font(8.2),
        color="0.18",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, out_base)
    plt.close(fig)


def project(points: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
    if plane == "xy":
        return points[:, 0], points[:, 1]
    if plane == "xz":
        return points[:, 0], points[:, 2]
    if plane == "yz":
        return points[:, 1], points[:, 2]
    raise ValueError(plane)


def draw_projected_cylinder(
    ax: plt.Axes,
    coil: dict[str, Any],
    *,
    active_idx: int,
    plane: str,
    show_label: bool = True,
) -> None:
    center = coil["position_m"] * 1000.0
    axis = coil["direction"]
    color = ACTIVE_RED if coil["idx"] == active_idx else INACTIVE_BLUE
    p1 = center - axis * 46.0
    p2 = center + axis * 46.0
    x, y = project(np.vstack([p1, p2]), plane)
    ax.plot(x, y, color=color, lw=8.0, alpha=0.70, solid_capstyle="round")
    ax.plot(x, y, color="white", lw=1.1, alpha=0.88, solid_capstyle="round")
    xc, yc = project(center[None, :], plane)
    if show_label:
        ax.text(
            xc[0],
            yc[0] + 18.0,
            f"C{coil['idx'] + 1}",
            color=color,
            fontproperties=en_font(21.0, "bold"),
            ha="center",
            va="center",
        )


def _scale_line_for_global_view(line: np.ndarray, scale: float) -> np.ndarray:
    anchor = line[0].copy()
    return anchor[None, :] + scale * (line - anchor[None, :])


def add_catheter_2d_global_scaled(ax: plt.Axes, case: dict[str, Any], plane: str, *, scale: float = 7.0) -> None:
    centerlines = [_scale_line_for_global_view(line * 1000.0, scale) for line in case["centerlines_m"]]
    iters = np.asarray(case["iters"], dtype=np.float64)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max()))
    stride = max(1, int(np.ceil(max(len(centerlines) - 2, 1) / 30)))
    for idx, line in enumerate(centerlines[1:-1], start=1):
        if idx % stride != 0:
            continue
        x, y = project(line, plane)
        ax.plot(x, y, color=cmap(norm(float(iters[idx]))), lw=0.92, alpha=0.16)
    for line, color, lw in [
        (centerlines[0], INITIAL_COLOR, 2.2),
        (centerlines[-1], FINAL_COLOR, 2.8),
    ]:
        x, y = project(line, plane)
        ax.plot(x, y, color=color, lw=lw)

    tips = np.asarray([line[-1] for line in centerlines], dtype=np.float64)
    x_tip, y_tip = project(tips, plane)
    ax.plot(x_tip, y_tip, color=TIP_COLOR, lw=1.5, alpha=0.72)

    for line in [centerlines[0], centerlines[-1]]:
        center, direction = magnet_center_direction(line)
        cxy = np.asarray(project(center[None, :], plane)).reshape(2)
        dxy = np.asarray(project(direction[None, :], plane)).reshape(2)
        dxy = dxy / max(float(np.linalg.norm(dxy)), 1e-12)
        ax.scatter([cxy[0]], [cxy[1]], s=38, facecolor="white", edgecolor="#B2182B", linewidth=1.1, zorder=7)
        ax.arrow(
            cxy[0],
            cxy[1],
            dxy[0] * 12.0,
            dxy[1] * 12.0,
            width=0.72,
            head_width=4.0,
            head_length=5.0,
            length_includes_head=True,
            color="#B2182B",
            alpha=0.92,
            zorder=8,
        )


def add_catheter_2d(
    ax: plt.Axes,
    case: dict[str, Any],
    plane: str,
    *,
    annotate_magnet: bool = True,
    main_lw_scale: float = 1.0,
    tip_lw_scale: float = 1.0,
) -> None:
    centerlines = [line * 1000.0 for line in case["centerlines_m"]]
    iters = np.asarray(case["iters"], dtype=np.float64)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max()))
    stride = max(1, int(np.ceil(max(len(centerlines) - 2, 1) / 48)))
    for idx, line in enumerate(centerlines[1:-1], start=1):
        if idx % stride != 0:
            continue
        x, y = project(line, plane)
        ax.plot(x, y, color=cmap(norm(float(iters[idx]))), lw=0.62, alpha=0.18)
    for line, color, lw, label in [
        (centerlines[0], INITIAL_COLOR, 1.9, "初始位姿"),
        (centerlines[-1], FINAL_COLOR, 2.25, "收敛位姿"),
    ]:
        x, y = project(line, plane)
        ax.plot(x, y, color=color, lw=lw * main_lw_scale, label=label)

    tips = np.asarray([line[-1] for line in centerlines], dtype=np.float64)
    x_tip, y_tip = project(tips, plane)
    ax.plot(x_tip, y_tip, color=TIP_COLOR, lw=1.25 * tip_lw_scale, alpha=0.78, label="末端轨迹")
    key_ids = np.unique(np.linspace(0, len(tips) - 1, 7, dtype=int))
    ax.scatter(x_tip[key_ids], y_tip[key_ids], s=12, color=TIP_COLOR, edgecolors="white", linewidths=0.3, zorder=6)

    for line, label_suffix in [(centerlines[0], "初始"), (centerlines[-1], "收敛")]:
        center, direction = magnet_center_direction(line)
        cxy = np.asarray(project(center[None, :], plane)).reshape(2)
        dxy = np.asarray(project(direction[None, :], plane)).reshape(2)
        norm_d = max(float(np.linalg.norm(dxy)), 1e-12)
        dxy = dxy / norm_d
        ax.scatter([cxy[0]], [cxy[1]], s=38, facecolor="white", edgecolor="#B2182B", linewidth=1.1, zorder=7)
        ax.arrow(
            cxy[0],
            cxy[1],
            dxy[0] * 5.0,
            dxy[1] * 5.0,
            width=0.35,
            head_width=1.8,
            head_length=2.2,
            length_includes_head=True,
            color="#B2182B",
            alpha=0.92,
            zorder=8,
        )
        if annotate_magnet:
            text_offsets = {
                "xy": {"初始": np.array([2.8, -4.6]), "收敛": np.array([2.8, 2.9])},
                "xz": {"初始": np.array([2.8, -5.0]), "收敛": np.array([2.8, 2.7])},
                "yz": {"初始": np.array([3.0, -5.0]), "收敛": np.array([3.0, 2.8])},
            }
            offset = text_offsets.get(plane, {}).get(label_suffix, np.array([3.0, 3.0]))
            ax.text(
                cxy[0] + float(offset[0]),
                cxy[1] + float(offset[1]),
                rf"$\mathbf{{m}}$({label_suffix})",
                color="#B2182B",
                fontproperties=zh_font(20.4),
            )


def add_field_2d(ax: plt.Axes, field_pts_mm: np.ndarray, field_dirs: np.ndarray, plane: str) -> None:
    pts2 = np.asarray(project(field_pts_mm, plane)).T
    dirs2 = np.asarray(project(field_dirs, plane)).T
    norms = np.linalg.norm(dirs2, axis=1)
    keep = norms > 0.18
    pts2 = pts2[keep]
    dirs2 = dirs2[keep] / np.maximum(norms[keep, None], 1e-12)
    ax.quiver(
        pts2[:, 0],
        pts2[:, 1],
        dirs2[:, 0],
        dirs2[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.11,
        width=0.0025,
        color=FIELD_COLOR,
        alpha=0.30,
        zorder=2,
    )


def point_from_plane(a_mm: float, b_mm: float, plane: str, anchor_mm: np.ndarray) -> np.ndarray:
    if plane == "xy":
        return np.array([a_mm, b_mm, anchor_mm[2]], dtype=np.float64)
    if plane == "xz":
        return np.array([a_mm, anchor_mm[1], b_mm], dtype=np.float64)
    if plane == "yz":
        return np.array([anchor_mm[0], a_mm, b_mm], dtype=np.float64)
    raise ValueError(plane)


def project_vector(vec: np.ndarray, plane: str) -> tuple[float, float]:
    if plane == "xy":
        return float(vec[0]), float(vec[1])
    if plane == "xz":
        return float(vec[0]), float(vec[2])
    if plane == "yz":
        return float(vec[1]), float(vec[2])
    raise ValueError(plane)


def add_dense_field_2d(
    ax: plt.Axes,
    mas: MagneticActuationSystem,
    currents: np.ndarray,
    plane: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    anchor_mm: np.ndarray,
    grid_n: int = 31,
    density: float = 1.55,
    stream_color: tuple[float, float, float, float] | None = None,
    stream_linewidth: float = 0.45,
    stream_arrowsize: float = 0.42,
    quiver_step: int = 3,
    quiver_scale: float = 0.42,
    quiver_width: float = 0.0020,
    quiver_color: str | None = None,
    quiver_alpha: float = 0.32,
) -> None:
    x = np.linspace(xlim[0], xlim[1], int(grid_n))
    y = np.linspace(ylim[0], ylim[1], int(grid_n))
    xx, yy = np.meshgrid(x, y)
    pts_mm = np.asarray(
        [point_from_plane(float(a), float(b), plane, anchor_mm) for a, b in zip(xx.reshape(-1), yy.reshape(-1))],
        dtype=np.float64,
    )
    table = mas._table
    a_mat = interpolate_A_vmap(table.A_table, table.xs, table.ys, table.zs, jnp.asarray(pts_mm / 1000.0))
    y8 = np.asarray(jnp.einsum("nij,j->ni", a_mat, jnp.asarray(currents, dtype=jnp.float64)), dtype=np.float64)
    b_vecs = y8[:, 0:3]
    uv = np.asarray([project_vector(vec, plane) for vec in b_vecs], dtype=np.float64)
    uu = uv[:, 0].reshape(xx.shape)
    vv = uv[:, 1].reshape(yy.shape)
    mag = np.hypot(uu, vv)
    uu = uu / np.maximum(mag, 1e-18)
    vv = vv / np.maximum(mag, 1e-18)
    ax.streamplot(
        x,
        y,
        uu,
        vv,
        density=density,
        color=stream_color if stream_color is not None else (0.0, 0.62, 0.45, 0.18),
        linewidth=stream_linewidth,
        arrowsize=stream_arrowsize,
        minlength=0.08,
        maxlength=2.2,
        zorder=1,
    )
    step = max(1, int(quiver_step))
    ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        uu[::step, ::step],
        vv[::step, ::step],
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=quiver_width,
        headwidth=2.5,
        headlength=2.8,
        headaxislength=2.5,
        color=quiver_color if quiver_color is not None else FIELD_COLOR,
        alpha=quiver_alpha,
        zorder=2,
    )


def catheter_view_limits(cat: np.ndarray, field_pts_mm: np.ndarray | None, plane: str, *, pad: float) -> tuple[np.ndarray, float]:
    points = cat if field_pts_mm is None else np.vstack([cat, field_pts_mm])
    px, py = project(points, plane)
    center = np.array([0.5 * (px.min() + px.max()), 0.5 * (py.min() + py.max())], dtype=np.float64)
    radius = 0.5 * max(float(px.max() - px.min()), float(py.max() - py.min())) + float(pad)
    return center, radius


def style_2d_axis(ax: plt.Axes, xlabel: str, ylabel: str, label: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontproperties=en_font(22.0))
    ax.set_ylabel(ylabel, fontproperties=en_font(22.0))
    ax.text(0.02, 0.96, label, transform=ax.transAxes, fontproperties=zh_font(22.8, "bold"), va="top")
    ax.grid(True, color="0.88", linewidth=0.55)
    ax.tick_params(width=0.65, length=3.0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(en_font(19.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_top_zoom_inset(ax: plt.Axes, case: dict[str, Any], cat: np.ndarray, plane: str) -> None:
    inset = ax.inset_axes([0.57, 0.08, 0.37, 0.37])
    add_catheter_2d(inset, case, plane, annotate_magnet=False)
    center, radius = catheter_view_limits(cat, None, plane, pad=5.0)
    inset.set_xlim(center[0] - radius, center[0] + radius)
    inset.set_ylim(center[1] - radius, center[1] + radius)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(True, color="0.90", linewidth=0.45)
    inset.set_facecolor((1.0, 1.0, 1.0, 0.92))
    for spine in inset.spines.values():
        spine.set_edgecolor("0.35")
        spine.set_linewidth(0.75)


def plot_views_case(
    case: dict[str, Any],
    coils: list[dict[str, Any]],
    field_pts_mm: np.ndarray,
    field_dirs: np.ndarray,
    mas: MagneticActuationSystem,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.2))
    planes = [("xy", "x (mm)", "y (mm)", "俯视 x-y"), ("xz", "x (mm)", "z (mm)", "正视 x-z"), ("yz", "y (mm)", "z (mm)", "侧视 y-z")]
    coil_centers = np.asarray([coil["position_m"] * 1000.0 for coil in coils], dtype=np.float64)
    cat = np.concatenate([line * 1000.0 for line in case["centerlines_m"]], axis=0)
    all_points = np.vstack([coil_centers, cat, field_pts_mm])
    active_idx = int(case["active_idx"])
    anchor_mm = np.mean(cat, axis=0)
    for col, (plane, xlabel, ylabel, panel_label) in enumerate(planes):
        for row in range(2):
            ax = axes[row, col]
            if row == 0:
                for coil in coils:
                    draw_projected_cylinder(ax, coil, active_idx=active_idx, plane=plane, show_label=False)
                add_catheter_2d_global_scaled(ax, case, plane, scale=7.0)
                px, py = project(all_points, plane)
                center = np.array([0.5 * (px.min() + px.max()), 0.5 * (py.min() + py.max())])
                radius = 0.5 * max(float(px.max() - px.min()), float(py.max() - py.min())) + 70.0
                label = panel_label + " / 全局"
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
            else:
                center, radius = catheter_view_limits(cat, field_pts_mm, plane, pad=8.5)
                xlim = (float(center[0] - radius), float(center[0] + radius))
                ylim = (float(center[1] - radius), float(center[1] + radius))
                add_dense_field_2d(
                    ax,
                    mas,
                    case["currents"],
                    plane,
                    xlim=xlim,
                    ylim=ylim,
                    anchor_mm=anchor_mm,
                )
                add_catheter_2d(ax, case, plane, annotate_magnet=True)
                label = panel_label + " / 放大"
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
            style_2d_axis(ax, xlabel, ylabel, label)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    axes[1, 0].legend(
        dedup.values(),
        dedup.keys(),
        loc="lower left",
        frameon=False,
        prop=zh_font(20.0),
        handlelength=1.8,
    )
    fig.text(
        0.055,
        0.025,
        f"C{active_idx + 1}: 15 A；最终残差 $||E||$={case['final_norm']:.2e}",
        fontproperties=zh_font(20.8),
        color="0.18",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    save_figure(fig, out_base)
    plt.close(fig)


def plot_catheter_zoom_3d_case(
    case: dict[str, Any],
    mas: MagneticActuationSystem,
    out_base: Path,
) -> None:
    cat = np.concatenate([line * 1000.0 for line in case["centerlines_m"]], axis=0)
    field_pts_mm, field_dirs = local_plane_field_samples(mas, case)
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    add_field_3d(
        ax,
        field_pts_mm,
        field_dirs,
        length=8.0,
        color=ZOOM_FIELD_COLOR,
        alpha=0.76,
        linewidth=1.45,
        arrow_length_ratio=0.24,
    )
    add_catheter_3d(ax, case, show_intermediate=True, annotation_size=13.6, main_lw_scale=1.32)
    style_3d_axis(ax, label_size=14.0, tick_size=12.2, grid_width=0.78)
    set_equal_3d_limits(ax, np.vstack([cat, field_pts_mm]), pad_mm=13.0)
    elev, azim = local_plane_view_angles(case)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_ylabel("")
    ax.set_yticks([])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    save_figure(fig, out_base)
    plt.close(fig)


def plot_catheter_zoom_views_case(
    case: dict[str, Any],
    mas: MagneticActuationSystem,
    out_base: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0))
    planes = [("xy", "x (mm)", "y (mm)", "俯视 x-y"), ("xz", "x (mm)", "z (mm)", "正视 x-z"), ("yz", "y (mm)", "z (mm)", "侧视 y-z")]
    cat = np.concatenate([line * 1000.0 for line in case["centerlines_m"]], axis=0)
    anchor_mm = np.mean(cat, axis=0)
    label_size = 20.4
    tick_size = 18.0
    for ax, (plane, xlabel, ylabel, panel_label) in zip(axes, planes):
        center, radius = catheter_view_limits(cat, None, plane, pad=8.0)
        xlim = (float(center[0] - radius), float(center[0] + radius))
        ylim = (float(center[1] - radius), float(center[1] + radius))
        add_dense_field_2d(
            ax,
            mas,
            case["currents"],
            plane,
            xlim=xlim,
            ylim=ylim,
            anchor_mm=anchor_mm,
            grid_n=49,
            density=2.75,
            stream_color=(0.03, 0.16, 0.48, 0.46),
            stream_linewidth=0.46,
            stream_arrowsize=0.18,
            quiver_step=2,
            quiver_scale=4.2,
            quiver_width=0.00145,
            quiver_color=ZOOM_FIELD_COLOR,
            quiver_alpha=0.72,
        )
        add_catheter_2d(ax, case, plane, annotate_magnet=True, main_lw_scale=1.75, tip_lw_scale=1.45)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel, fontproperties=en_font(label_size))
        ax.set_ylabel(ylabel, fontproperties=en_font(label_size))
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, fontproperties=zh_font(label_size, "bold"), va="top")
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(width=0.85, length=4.0)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(en_font(tick_size))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    axes[0].legend(dedup.values(), dedup.keys(), loc="lower left", frameon=False, prop=zh_font(label_size), handlelength=2.0)
    fig.text(
        0.055,
        0.025,
        f"C{int(case['active_idx']) + 1}: 15 A；最终残差 $||E||$={case['final_norm']:.2e}",
        fontproperties=zh_font(18.0),
        color="0.18",
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0), w_pad=2.7)
    save_figure(fig, out_base)
    plt.close(fig)


def plot_catheter_xz_overview(cases: list[dict[str, Any]], out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))
    for ax, case in zip(axes.reshape(-1), cases):
        add_catheter_2d(ax, case, "xz", annotate_magnet=True)
        cat = np.concatenate([line * 1000.0 for line in case["centerlines_m"]], axis=0)
        px, py = project(cat, "xz")
        center = np.array([0.5 * (px.min() + px.max()), 0.5 * (py.min() + py.max())])
        radius = 0.5 * max(float(px.max() - px.min()), float(py.max() - py.min())) + 6.5
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)", fontproperties=en_font(8.0))
        ax.set_ylabel("z (mm)", fontproperties=en_font(8.0))
        ax.text(
            0.03,
            0.95,
            f"C{int(case['active_idx']) + 1}: 15 A",
            transform=ax.transAxes,
            fontproperties=en_font(8.2, "bold"),
            va="top",
        )
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(width=0.65, length=3.0)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(en_font(7.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=3, frameon=False, prop=zh_font(7.6))
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    save_figure(fig, out_base)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--calib_file", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--actuation_table_pkl", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    setup_plot_style()
    base_dir = args.base_dir.resolve()
    out_dir = (args.out_dir or (base_dir / "visualization")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coils = load_coils(args.calib_file.resolve())
    mas = MagneticActuationSystem(
        calib_file=str(args.calib_file.resolve()),
        actuation_table_pkl=str(args.actuation_table_pkl.resolve()),
        enable_x64=True,
    )

    summary = []
    loaded_cases: list[dict[str, Any]] = []
    zoom_dir = out_dir / "catheter_zoom"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    for case_name in ["coil1_15A", "coil2_15A", "coil3_15A", "coil4_15A"]:
        case = load_case(base_dir / case_name)
        loaded_cases.append(case)
        field_pts_mm, field_dirs = field_samples(mas, case["currents"])
        plot_3d_case(case, coils, field_pts_mm, field_dirs, out_dir / f"{case_name}_3d")
        plot_views_case(case, coils, field_pts_mm, field_dirs, mas, out_dir / f"{case_name}_three_views")
        plot_catheter_zoom_3d_case(case, mas, zoom_dir / f"{case_name}_catheter_3d")
        plot_catheter_zoom_views_case(case, mas, zoom_dir / f"{case_name}_catheter_three_views")
        final_line = case["centerlines_m"][-1]
        tip = final_line[-1]
        summary.append(
            {
                "case": case_name,
                "active_coil": int(case["active_idx"]) + 1,
                "final_normE": float(case["final_norm"]),
                "final_tip_m": [float(v) for v in tip],
                "iteration_count": int(case["iters"][-1]),
            }
        )
    plot_catheter_xz_overview(loaded_cases, zoom_dir / "all_cases_catheter_xz_overview")

    (out_dir / "visualization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "cases": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
