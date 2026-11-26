"""Plot straight initial postures for deliverable catheter configurations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "catheter_physical_configs.yaml"
OUT_DIR = ROOT / "figures_initial_postures"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _axis_label(vec: list[float]) -> str:
    if abs(vec[0]) > 0.5:
        return "x 向径向磁化"
    if abs(vec[2]) > 0.5:
        return "z 向轴向磁化"
    return "自定义磁化"


def _axis_label_short(vec: list[float]) -> str:
    if abs(vec[0]) > 0.5:
        return "x 向"
    if abs(vec[2]) > 0.5:
        return "z 向"
    return "自定义"


def _segments_from_config(item: dict) -> tuple[list[tuple[float, float]], list[dict], float]:
    """Return straight undeformed segments in proximal-to-distal order."""
    solver = item["solver_input"]
    L_controlled = float(item["shared_controlled_flex_length_m"])
    x = 0.0
    flex_segments: list[tuple[float, float]] = []
    magnets: list[dict] = []

    if solver["segment_count_N"] == 1:
        flex_segments.append((x, x + L_controlled))
        x += L_controlled
        m = item["true_segments_distal_to_proximal"][0]
        Lm = float(m["length_m"])
        magnets.append(
            {
                "start": x,
                "end": x + Lm,
                "center": x + 0.5 * Lm,
                "axis": m["magnetization_axis_body"],
                "label": "磁体 1",
            }
        )
        x += Lm
        return flex_segments, magnets, x

    if solver["segment_count_N"] == 2:
        distal_magnet, fixed_flex, proximal_magnet, _controlled_flex = item["true_segments_distal_to_proximal"]
        flex_segments.append((x, x + L_controlled))
        x += L_controlled
        Lm = float(proximal_magnet["length_m"])
        magnets.append(
            {
                "start": x,
                "end": x + Lm,
                "center": x + 0.5 * Lm,
                "axis": proximal_magnet["magnetization_axis_body"],
                "label": "近端磁体 2",
            }
        )
        x += Lm
        Lf = float(fixed_flex["fixed_length_m"])
        flex_segments.append((x, x + Lf))
        x += Lf
        Lm = float(distal_magnet["length_m"])
        magnets.append(
            {
                "start": x,
                "end": x + Lm,
                "center": x + 0.5 * Lm,
                "axis": distal_magnet["magnetization_axis_body"],
                "label": "远端磁体 1",
            }
        )
        x += Lm
        return flex_segments, magnets, x

    raise ValueError(f"Unsupported segment_count_N={solver['segment_count_N']}")


def _prepare_items(config: dict) -> dict:
    items = {}
    for cid, item in config["catheter_types"].items():
        copied = dict(item)
        copied["shared_controlled_flex_length_m"] = config["shared_physical_parameters"]["flexible_body"][
            "max_controlled_length_m"
        ]
        items[cid] = copied
    return items


def _draw_overview(items: dict, font: FontProperties) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=180)
    blue = "#1764c8"
    red = "#d92525"
    grey = "#6a6a6a"

    y_gap = 1.0
    y_positions = list(reversed(range(len(items))))
    max_len = 0.0

    for y, (cid, item) in zip(y_positions, items.items()):
        flex_segments, magnets, total_len = _segments_from_config(item)
        max_len = max(max_len, total_len)
        ax.text(
            -0.055,
            y,
            f"{cid}  {item['label_zh']}",
            ha="right",
            va="center",
            fontproperties=font,
            fontsize=11,
            color="#202020",
        )
        for x0, x1 in flex_segments:
            ax.plot([x0, x1], [y, y], color=blue, lw=8, solid_capstyle="round", zorder=2)
        for mag_idx, mag in enumerate(magnets):
            ax.plot([mag["start"], mag["end"]], [y, y], color=red, lw=2.8, alpha=0.85, zorder=3)
            ax.scatter(
                [mag["center"]],
                [y],
                s=145,
                facecolors="white",
                edgecolors=red,
                linewidths=2.4,
                zorder=4,
            )
            axis_label = _axis_label_short(mag["axis"])
            y_text = y + (0.23 if mag_idx % 2 == 0 else 0.38)
            ax.text(
                mag["center"],
                y_text,
                axis_label,
                ha="center",
                va="bottom",
                fontproperties=font,
                fontsize=9.0,
                color=red,
                rotation=0,
            )
        ax.scatter([0.0], [y], s=34, color=grey, zorder=5)
        ax.text(0.0, y - 0.23, "近端", ha="center", va="top", fontproperties=font, fontsize=8.5, color=grey)
        ax.text(total_len, y - 0.23, "远端", ha="center", va="top", fontproperties=font, fontsize=8.5, color=grey)

    ax.plot([], [], color=blue, lw=8, solid_capstyle="round", label="柔性段")
    ax.scatter([], [], s=145, facecolors="white", edgecolors=red, linewidths=2.4, label="磁体段中心")
    leg = ax.legend(prop=font, loc="lower right", frameon=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("#dddddd")

    ax.set_xlim(-0.17, max_len + 0.04)
    ax.set_ylim(-0.75, len(items) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("沿导管轴线的弧长位置 s (m)", fontproperties=font, fontsize=12)
    ax.set_title("A1/A2/B1/B2/B3 导管未变形初始姿态总览", fontproperties=font, fontsize=17, pad=18)
    ax.grid(axis="x", color="#d9e4f2", lw=0.8)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    fig.tight_layout()
    out = OUT_DIR / "catheter_initial_postures_overview.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_distal_zoom(items: dict, font: FontProperties) -> Path:
    fig, axes = plt.subplots(len(items), 1, figsize=(10.5, 9.0), dpi=180, sharex=False)
    blue = "#1764c8"
    red = "#d92525"
    grey = "#5f6368"

    for ax, (cid, item) in zip(axes, items.items()):
        flex_segments, magnets, total_len = _segments_from_config(item)
        window_start = max(0.0, total_len - 0.085)
        for x0, x1 in flex_segments:
            lo = max(x0, window_start)
            hi = min(x1, total_len)
            if hi > lo:
                ax.plot([lo, hi], [0, 0], color=blue, lw=8, solid_capstyle="round", zorder=2)
        for mag in magnets:
            if mag["end"] >= window_start:
                ax.plot([mag["start"], mag["end"]], [0, 0], color=red, lw=3.0, zorder=3)
                ax.scatter([mag["center"]], [0], s=160, facecolors="white", edgecolors=red, linewidths=2.5, zorder=4)
                ax.text(
                    mag["center"],
                    0.23,
                    f"{mag['label']}\n{_axis_label(mag['axis'])}",
                    ha="center",
                    va="bottom",
                    fontproperties=font,
                    fontsize=8.5,
                    color=red,
                )
        ax.axvline(total_len, color=grey, lw=1.0, ls="--", alpha=0.7)
        ax.text(
            window_start,
            -0.33,
            cid,
            ha="left",
            va="top",
            fontproperties=font,
            fontsize=12,
            weight="bold",
            color="#202020",
        )
        ax.text(total_len, -0.33, "远端", ha="center", va="top", fontproperties=font, fontsize=9, color=grey)
        ax.set_xlim(window_start, total_len + 0.006)
        ax.set_ylim(-0.5, 0.62)
        ax.set_yticks([])
        ax.grid(axis="x", color="#e7edf5", lw=0.8)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color("#9aa0a6")
    axes[-1].set_xlabel("远端局部弧长位置 s (m)", fontproperties=font, fontsize=12)
    fig.suptitle("导管未变形初始姿态：远端局部放大", fontproperties=font, fontsize=17, y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "catheter_initial_postures_distal_zoom.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _set_axes_equal_3d(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range)
    x_mid = sum(x_limits) * 0.5
    y_mid = sum(y_limits) * 0.5
    z_mid = sum(z_limits) * 0.5
    ax.set_xlim3d(x_mid - radius, x_mid + radius)
    ax.set_ylim3d(y_mid - radius, y_mid + radius)
    ax.set_zlim3d(z_mid - radius, z_mid + radius)


def _draw_cylinder_like_line(ax, start, end, *, color: str, linewidth: float, alpha: float = 1.0, zorder: int = 2):
    collection = Line3DCollection([[start, end]], colors=[color], linewidths=[linewidth], alpha=alpha)
    collection.set_zorder(zorder)
    ax.add_collection3d(collection)


def _draw_3d_overview(items: dict, font: FontProperties) -> Path:
    fig = plt.figure(figsize=(12.5, 8.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    blue = "#1764c8"
    red = "#d92525"
    grey = "#686868"
    y_gap = 0.115
    z_offsets = {"A1": 0.02, "A2": -0.02, "B1": 0.02, "B2": -0.02, "B3": 0.0}
    max_len = 0.0

    for idx, (cid, item) in enumerate(items.items()):
        y = (len(items) - 1 - idx) * y_gap
        z = z_offsets.get(cid, 0.0)
        flex_segments, magnets, total_len = _segments_from_config(item)
        max_len = max(max_len, total_len)
        ax.text(
            -0.075,
            y,
            z,
            cid,
            fontproperties=font,
            fontsize=12,
            color="#222222",
            ha="right",
            va="center",
        )
        for x0, x1 in flex_segments:
            _draw_cylinder_like_line(ax, (x0, y, z), (x1, y, z), color=blue, linewidth=8.5, zorder=2)
        for mag_idx, mag in enumerate(magnets):
            _draw_cylinder_like_line(
                ax,
                (mag["start"], y, z),
                (mag["end"], y, z),
                color=red,
                linewidth=3.2,
                zorder=3,
            )
            ax.scatter(
                [mag["center"]],
                [y],
                [z],
                s=115,
                marker="o",
                facecolors="white",
                edgecolors=red,
                linewidths=2.2,
                depthshade=False,
                zorder=4,
            )
            ax.text(
                mag["center"],
                y,
                z + 0.035 + 0.012 * mag_idx,
                _axis_label_short(mag["axis"]),
                fontproperties=font,
                fontsize=8.5,
                color=red,
                ha="center",
                va="bottom",
            )
        ax.scatter([0.0], [y], [z], s=36, color=grey, depthshade=False, zorder=5)

    ax.plot([], [], [], color=blue, lw=8.5, label="柔性段")
    ax.scatter([], [], [], s=115, marker="o", facecolors="white", edgecolors=red, linewidths=2.2, label="磁体段中心")
    leg = ax.legend(prop=font, loc="upper left", bbox_to_anchor=(0.03, 0.96))
    leg.get_frame().set_alpha(0.95)

    ax.set_xlim(-0.08, max_len + 0.05)
    ax.set_ylim(-0.08, len(items) * y_gap)
    ax.set_zlim(-0.13, 0.13)
    ax.set_xlabel("x / 轴向弧长 s (m)", fontproperties=font, labelpad=12)
    ax.set_ylabel("y / 导管类型分列", fontproperties=font, labelpad=12)
    ax.set_zlabel("z / 展示偏移 (m)", fontproperties=font, labelpad=12)
    ax.set_title("A1/A2/B1/B2/B3 导管未变形初始姿态三维总览", fontproperties=font, fontsize=16, pad=22)
    ax.view_init(elev=22, azim=-63)
    ax.grid(True, color="#dfe8f4")
    _set_axes_equal_3d(ax)
    fig.tight_layout()
    out = OUT_DIR / "catheter_initial_postures_3d_overview.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_3d_distal_zoom(items: dict, font: FontProperties) -> Path:
    fig = plt.figure(figsize=(12.5, 8.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    blue = "#1764c8"
    red = "#d92525"
    grey = "#686868"
    y_gap = 0.105
    window_width = 0.09

    for idx, (cid, item) in enumerate(items.items()):
        y = (len(items) - 1 - idx) * y_gap
        z = 0.0
        flex_segments, magnets, total_len = _segments_from_config(item)
        window_start = max(0.0, total_len - window_width)
        ax.text(window_start - 0.014, y, z, cid, fontproperties=font, fontsize=12, ha="right", va="center")
        for x0, x1 in flex_segments:
            lo = max(x0, window_start)
            hi = min(x1, total_len)
            if hi > lo:
                _draw_cylinder_like_line(ax, (lo, y, z), (hi, y, z), color=blue, linewidth=8.5, zorder=2)
        for mag_idx, mag in enumerate(magnets):
            if mag["end"] >= window_start:
                _draw_cylinder_like_line(
                    ax,
                    (mag["start"], y, z),
                    (mag["end"], y, z),
                    color=red,
                    linewidth=3.2,
                    zorder=3,
                )
                ax.scatter(
                    [mag["center"]],
                    [y],
                    [z],
                    s=125,
                    marker="o",
                    facecolors="white",
                    edgecolors=red,
                    linewidths=2.3,
                    depthshade=False,
                    zorder=4,
                )
                ax.text(
                    mag["center"],
                    y,
                    z + 0.025 + 0.012 * mag_idx,
                    f"{mag['label']}\n{_axis_label(mag['axis'])}",
                    fontproperties=font,
                    fontsize=8.4,
                    color=red,
                    ha="center",
                    va="bottom",
                )
        _draw_cylinder_like_line(ax, (total_len, y, -0.025), (total_len, y, 0.025), color=grey, linewidth=1.2, alpha=0.8)

    ax.plot([], [], [], color=blue, lw=8.5, label="柔性段")
    ax.scatter([], [], [], s=125, marker="o", facecolors="white", edgecolors=red, linewidths=2.3, label="磁体段中心")
    leg = ax.legend(prop=font, loc="upper left", bbox_to_anchor=(0.03, 0.96))
    leg.get_frame().set_alpha(0.95)

    ax.set_xlabel("x / 远端局部弧长 s (m)", fontproperties=font, labelpad=12)
    ax.set_ylabel("y / 导管类型分列", fontproperties=font, labelpad=12)
    ax.set_zlabel("z / 展示偏移 (m)", fontproperties=font, labelpad=12)
    ax.set_title("导管未变形初始姿态三维图：远端局部放大", fontproperties=font, fontsize=16, pad=22)
    ax.view_init(elev=24, azim=-62)
    ax.grid(True, color="#dfe8f4")
    _set_axes_equal_3d(ax)
    fig.tight_layout()
    out = OUT_DIR / "catheter_initial_postures_3d_distal_zoom.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _magnet_plot_direction(axis_body: list[float]) -> tuple[float, float, float]:
    """Map catheter body-frame magnetization to the plot frame.

    In the straight initial posture, body z is shown along the plot x axis
    and body x is shown along the plot z axis.
    """
    if abs(axis_body[2]) > 0.5:
        return (1.0, 0.0, 0.0)
    if abs(axis_body[0]) > 0.5:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def _user_magnets_with_centers(item: dict) -> list[dict]:
    """Return magnets in user order with centers in proximal-to-distal arc length."""
    solver = item["solver_input"]
    L_controlled = float(item["shared_controlled_flex_length_m"])
    if solver["segment_count_N"] == 1:
        magnet = item["true_segments_distal_to_proximal"][0]
        Lm = float(magnet["length_m"])
        return [
            {
                "name": "m1",
                "center_m": L_controlled + 0.5 * Lm,
                "axis": magnet["magnetization_axis_body"],
                "label": "m1",
            }
        ]

    distal_magnet, fixed_flex, proximal_magnet, _controlled_flex = item["true_segments_distal_to_proximal"]
    Lm = float(distal_magnet["length_m"])
    fixed_len = float(fixed_flex["fixed_length_m"])
    proximal_center = L_controlled + 0.5 * Lm
    distal_center = L_controlled + Lm + fixed_len + 0.5 * Lm
    return [
        {
            "name": "m1",
            "center_m": distal_center,
            "axis": distal_magnet["magnetization_axis_body"],
            "label": "m1",
        },
        {
            "name": "m2",
            "center_m": proximal_center,
            "axis": proximal_magnet["magnetization_axis_body"],
            "label": "m2",
        },
    ]


def _draw_sample_style_single_config(cid: str, item: dict, font: FontProperties) -> Path:
    fig = plt.figure(figsize=(8.4, 6.8), dpi=210)
    ax = fig.add_subplot(111, projection="3d")
    red = "#c61f3b"
    orange = "#e56b00"
    black = "#282828"

    _flex_segments, _magnets, total_len_m = _segments_from_config(item)
    total_len_mm = total_len_m * 1000.0
    magnets = _user_magnets_with_centers(item)

    # Draw the straight undeformed centerline.
    ax.plot(
        [0.0, total_len_mm],
        [0.0, 0.0],
        [0.0, 0.0],
        color=black,
        lw=3.0,
        solid_capstyle="round",
        label="初始位姿",
        zorder=2,
    )

    # Use a short orange distal cap to visually echo the reference style without
    # pretending that a deformed pose exists.
    distal_cap_start = max(0.0, total_len_mm - 18.0)
    ax.plot(
        [distal_cap_start, total_len_mm],
        [0.0, 0.0],
        [0.0, 0.0],
        color=orange,
        lw=3.0,
        solid_capstyle="round",
        label="远端段",
        zorder=3,
    )

    arrow_len = 38.0
    for idx, mag in enumerate(magnets):
        x = mag["center_m"] * 1000.0
        y = 0.0
        z = 0.0
        dx, dy, dz = _magnet_plot_direction(mag["axis"])
        ax.scatter(
            [x],
            [y],
            [z],
            s=75,
            facecolors="white",
            edgecolors=red,
            linewidths=1.8,
            depthshade=False,
            zorder=5,
        )
        ax.quiver(
            x,
            y,
            z,
            dx,
            dy,
            dz,
            length=arrow_len,
            normalize=True,
            color=red,
            linewidth=1.3,
            arrow_length_ratio=0.22,
            zorder=4,
        )
        label_dz = 10.0 + 5.0 * idx
        ax.text(
            x,
            y + 3.0,
            z + label_dz,
            mag["label"],
            color=red,
            fontproperties=font,
            fontsize=10.0,
            ha="center",
            va="bottom",
        )

    ax.scatter(
        [total_len_mm],
        [0.0],
        [0.0],
        s=18,
        color="black",
        depthshade=False,
        label="末端轨迹",
        zorder=6,
    )
    ax.text(
        18.0,
        -32.0,
        42.0,
        f"{cid}: {item['label_zh']}",
        fontproperties=font,
        fontsize=10.8,
        color="#202020",
        ha="left",
        va="center",
    )

    legend_handles = [
        Line2D([0], [0], color=black, lw=3.0, label="初始位姿"),
        Line2D([0], [0], color=orange, lw=3.0, label="远端段"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", lw=0, markersize=3.5, label="末端轨迹"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=red,
            markerfacecolor="white",
            markeredgecolor=red,
            lw=0,
            markersize=7.0,
            label="磁化点",
        ),
        Line2D([0], [0], color=red, lw=1.3, label="磁化方向"),
    ]
    ax.legend(handles=legend_handles, prop=font, loc="upper left", frameon=False)

    ax.set_xlabel("x (mm)", fontproperties=font, labelpad=8)
    ax.set_ylabel("y (mm)", fontproperties=font, labelpad=8)
    ax.set_zlabel("z (mm)", fontproperties=font, labelpad=8)
    ax.set_xlim(-25.0, total_len_mm + 50.0)
    ax.set_ylim(-55.0, 75.0)
    ax.set_zlim(-70.0, 70.0)
    try:
        ax.set_box_aspect((3.2, 1.0, 1.0))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-58)
    ax.grid(True)
    ax.set_title(f"{cid} 导管未变形初始姿态", fontproperties=font, fontsize=14, pad=14)
    fig.tight_layout()

    out = OUT_DIR / f"catheter_initial_posture_{cid}_sample_style.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_sample_style_all_configs(items: dict, font: FontProperties) -> list[Path]:
    return [_draw_sample_style_single_config(cid, item, font) for cid, item in items.items()]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    items = _prepare_items(config)
    font = _font()
    overview = _draw_overview(items, font)
    zoom = _draw_distal_zoom(items, font)
    overview_3d = _draw_3d_overview(items, font)
    zoom_3d = _draw_3d_distal_zoom(items, font)
    per_config = _draw_sample_style_all_configs(items, font)
    print(f"wrote {overview}")
    print(f"wrote {zoom}")
    print(f"wrote {overview_3d}")
    print(f"wrote {zoom_3d}")
    for path in per_config:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
