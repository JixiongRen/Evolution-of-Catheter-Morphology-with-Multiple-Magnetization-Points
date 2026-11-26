from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
CJK_FONT_PATH = ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"
LATIN_REGULAR_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
LATIN_ITALIC_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf")
LATIN_BOLD_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf")
LAYERED_DIR = ROOT / "experiment_records_ik_optimize" / "20260605002600_p2c10f_feasibility_first_layered_shooting"
SENS_DIR = ROOT / "experiment_records_ik_optimize" / "20260606173500_p2c28_cross_condition_reproducibility_map"
SINGLE_DIR = ROOT / "experiment_records_ik_optimize" / "20260618_single_layer_ik_strict_fk_roundtrip_v3"


def _available_font_family(preferred: list[str], fallback: str) -> str:
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return fallback


def setup_report_fonts() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    if CJK_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(CJK_FONT_PATH))
        cjk_prop = font_manager.FontProperties(fname=str(CJK_FONT_PATH))
        cjk_family = "Source Han Serif CN"
    else:
        cjk_family = _available_font_family(["SimSun", "宋体", "Noto Serif CJK SC", "STSong"], "WenQuanYi Zen Hei")
        cjk_prop = font_manager.FontProperties(family=[cjk_family])

    for font_path in (LATIN_REGULAR_PATH, LATIN_ITALIC_PATH, LATIN_BOLD_PATH):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    if LATIN_REGULAR_PATH.exists():
        latin_prop = font_manager.FontProperties(fname=str(LATIN_REGULAR_PATH))
        latin_family = latin_prop.get_name()
    else:
        latin_family = _available_font_family(["Times New Roman", "Liberation Serif", "Nimbus Roman"], "DejaVu Serif")
        latin_prop = font_manager.FontProperties(family=[latin_family])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [latin_family, cjk_family, "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": latin_family,
            "mathtext.it": f"{latin_family}:italic",
            "mathtext.bf": f"{latin_family}:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    return cjk_prop, latin_prop


def _apply_latin_ticks(ax: plt.Axes, latin_prop: font_manager.FontProperties) -> None:
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(latin_prop)


def _save_all(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=360)
    fig.savefig(out_base.with_suffix(".pdf"))
    svg_path = out_base.with_suffix(".svg")
    fig.savefig(svg_path)
    _rewrite_svg_font_families(svg_path)
    plt.close(fig)


def _rewrite_svg_font_families(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    replacements = {
        "'Source Han Serif CN'": "'SimSun'",
        "Source Han Serif CN": "SimSun",
        "'Liberation Serif'": "'Times New Roman'",
        "Liberation Serif": "Times New Roman",
        "'Nimbus Roman'": "'Times New Roman'",
        "Nimbus Roman": "Times New Roman",
        "'DejaVu Sans'": "'Times New Roman'",
        "DejaVu Sans": "Times New Roman",
        "'DejaVu Serif'": "'Times New Roman'",
        "DejaVu Serif": "Times New Roman",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = _rewrite_svg_cjk_nodes_to_simsun(text)
    svg_path.write_text(text, encoding="utf-8")


def _rewrite_svg_cjk_nodes_to_simsun(svg_text: str) -> str:
    cjk_re = re.compile(r"[\u3400-\u9fff]")
    tag_re = re.compile(r"<(?P<tag>text|tspan)(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)
    font_re = re.compile(r"font-family:[^;\"']*(?:'[^']*'[^;\"']*)*")

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        attrs = match.group("attrs")
        if not cjk_re.search(body):
            return match.group(0)
        if "font-family:" in attrs:
            attrs = font_re.sub("font-family: 'SimSun'", attrs)
        elif "style=" in attrs:
            attrs = attrs.replace('style="', "style=\"font-family: 'SimSun'; ")
        else:
            attrs = attrs + " style=\"font-family: 'SimSun'\""
        return f"<{match.group('tag')}{attrs}>{body}</{match.group('tag')}>"

    return tag_re.sub(repl, svg_text)


def plot_layered_residuals(cjk_prop: font_manager.FontProperties, latin_prop: font_manager.FontProperties) -> Path:
    rows: list[dict[str, float]] = []
    with (LAYERED_DIR / "summary.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "alpha": float(row["alpha"]),
                    "lambda_t": float(row["task_scale"]),
                    "task": float(row["task_norm"]),
                    "boundary": float(row["boundary_norm"]),
                    "continuity": float(row["continuity_norm"]),
                }
            )

    alphas = sorted({row["alpha"] for row in rows})
    series = [
        ("任务残差", "任务残差判据", "task", "#1f77b4", "o", 1.0e-2),
        ("末端边界残差", "末端边界判据", "boundary", "#d55e00", "s", 2.0e-1),
        ("段间连续残差", "段间连续判据", "continuity", "#009e73", "^", 1.0e-3),
    ]

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
        }
    )
    fig, axes = plt.subplots(1, len(alphas), figsize=(10.8, 4.7), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    handles: list[object] = []
    labels: list[str] = []
    for ax, alpha in zip(axes, alphas):
        sub = sorted([row for row in rows if row["alpha"] == alpha], key=lambda r: r["lambda_t"])
        x = np.asarray([row["lambda_t"] for row in sub], dtype=float)
        x_fit = np.linspace(float(x.min()), float(x.max()), 280)
        for name, threshold_name, key, color, marker, threshold in series:
            y = np.asarray([row[key] for row in sub], dtype=float)
            fit = PchipInterpolator(x, np.log10(y))
            y_fit = 10.0 ** fit(x_fit)
            (line,) = ax.plot(x_fit, y_fit, color=color, linewidth=3.0, solid_capstyle="round", label=name)
            ax.scatter(x, y, color="white", edgecolor=color, marker=marker, s=92, linewidth=2.4, zorder=3)
            hline = ax.axhline(
                threshold,
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=1.7,
                alpha=0.44,
                zorder=0,
                label=threshold_name,
            )
            if ax is axes[0]:
                handles.extend([line, hline])
                labels.extend([name, threshold_name])

        ax.set_yscale("log")
        ax.set_ylim(4.0e-5, 2.2)
        ax.set_xlim(-0.035, 1.035)
        ax.set_xticks([0.0, 0.1, 0.3, 1.0])
        ax.set_xlabel(r"任务强度 $\lambda_t$", fontproperties=cjk_prop)
        ax.set_title(rf"$\alpha={alpha:.2f}$", fontproperties=latin_prop)
        ax.grid(True, which="major", color="#b7b7b7", alpha=0.34, linewidth=0.9)
        ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=0.18, linewidth=0.55)
        ax.tick_params(direction="out", length=5.0, width=1.1)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=7))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=30))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.text(
            0.03,
            0.06,
            "虚线：对应残差判据",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            color="#555555",
            fontproperties=cjk_prop,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _apply_latin_ticks(ax, latin_prop)

    axes[0].set_ylabel("残差范数", fontproperties=cjk_prop)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
        handlelength=2.2,
        columnspacing=1.7,
        prop=cjk_prop,
    )
    fig.subplots_adjust(top=0.82, left=0.09, right=0.99, bottom=0.18, wspace=0.12)
    out_base = LAYERED_DIR / "fig_322_layered_shooting_residuals"
    _save_all(fig, out_base)
    return out_base.with_suffix(".png")


def plot_sensitivity_reliability(cjk_prop: font_manager.FontProperties, latin_prop: font_manager.FontProperties) -> Path:
    scene_labels = {
        ("n3_mixed_base_i4", "rep_a"): "三磁体混合\n低电流-A",
        ("n3_mixed_base_i4", "rep_b"): "三磁体混合\n低电流-B",
        ("n1_tip_z_rand15", "diag_n1"): "末端单磁体\n轴向",
        ("n2_z_pair_rand12", "diag_n2"): "双磁体\n反向轴向",
        ("n3_mixed_rand20", "diag_rand20"): "三磁体\n随机强电流",
    }

    rows: list[dict[str, object]] = []
    with (SENS_DIR / "case_summary.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["case"], row["repeat"])
            rows.append(
                {
                    "label": scene_labels.get(key, row["case"]),
                    "publishable": str(row["publishable_gate"]).strip().lower() == "true",
                    "stationarity_ratio": float(row["baseline_stationarity_norm"]) / 1.0e-5,
                    "linearization_ratio": float(row["max_linearization_error"]) / 0.05,
                    "branch_ratio": float(row["branch_rms_max"]) / 1.0e-3,
                    "load_ratio": max(
                        float(row["baseline_slack_force_ratio"]),
                        float(row["max_perturb_slack_force_ratio"]),
                    )
                    / 2.0,
                }
            )

    x = np.arange(len(rows), dtype=float)
    labels = [str(row["label"]) for row in rows]
    series = [
        ("一阶最优性", "stationarity_ratio", "#0072B2", "o"),
        ("局部线性化", "linearization_ratio", "#D55E00", "s"),
        ("分支漂移", "branch_ratio", "#009E73", "^"),
        ("载荷传播", "load_ratio", "#CC79A7", "D"),
    ]

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 13.5,
            "axes.linewidth": 1.15,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    handles: list[object] = []
    for name, key, color, marker in series:
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        xx = np.linspace(x.min(), x.max(), 320)
        fit = PchipInterpolator(x, np.log10(np.clip(y, 1.0e-12, None)))
        yy = 10.0 ** fit(xx)
        (line,) = ax.plot(xx, yy, color=color, lw=2.6, solid_capstyle="round", label=name)
        ax.scatter(x, y, marker=marker, s=80, facecolor="white", edgecolor=color, linewidth=2.0, zorder=3)
        handles.append(line)

    gate = ax.axhline(1.0, color="#555555", lw=1.6, ls=(0, (4, 3)), alpha=0.68, label="归一化判据")
    handles.append(gate)

    for idx, row in enumerate(rows):
        status = "满足" if bool(row["publishable"]) else "未满足"
        color = "#2e7d32" if bool(row["publishable"]) else "#b35c00"
        ax.text(idx, 2.5e2, status, ha="center", va="center", color=color, fontsize=13, fontproperties=cjk_prop)

    ax.set_yscale("log")
    ax.set_ylim(1.0e-7, 5.5e5)
    ax.set_xlim(-0.25, len(rows) - 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=cjk_prop)
    ax.set_ylabel("指标/判据", fontproperties=cjk_prop)
    ax.grid(True, which="major", axis="y", color="#b7b7b7", alpha=0.34, linewidth=0.9)
    ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=0.15, linewidth=0.5)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=9))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.8, width=1.1)
    _apply_latin_ticks(ax, latin_prop)
    ax.set_xticklabels(labels, fontproperties=cjk_prop)
    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.15), prop=cjk_prop)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.095, right=0.99)

    out_base = SENS_DIR / "fig_322_sensitivity_reliability_metrics"
    _save_all(fig, out_base)
    return out_base.with_suffix(".png")


def _axis_equal_2d(ax: plt.Axes) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)


def plot_single_layer_roundtrip_pose(cjk_prop: font_manager.FontProperties, latin_prop: font_manager.FontProperties) -> Path:
    cfg = json.loads((SINGLE_DIR / "suite_config.json").read_text(encoding="utf-8"))
    cases = [case for case in cfg["cases"] if not bool(case.get("appendix_only", False))]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "axes.linewidth": 1.05,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1))
    axes_flat = axes.ravel()
    colors = {
        "ik_centerline": "0.18",
        "state_inherited_centerline": "#0072B2",
        "cold_start_centerline": "#D55E00",
    }
    labels = {
        "ik_centerline": "IK终态",
        "state_inherited_centerline": "状态继承FK",
        "cold_start_centerline": "冷启动FK",
    }
    linestyles = {
        "ik_centerline": "-",
        "state_inherited_centerline": "-",
        "cold_start_centerline": "--",
    }

    legend_handles = None
    for ax, case in zip(axes_flat, cases):
        run_dir = SINGLE_DIR / case["name"]
        cl_path = run_dir / "fk_roundtrip_centerlines.npz"
        state_path = run_dir / "state_final.npz"
        if not cl_path.exists() or not state_path.exists():
            ax.set_visible(False)
            continue
        cls = np.load(cl_path)
        state = np.load(state_path)
        target = state["p_des"] * 1e3
        for key in ("ik_centerline", "state_inherited_centerline", "cold_start_centerline"):
            if key in cls.files:
                c = cls[key] * 1e3
                ax.plot(c[:, 0], c[:, 2], lw=2.4, color=colors[key], linestyle=linestyles[key], label=labels[key])
        ax.scatter(target[0], target[2], marker="*", s=130, color="#C51B29", edgecolor="white", linewidth=0.6, label="目标点", zorder=4)
        title = str(case["label"]).replace("T1", r"$\mathrm{T1}$").replace("T2", r"$\mathrm{T2}$").replace("T4", r"$\mathrm{T4}$")
        ax.set_title(title, fontproperties=cjk_prop, pad=6)
        ax.set_xlabel(r"$x$ (mm)", fontproperties=latin_prop)
        ax.set_ylabel(r"$z$ (mm)", fontproperties=latin_prop)
        ax.grid(True, color="#c9c9c9", alpha=0.35, linewidth=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4.0, width=0.95)
        _apply_latin_ticks(ax, latin_prop)
        _axis_equal_2d(ax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(cases) :]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            prop=cjk_prop,
            handlelength=2.2,
        )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.985, hspace=0.34, wspace=0.25)
    out_base = SINGLE_DIR / "fig_strict_fk_roundtrip_pose"
    _save_all(fig, out_base)
    return out_base.with_suffix(".png")


def main() -> None:
    cjk_prop, latin_prop = setup_report_fonts()
    outputs = [
        plot_layered_residuals(cjk_prop, latin_prop),
        plot_sensitivity_reliability(cjk_prop, latin_prop),
        plot_single_layer_roundtrip_pose(cjk_prop, latin_prop),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
