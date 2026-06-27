from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "case_summary.csv"
SOURCE_HAN_SERIF = ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"


SCENE_LABELS = {
    ("n3_mixed_base_i4", "rep_a"): "三磁体混合\n低电流-A",
    ("n3_mixed_base_i4", "rep_b"): "三磁体混合\n低电流-B",
    ("n1_tip_z_rand15", "diag_n1"): "末端单磁体\n轴向",
    ("n2_z_pair_rand12", "diag_n2"): "双磁体\n反向轴向",
    ("n3_mixed_rand20", "diag_rand20"): "三磁体\n随机强电流",
}


def _available_font_family(preferred: list[str], fallback: str) -> str:
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return fallback


def _setup_fonts() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    if SOURCE_HAN_SERIF.exists():
        font_manager.fontManager.addfont(str(SOURCE_HAN_SERIF))
        zh_family = "Source Han Serif CN"
        cjk_prop = font_manager.FontProperties(fname=str(SOURCE_HAN_SERIF))
    else:
        zh_family = _available_font_family(["SimSun", "Songti SC", "Noto Serif CJK SC", "STSong"], "DejaVu Serif")
        cjk_prop = font_manager.FontProperties(family=[zh_family])
    en_family = _available_font_family(["Times New Roman", "Liberation Serif"], "DejaVu Serif")
    latin_prop = font_manager.FontProperties(family=[en_family])
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [en_family, zh_family, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return cjk_prop, latin_prop


def _bool(text: str) -> bool:
    return str(text).strip().lower() == "true"


def _load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["case"], row["repeat"])
            rows.append(
                {
                    "label": SCENE_LABELS.get(key, row["case"]),
                    "magnet_count": int(row["magnet_count"]),
                    "eps": 0.01,
                    "publishable": _bool(row["publishable_gate"]),
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
    return rows


def main() -> None:
    cjk_prop, latin_prop = _setup_fonts()
    plt.rcParams.update(
        {
            "font.size": 11.5,
            "axes.labelsize": 13,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.05,
            "mathtext.fontset": "stix",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = _load_rows()
    x = np.arange(len(rows), dtype=float)
    labels = [str(row["label"]) for row in rows]

    series = [
        ("一阶最优性", "stationarity_ratio", "#0072B2", "o"),
        ("局部线性化", "linearization_ratio", "#D55E00", "s"),
        ("分支漂移", "branch_ratio", "#009E73", "^"),
        ("载荷传播", "load_ratio", "#CC79A7", "D"),
    ]

    fig, ax = plt.subplots(figsize=(8.7, 4.15))
    handles = []
    for name, key, color, marker in series:
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        xx = np.linspace(x.min(), x.max(), 320)
        fit = PchipInterpolator(x, np.log10(np.clip(y, 1.0e-12, None)))
        yy = 10.0 ** fit(xx)
        (line,) = ax.plot(xx, yy, color=color, lw=2.15, solid_capstyle="round", label=name)
        ax.scatter(
            x,
            y,
            marker=marker,
            s=54,
            facecolor="white",
            edgecolor=color,
            linewidth=1.65,
            zorder=3,
        )
        handles.append(line)

    gate = ax.axhline(1.0, color="#555555", lw=1.25, ls=(0, (4, 3)), alpha=0.65, label="归一化判据")
    handles.append(gate)

    for idx, row in enumerate(rows):
        status = "满足" if bool(row["publishable"]) else "未满足"
        color = "#2e7d32" if bool(row["publishable"]) else "#b35c00"
        ax.text(
            idx,
            2.5e2,
            status,
            ha="center",
            va="center",
            color=color,
            fontsize=10.5,
            fontproperties=cjk_prop,
        )

    ax.set_yscale("log")
    ax.set_ylim(1.0e-7, 5.5e5)
    ax.set_xlim(-0.25, len(rows) - 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=cjk_prop)
    ax.set_ylabel("指标/判据", fontproperties=cjk_prop)
    ax.grid(True, which="major", axis="y", color="#b7b7b7", alpha=0.34, linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=0.15, linewidth=0.5)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=9))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=80))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.2, width=1.0)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(latin_prop)

    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.16), prop=cjk_prop)
    fig.subplots_adjust(top=0.78, bottom=0.19, left=0.095, right=0.99)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_sensitivity_reliability_metrics.{suffix}", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
