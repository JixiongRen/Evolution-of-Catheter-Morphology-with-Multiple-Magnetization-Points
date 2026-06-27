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
CSV_PATH = OUT_DIR / "summary.csv"
SOURCE_HAN_SERIF = ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"


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


def _load_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
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
    return rows


def main() -> None:
    cjk_prop, latin_prop = _setup_fonts()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
            "mathtext.fontset": "stix",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = _load_rows()
    alphas = sorted({row["alpha"] for row in rows})

    series = [
        ("任务残差", "任务残差判据", "task", "#1f77b4", "o", 1.0e-2),
        ("末端边界残差", "末端边界判据", "boundary", "#d55e00", "s", 2.0e-1),
        ("段间连续残差", "段间连续判据", "continuity", "#009e73", "^", 1.0e-3),
    ]

    fig, axes = plt.subplots(1, len(alphas), figsize=(9.2, 3.85), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    handles = []
    labels = []
    for ax, alpha in zip(axes, alphas):
        sub = sorted([row for row in rows if row["alpha"] == alpha], key=lambda r: r["lambda_t"])
        x = np.asarray([row["lambda_t"] for row in sub], dtype=float)
        x_fit = np.linspace(float(x.min()), float(x.max()), 260)
        for name, threshold_name, key, color, marker, threshold in series:
            y = np.asarray([row[key] for row in sub], dtype=float)
            # Shape-preserving fit in log space. It gives a smooth trend line
            # without the overshoot that high-order polynomials create here.
            fit = PchipInterpolator(x, np.log10(y))
            y_fit = 10.0 ** fit(x_fit)
            (line,) = ax.plot(
                x_fit,
                y_fit,
                color=color,
                linewidth=2.45,
                solid_capstyle="round",
                label=name,
            )
            ax.scatter(
                x,
                y,
                color="white",
                edgecolor=color,
                marker=marker,
                s=58,
                linewidth=1.8,
                zorder=3,
            )
            hline = ax.axhline(
                threshold,
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=1.25,
                alpha=0.42,
                zorder=0,
                label=threshold_name,
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(name)
                handles.append(hline)
                labels.append(threshold_name)

        ax.set_yscale("log")
        ax.set_ylim(4.0e-5, 2.2)
        ax.set_xlim(-0.035, 1.035)
        ax.set_xticks([0.0, 0.1, 0.3, 1.0])
        ax.set_xlabel(r"任务强度 $\lambda_t$", fontproperties=cjk_prop)
        ax.set_title(rf"$\alpha={alpha:.2f}$")
        ax.grid(True, which="major", color="#b7b7b7", alpha=0.34, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=0.18, linewidth=0.55)
        ax.tick_params(direction="out", length=4.5, width=1.0)
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
            fontsize=10.5,
            color="#555555",
            fontproperties=cjk_prop,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("残差范数", fontproperties=cjk_prop)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.13),
        handlelength=2.2,
        columnspacing=1.8,
        prop=cjk_prop,
    )
    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(latin_prop)
    fig.subplots_adjust(top=0.74, left=0.09, right=0.99, bottom=0.18, wspace=0.12)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_layered_shooting_residuals.{suffix}", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
