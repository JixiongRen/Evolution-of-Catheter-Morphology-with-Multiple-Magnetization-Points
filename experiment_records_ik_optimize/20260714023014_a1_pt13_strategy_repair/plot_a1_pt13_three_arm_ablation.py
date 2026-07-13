"""Plot the A1-pt13 three-arm ablation summary."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


EXP_DIR = Path(__file__).resolve().parent
FONT = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def main() -> None:
    rows = []
    with (EXP_DIR / "ablation_summary.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    labels = [row["arm"].replace("_", "\n") for row in rows]
    errors = [float(row["err_true_mm"]) for row in rows]
    colors = ["#5b8ff9" if err > 5.0 else "#37a354" for err in errors]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, errors, color=colors, edgecolor="#1f2d3d", linewidth=0.8)
    ax.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.6, label="5 mm 成功阈值")
    ax.axhline(5.778239956594556, color="#8c564b", linestyle=":", linewidth=1.6, label="源近解误差")
    for bar, value in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=FONT,
            fontsize=10,
        )
    ax.set_title("A1-pt13 同分支三臂精修消融", fontproperties=FONT, fontsize=15)
    ax.set_ylabel("真实末端误差 (mm)", fontproperties=FONT, fontsize=12)
    ax.set_ylim(4.6, 5.95)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(prop=FONT, loc="upper right")
    ax.tick_params(axis="x", labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "ablation_error_summary.png", dpi=220)
    fig.savefig(EXP_DIR / "ablation_error_summary.svg")


if __name__ == "__main__":
    main()
