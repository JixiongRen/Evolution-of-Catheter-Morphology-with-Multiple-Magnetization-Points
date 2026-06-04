"""Visualize P2c shooting sensitivity audit results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SRC = ROOT / "experiment_records_ik_optimize"

P2C9F = SRC / "20260604141002_p2c9f_exact_stationarity_ift_probe" / "summary.json"
P2C9G = SRC / "20260604145803_p2c9g_exact_ift_shadow_replay" / "summary.json"
P2C9I = SRC / "20260604162320_p2c9i_branch_roundtrip_gate" / "summary.json"
P2C9J = SRC / "20260604164000_p2c9j_target_current_continuation" / "summary.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def short_case(name: str) -> str:
    mapping = {
        "n3_mixed_base_i4": "n3_base_i4",
        "n3_mixed_rand20": "n3_rand20",
        "n2_z_pair_rand12": "n2_rand12",
        "n1_tip_z_rand15": "n1_rand15",
    }
    return mapping.get(name, name)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180)
    plt.close(fig)


def plot_exact_vs_gn(summary: dict) -> None:
    rows = summary["direction_rows"]
    labels = [f"{short_case(r['case'])}\n{r['direction'].replace('active_coil_', 'coil')}" for r in rows]
    x = np.arange(len(rows))
    exact = [r["median_rel_err_exact"] for r in rows]
    gn = [r["median_rel_err_gn"] for r in rows]
    usable = [r["direction_usable"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.18, gn, width=0.36, label="GN chain", color="#9aa0a6")
    ax.bar(x + 0.18, exact, width=0.36, label="Exact stationarity IFT", color="#2671b9")
    for i, ok in enumerate(usable):
        ax.scatter(i, 0.31, marker="v", s=70, color="#188038" if ok else "#c5221f", zorder=5)
    ax.axhline(0.3, color="#c5221f", linestyle="--", linewidth=1.5, label="usable rel_err gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("median relative error")
    ax.set_title("P2c9f: Exact IFT vs GN chain")
    ax.set_ylim(0.0, max(0.5, max(gn + exact) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "p2c9f_exact_vs_gn_rel_error.png")


def plot_shadow(summary: dict) -> None:
    rows = summary["rows"]
    labels = [f"{short_case(r['case'])}\n{r['direction'].replace('active_coil_', 'coil')}" for r in rows]
    x = np.arange(len(rows))
    rel = [r["rel_err"] for r in rows]
    cosine = [r["cosine"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x, rel, color="#188038", alpha=0.85, label="relative error")
    ax1.axhline(0.3, color="#c5221f", linestyle="--", linewidth=1.5, label="usable rel_err gate")
    ax1.set_ylabel("predicted vs actual rel_err")
    ax1.set_ylim(0.0, 0.35)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, cosine, color="#f29900", marker="o", linewidth=2, label="cosine")
    ax2.set_ylabel("direction cosine")
    ax2.set_ylim(0.98, 1.001)
    ax1.set_title("P2c9g: Exact IFT shadow replay")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    save(fig, "p2c9g_shadow_replay.png")


def plot_branch(summary: dict) -> None:
    case_rows = summary["case_rows"]
    roundtrip = {r["case"]: r for r in summary["roundtrip_rows"]}
    labels = [short_case(r["case"]) for r in case_rows]
    x = np.arange(len(case_rows))
    task_spread = [r["branch_task_spread"] for r in case_rows]
    state_spread = [r["branch_state_spread"] for r in case_rows]
    rt_task = [roundtrip.get(r["case"], {}).get("roundtrip_task_error", np.nan) for r in case_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - 0.18, task_spread, width=0.36, label="branch task spread", color="#2671b9")
    axes[0].bar(x + 0.18, rt_task, width=0.36, label="roundtrip task error", color="#188038")
    axes[0].set_yscale("symlog", linthresh=1e-12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("task-space error")
    axes[0].set_title("Task branch / roundtrip error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    colors = ["#188038" if r["branch_consistent"] else "#c5221f" for r in case_rows]
    axes[1].bar(x, state_spread, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("max state spread")
    axes[1].set_title("Branch state spread")
    axes[1].grid(axis="y", alpha=0.25)
    save(fig, "p2c9i_branch_roundtrip.png")


def plot_continuation(summary: dict) -> None:
    rows = summary["per_alpha"]
    alpha = np.array([r["alpha"] for r in rows], dtype=float)
    residual = np.array([r["best_residual_norm"] for r in rows], dtype=float)
    stationarity = np.array([r["best_stationarity_norm"] for r in rows], dtype=float)
    boundary = np.array([r["best_boundary_residual_norm"] for r in rows], dtype=float)
    linear_ok = [r["any_linearizable"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alpha, residual, marker="o", label="residual norm", color="#2671b9")
    ax.plot(alpha, boundary, marker="o", label="boundary residual", color="#188038")
    ax.plot(alpha, stationarity, marker="o", label="stationarity norm", color="#c5221f")
    ax.axvline(summary["first_linearization_failure_alpha"], color="#c5221f", linestyle="--", label="linearization break")
    ax.axvline(summary["first_failure_alpha"], color="#5f6368", linestyle=":", label="solve failure")
    for a, ok in zip(alpha, linear_ok):
        ax.scatter([a], [1e-3], marker="s", s=70, color="#188038" if ok else "#c5221f", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("continuation alpha")
    ax.set_ylabel("gate metrics, log scale")
    ax.set_title("P2c9j: current/target continuation to hard case")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save(fig, "p2c9j_continuation_break.png")


def plot_dashboard(p2c9f: dict, p2c9g: dict, p2c9i: dict, p2c9j: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(["GN", "Exact IFT"], [p2c9f["median_rel_err_gn"], p2c9f["median_rel_err_exact"]], color=["#9aa0a6", "#2671b9"])
    axes[0, 0].axhline(0.3, color="#c5221f", linestyle="--")
    axes[0, 0].set_title("Sensitivity audit median rel_err")
    axes[0, 0].set_ylabel("relative error")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(["shadow rel_err", "shadow cosine"], [p2c9g["median_rel_err"], p2c9g["median_cosine"]], color=["#188038", "#f29900"])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Shadow replay summary")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(["branch ok", "roundtrip ok", "case total"], [p2c9i["branch_consistent_count"], p2c9i["roundtrip_consistent_count"], p2c9i["case_count"]], color=["#188038", "#2671b9", "#9aa0a6"])
    axes[1, 0].set_title("Branch gate count")
    axes[1, 0].grid(axis="y", alpha=0.25)

    alpha = [r["alpha"] for r in p2c9j["per_alpha"]]
    stationarity = [r["best_stationarity_norm"] for r in p2c9j["per_alpha"]]
    axes[1, 1].plot(alpha, stationarity, marker="o", color="#c5221f")
    axes[1, 1].axvline(p2c9j["first_linearization_failure_alpha"], color="#c5221f", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Hard path stationarity")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].grid(True, which="both", alpha=0.25)

    save(fig, "p2c_sensitivity_dashboard.png")


def main() -> None:
    p2c9f = load_json(P2C9F)
    p2c9g = load_json(P2C9G)
    p2c9i = load_json(P2C9I)
    p2c9j = load_json(P2C9J)
    plot_exact_vs_gn(p2c9f)
    plot_shadow(p2c9g)
    plot_branch(p2c9i)
    plot_continuation(p2c9j)
    plot_dashboard(p2c9f, p2c9g, p2c9i, p2c9j)


if __name__ == "__main__":
    main()
