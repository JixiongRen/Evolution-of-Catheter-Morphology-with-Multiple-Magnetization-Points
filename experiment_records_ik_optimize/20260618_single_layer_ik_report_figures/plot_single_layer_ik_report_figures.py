from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
SOURCE_HAN_SERIF = ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"


def _available_font_family(preferred: list[str], fallback: str) -> str:
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return fallback


def setup_fonts() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
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
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return cjk_prop, latin_prop


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_trace(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def final_diag(path: str) -> dict:
    return read_json(path)["diag"]


def metric_summary(trace_path: str, result_path: str) -> dict[str, float]:
    trace = read_trace(trace_path)
    diag = final_diag(result_path)
    low_rho = sum((row.get("rho") is not None and row["rho"] < 0.25) for row in trace) / max(len(trace), 1)
    mean_backtrack = sum(row.get("backtrack_count", 0.0) for row in trace) / max(len(trace), 1)
    return {
        "pos_err_mm": float(diag["pos_err_norm_mm"]),
        "norm_E": float(diag["norm_E_raw"]),
        "smooth": float(diag.get("smooth_metric_raw", 0.0)),
        "low_rho_ratio": float(low_rho),
        "mean_backtrack": float(mean_backtrack),
        "normR": float(read_json(result_path)["normR"]),
        "norm_r_E": float(diag.get("norm_r_E", np.nan)),
    }


def set_common(ax, cjk_prop, latin_prop) -> None:
    ax.grid(True, which="major", color="#b7b7b7", alpha=0.30, linewidth=0.8)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4.0, width=1.0)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(cjk_prop)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(latin_prop)


def plot_nomag_and_geometry(cjk_prop, latin_prop) -> None:
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.05,
        }
    )

    traces = {
        "原始单阶段": read_trace("ik_out/smoke_baseline/trace.jsonl"),
        "分阶段约束": read_trace("ik_out/smoke_v1/trace.jsonl"),
    }
    metrics = {
        "原始单阶段": metric_summary("ik_out/smoke_baseline/trace.jsonl", "ik_out/smoke_baseline/result.json"),
        "分阶段约束": metric_summary("ik_out/smoke_v1/trace.jsonl", "ik_out/smoke_v1/result.json"),
    }
    geom = {
        "旧几何边界": metric_summary(
            "ik_out/joint_ik_experiments_20260316_boundsfix/legacy_geometry/trace.jsonl",
            "ik_out/joint_ik_experiments_20260316_boundsfix/legacy_geometry/result.json",
        ),
        "几何边界修正": metric_summary(
            "ik_out/joint_ik_experiments_20260316_boundsfix/fixed_geometry/trace.jsonl",
            "ik_out/joint_ik_experiments_20260316_boundsfix/fixed_geometry/result.json",
        ),
        "平衡热启动修正": metric_summary(
            "ik_out/joint_ik_experiments_20260316_boundsfix/fixed_equilibrium/trace.jsonl",
            "ik_out/joint_ik_experiments_20260316_boundsfix/fixed_equilibrium/result.json",
        ),
    }

    colors = {"原始单阶段": "#0072B2", "分阶段约束": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for label, rows in traces.items():
        ax.plot([r["iter"] for r in rows], [r["dist_to_target_mm"] for r in rows], lw=2.2, color=colors[label], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("迭代步", fontproperties=cjk_prop)
    ax.set_ylabel("末端误差 (mm)", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[0, 1]
    groups = ["末端误差", "平衡残差", "平滑指标"]
    vals = np.array(
        [
            [metrics["原始单阶段"]["pos_err_mm"], metrics["原始单阶段"]["norm_E"], metrics["原始单阶段"]["smooth"]],
            [metrics["分阶段约束"]["pos_err_mm"], metrics["分阶段约束"]["norm_E"], metrics["分阶段约束"]["smooth"]],
        ]
    )
    x = np.arange(len(groups))
    width = 0.36
    ax.bar(x - width / 2, vals[0], width=width, color=colors["原始单阶段"], alpha=0.85, label="原始单阶段")
    ax.bar(x + width / 2, vals[1], width=width, color=colors["分阶段约束"], alpha=0.85, label="分阶段约束")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontproperties=cjk_prop)
    ax.set_ylabel("指标值", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 0]
    groups = ["低质量步比例", "平均回溯次数"]
    vals = np.array(
        [
            [metrics["原始单阶段"]["low_rho_ratio"], metrics["原始单阶段"]["mean_backtrack"]],
            [metrics["分阶段约束"]["low_rho_ratio"], metrics["分阶段约束"]["mean_backtrack"]],
        ]
    )
    x = np.arange(len(groups))
    ax.bar(x - width / 2, vals[0], width=width, color=colors["原始单阶段"], alpha=0.85)
    ax.bar(x + width / 2, vals[1], width=width, color=colors["分阶段约束"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontproperties=cjk_prop)
    ax.set_ylabel("指标值", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 1]
    labels = list(geom.keys())
    x = np.arange(len(labels))
    pos = [geom[label]["pos_err_mm"] for label in labels]
    rE = [geom[label]["norm_r_E"] for label in labels]
    low = [geom[label]["low_rho_ratio"] for label in labels]
    ax.plot(x, pos, marker="o", ms=7, lw=2.2, color="#0072B2", label="末端误差/mm")
    ax.plot(x, rE, marker="s", ms=7, lw=2.2, color="#D55E00", label="||r_E||")
    ax.plot(x, low, marker="^", ms=7, lw=2.2, color="#009E73", label="低质量步比例")
    ax.set_xticks(x)
    ax.set_xticklabels(["旧几何\n边界", "几何边界\n修正", "平衡热启动\n修正"], fontproperties=cjk_prop)
    ax.set_ylabel("指标值", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop, loc="upper right")
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_single_layer_nomag_geometry.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _max_neighbor_jump_mm(npz_path: str) -> float:
    c = np.load(ROOT / npz_path)["centerline"]
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).max() * 1000.0)


def _trace_array(trace_path: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_trace(trace_path)
    it = np.asarray([float(r["iter"]) for r in rows], dtype=float)
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return it, vals


def _final_centerline(run_dir: str) -> np.ndarray:
    return np.load(ROOT / run_dir / "poses" / "final.npz")["centerline"]


def _resample_polyline(points: np.ndarray, n: int = 200) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    grid = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(grid, s, pts[:, j]) for j in range(3)])


def plot_control_evolution(cjk_prop, latin_prop) -> None:
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.05,
        }
    )
    nomag = {
        "原始单阶段": {"trace": "ik_out/smoke_baseline/trace.jsonl", "color": "#0072B2"},
        "分阶段约束": {"trace": "ik_out/smoke_v1/trace.jsonl", "color": "#D55E00"},
    }
    magnetic = {
        "0号线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_c0/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_c0/result.json",
            "color": "#0072B2",
        },
        "4号线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_c4/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_c4/result.json",
            "color": "#D55E00",
        },
        "多线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_auto_init/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_auto_init/result.json",
            "color": "#009E73",
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    for label, spec in nomag.items():
        it, x_m = _trace_array(spec["trace"], "x_m")
        ax.plot(it, x_m * 1000.0, lw=2.4, color=spec["color"], label=label)
    ax.set_xlabel("迭代步", fontproperties=cjk_prop)
    ax.set_ylabel("插入深度 x (mm)", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[0, 1]
    for label, spec in magnetic.items():
        it, x_m = _trace_array(spec["trace"], "x_m")
        ax.plot(it, x_m * 1000.0, lw=2.4, color=spec["color"], label=label)
    ax.axhline(69.001, color="0.35", ls="--", lw=1.4, label="可行下界")
    ax.set_xlabel("迭代步", fontproperties=cjk_prop)
    ax.set_ylabel("插入深度 x (mm)", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop, ncols=2)
    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 0]
    selected = [
        ("0号线圈先验", 0, "#0072B2", "0号线圈先验 I0"),
        ("4号线圈先验", 4, "#D55E00", "4号线圈先验 I4"),
        ("多线圈先验", 2, "#009E73", "多线圈先验 I2"),
        ("多线圈先验", 6, "#CC79A7", "多线圈先验 I6"),
    ]
    for label, idx, color, text in selected:
        it, I = _trace_array(magnetic[label]["trace"], "I_A")
        ax.plot(it, I[:, idx], lw=2.2, color=color, label=text)
    ax.set_xlabel("迭代步", fontproperties=cjk_prop)
    ax.set_ylabel("主导线圈电流 (A)", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop, ncols=2)
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 1]
    labels = list(magnetic.keys())
    final_I = np.asarray([read_json(magnetic[label]["result"])["I_A"] for label in labels], dtype=float)
    im = ax.imshow(final_I, cmap="RdBu_r", aspect="auto", vmin=-15, vmax=15)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(["0号线圈\n先验", "4号线圈\n先验", "多线圈\n先验"], fontproperties=cjk_prop)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(8)], fontproperties=latin_prop)
    ax.set_xlabel("线圈编号", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("最终电流 (A)", fontproperties=cjk_prop)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(latin_prop)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_single_layer_control_evolution.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_fk_consistency_evidence(cjk_prop, latin_prop) -> None:
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.05,
        }
    )
    cases = {
        "0号线圈先验": {
            "dir": "ik_out/diag_mag_realistic_t2_c0",
            "result": "ik_out/diag_mag_realistic_t2_c0/result.json",
            "color": "#0072B2",
        },
        "4号线圈先验": {
            "dir": "ik_out/diag_mag_realistic_t2_c4",
            "result": "ik_out/diag_mag_realistic_t2_c4/result.json",
            "color": "#D55E00",
        },
        "多线圈先验": {
            "dir": "ik_out/diag_mag_realistic_t2_auto_init",
            "result": "ik_out/diag_mag_realistic_t2_auto_init/result.json",
            "color": "#009E73",
        },
    }
    target = np.asarray(read_json("ik_out/diag_mag_realistic_t2_c0/result.json")["p_des_m"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    for label, spec in cases.items():
        c = _final_centerline(spec["dir"]) * 1000.0
        ax.plot(c[:, 0], c[:, 2], lw=2.4, color=spec["color"], label=label)
        ax.scatter(c[-1, 0], c[-1, 2], s=38, color=spec["color"], edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(target[0] * 1000.0, target[2] * 1000.0, marker="*", s=120, color="#C51B29", label="目标点", zorder=4)
    ax.set_xlabel("x (mm)", fontproperties=latin_prop)
    ax.set_ylabel("z (mm)", fontproperties=latin_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[0, 1]
    labels = list(cases.keys())
    tip_err = [final_diag(cases[label]["result"])["pos_err_norm_mm"] for label in labels]
    norm_E = [final_diag(cases[label]["result"])["norm_E_raw"] for label in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, tip_err, width=0.36, color="#0072B2", alpha=0.84, label="末端误差/mm")
    ax.bar(x + 0.18, norm_E, width=0.36, color="#D55E00", alpha=0.84, label="FK平衡残差")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0号线圈\n先验", "4号线圈\n先验", "多线圈\n先验"], fontproperties=cjk_prop)
    ax.set_ylabel("指标值", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 0]
    base = _resample_polyline(_final_centerline("ik_out/diag_mag_realistic_t2_c0"))
    drift = []
    for label in labels:
        c = _resample_polyline(_final_centerline(cases[label]["dir"]))
        drift.append(float(np.sqrt(np.mean(np.sum((c - base) ** 2, axis=1))) * 1000.0))
    ax.bar(np.arange(len(labels)), drift, color=[cases[label]["color"] for label in labels], alpha=0.84)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["0号线圈\n先验", "4号线圈\n先验", "多线圈\n先验"], fontproperties=cjk_prop)
    ax.set_ylabel("相对可收敛分支形态 RMS (mm)", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 1]
    jump_original = _max_neighbor_jump_mm("ik_out/diag_mag_realistic_t2_c4/poses/final.npz")
    gap_constrained = final_diag("ik_out/diag_mag_realistic_t2_c4_phys_v1/result.json")["max_interface_gap_mm"]
    gap_auto = final_diag("ik_out/diag_mag_realistic_t2_auto_init/result.json")["max_interface_gap_mm"]
    labels_gap = ["4号线圈\n原始中心线跳距", "接口约束\n诊断案例", "多线圈先验\n接口间隙"]
    vals_gap = [jump_original, gap_constrained, gap_auto]
    ax.bar(np.arange(3), vals_gap, color=["#D55E00", "#CC79A7", "#009E73"], alpha=0.84)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels_gap, fontproperties=cjk_prop)
    ax.set_ylabel("间隙或跳距 (mm)", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13.5)
    set_common(ax, cjk_prop, latin_prop)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_single_layer_fk_consistency.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_magnetic_diagnostics(cjk_prop, latin_prop) -> None:
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.linewidth": 1.05,
        }
    )
    cases = {
        "0号线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_c0/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_c0/result.json",
            "color": "#0072B2",
        },
        "4号线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_c4/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_c4/result.json",
            "color": "#D55E00",
        },
        "多线圈先验": {
            "trace": "ik_out/diag_mag_realistic_t2_auto_init/trace.jsonl",
            "result": "ik_out/diag_mag_realistic_t2_auto_init/result.json",
            "color": "#009E73",
        },
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    for label, spec in cases.items():
        rows = read_trace(spec["trace"])
        ax.plot([r["iter"] for r in rows], [r["dist_to_target_mm"] for r in rows], lw=2.2, color=spec["color"], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("迭代步", fontproperties=cjk_prop)
    ax.set_ylabel("末端误差 (mm)", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[0, 1]
    labels = list(cases.keys())
    x = np.arange(len(labels))
    err = [final_diag(cases[label]["result"])["pos_err_norm_mm"] for label in labels]
    normE = [final_diag(cases[label]["result"])["norm_E_raw"] for label in labels]
    ax.bar(x - 0.18, err, width=0.36, color="#0072B2", alpha=0.82, label="末端误差/mm")
    ax.bar(x + 0.18, normE, width=0.36, color="#D55E00", alpha=0.82, label="||E||")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["0号线圈\n先验", "4号线圈\n先验", "多线圈\n先验"], fontproperties=cjk_prop)
    ax.set_ylabel("最终指标", fontproperties=cjk_prop)
    ax.legend(frameon=False, prop=cjk_prop)
    ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    ax = axes[1, 0]
    current = np.array([read_json(cases[label]["result"])["I_A"] for label in labels], dtype=float)
    im = ax.imshow(current, cmap="RdBu_r", aspect="auto", vmin=-15, vmax=15)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(["0号线圈\n先验", "4号线圈\n先验", "多线圈\n先验"], fontproperties=cjk_prop)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(8)], fontproperties=latin_prop)
    ax.set_xlabel("线圈编号", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("电流 (A)", fontproperties=cjk_prop)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(latin_prop)

    ax = axes[1, 1]
    jump_original = _max_neighbor_jump_mm("ik_out/diag_mag_realistic_t2_c4/poses/final.npz")
    gap_constrained = final_diag("ik_out/diag_mag_realistic_t2_c4_phys_v1/result.json")["max_interface_gap_mm"]
    gap_auto = final_diag("ik_out/diag_mag_realistic_t2_auto_init/result.json")["max_interface_gap_mm"]
    labels_gap = ["4号线圈\n原始中心线跳距", "接口约束\n诊断案例", "多线圈先验\n接口间隙"]
    vals_gap = [jump_original, gap_constrained, gap_auto]
    ax.bar(np.arange(3), vals_gap, color=["#D55E00", "#CC79A7", "#009E73"], alpha=0.84)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels_gap, fontproperties=cjk_prop)
    ax.set_ylabel("间隙或跳距 (mm)", fontproperties=cjk_prop)
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes, fontproperties=latin_prop, fontsize=13)
    set_common(ax, cjk_prop, latin_prop)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_322_single_layer_magnetic_diagnostics.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cjk_prop, latin_prop = setup_fonts()
    plot_nomag_and_geometry(cjk_prop, latin_prop)
    plot_control_evolution(cjk_prop, latin_prop)
    plot_fk_consistency_evidence(cjk_prop, latin_prop)
    plot_magnetic_diagnostics(cjk_prop, latin_prop)


if __name__ == "__main__":
    main()
