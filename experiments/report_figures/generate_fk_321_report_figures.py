"""Generate report-oriented FK figures for section 3.2.1.

The script is intentionally separate from the production FK code.  It creates
publication-style figures from existing benchmark records and optionally runs a
small set of report cases with 20-iteration morphology checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FK_OPT_DIR = REPO_ROOT / "forward_kinematics_optimized"
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

import jax
import jax.numpy as jnp
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium_solver_nondim import residual_bar, with_coil_currents, with_L1_dim
from fk_engine import ForwardKinematicsEngine
from rod_mesh_nondim import build_uniform_mesh
from utils_nondim import make_initial_guess_multi_bar_jax
from experiments.efficiency_optimization.fk_under_60s_driver import (
    _block,
    _extract_centerline,
    _get_calib_file,
    _repo_commit,
    _rss_mb,
)


DEFAULT_OUT = REPO_ROOT / "experiment_records_fk_optimize" / "20260613_fk_321_report_figures"
SOURCE_HAN_SERIF = REPO_ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"
COMPLEXITY_CSV = (
    REPO_ROOT
    / "experiment_records_fk_optimize"
    / "20260524180122_fk_M5_to_M10_current20_validation"
    / "batch_summary.csv"
)
EXISTING_STAGE_BASE = (
    REPO_ROOT
    / "experiment_records_fk_optimize"
    / "20260524180122_fk_M5_to_M10_current20_validation"
)
EXISTING_STAGE_CASES = {
    "M555_single_tip_rand_c": "单磁化点",
    "M555_double_z_rand_c": "双磁化点",
    "M555_triple_mixed_rand_c": "三磁化点",
}

MAGNETS = {
    "n1_tip_z": {
        "label": "单磁化点-末端z向",
        "m_body_list": "0,0,0; 0,0,0; 0,0,0.005301",
        "active_magnets": 1,
    },
    "n1_base_x": {
        "label": "单磁化点-基部x向",
        "m_body_list": "0.005301,0,0; 0,0,0; 0,0,0",
        "active_magnets": 1,
    },
    "n2_z_pair": {
        "label": "双磁化点-z向配对",
        "m_body_list": "0,0,-0.005301; 0,0,0.005301; 0,0,0",
        "active_magnets": 2,
    },
    "n2_xz_pair": {
        "label": "双磁化点-x/z组合",
        "m_body_list": "0.005301,0,0; 0,0,0; 0,0,0.005301",
        "active_magnets": 2,
    },
    "n3_mixed": {
        "label": "三磁化点-混合",
        "m_body_list": "0,0,-0.005301; 0,0,0.005301; 0.005301,0,0",
        "active_magnets": 3,
    },
    "n3_xyz": {
        "label": "三磁化点-xyz组合",
        "m_body_list": "0.005301,0,0; 0,0.005301,0; 0,0,0.005301",
        "active_magnets": 3,
    },
}

CURRENTS = {
    "rand_a": "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4",
    "rand_b": "-19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2",
}

M555_REPORT_CASES = [
    ("n1_tip_z_rand_a", "n1_tip_z", "rand_a"),
    ("n1_base_x_rand_b", "n1_base_x", "rand_b"),
    ("n2_z_pair_rand_a", "n2_z_pair", "rand_a"),
    ("n2_xz_pair_rand_b", "n2_xz_pair", "rand_b"),
    ("n3_xyz_rand_a", "n3_xyz", "rand_a"),
]

GRAVITY_SINGLE_MAGNETS = ("n1_tip_z", "n2_z_pair", "n3_mixed")


def available_font_family(preferred: list[str], fallback: str) -> str:
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return fallback


def parse_vector(text: str, expected: int | None = None) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}: {text!r}")
    return values


def parse_m_body_list(text: str) -> tuple[tuple[float, float, float], ...]:
    rows = []
    for row in str(text).split(";"):
        row = row.strip()
        if not row:
            continue
        x, y, z = parse_vector(row, expected=3)
        rows.append((x, y, z))
    if len(rows) != 3:
        raise ValueError("Current FK setup expects exactly three rigid magnetic segments")
    return tuple(rows)


def parse_m_list(text: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if len(values) != 3:
        raise ValueError(f"M_list must contain three integers: {text!r}")
    return values


def setup_plot_style() -> None:
    if SOURCE_HAN_SERIF.exists():
        font_manager.fontManager.addfont(str(SOURCE_HAN_SERIF))
    zh_family = available_font_family(["SimSun", "Songti SC", "Source Han Serif CN", "Noto Serif CJK SC", "STSong"], "DejaVu Serif")
    en_family = available_font_family(["Times New Roman", "Liberation Serif"], "DejaVu Serif")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [en_family, zh_family, "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def zh_font(size: float | None = None, weight: str | None = None) -> font_manager.FontProperties:
    if SOURCE_HAN_SERIF.exists():
        return font_manager.FontProperties(fname=str(SOURCE_HAN_SERIF), size=size, weight=weight)
    return font_manager.FontProperties(
        family=[available_font_family(["SimSun", "Songti SC", "Noto Serif CJK SC", "STSong"], "DejaVu Serif")],
        size=size,
        weight=weight,
    )


def en_font(size: float | None = None, weight: str | None = None) -> font_manager.FontProperties:
    return font_manager.FontProperties(
        family=[available_font_family(["Times New Roman", "Liberation Serif"], "DejaVu Serif")],
        size=size,
        weight=weight,
    )


def current_short(text: str) -> str:
    values = parse_vector(text, expected=8)
    return "[" + ", ".join(f"{value:g}" for value in values) + "] A"


def magnet_short(text: str) -> str:
    rows = parse_m_body_list(text)
    labels = []
    for idx, row in enumerate(rows, start=1):
        axis = "0"
        if abs(row[0]) > 0:
            axis = "+x" if row[0] > 0 else "-x"
        elif abs(row[1]) > 0:
            axis = "+y" if row[1] > 0 else "-y"
        elif abs(row[2]) > 0:
            axis = "+z" if row[2] > 0 else "-z"
        labels.append(f"m{idx}:{axis}")
    return "; ".join(labels)


def build_engine(
    *,
    m_body_list: str,
    m_list: str,
    max_iter: int,
    tol: float,
    enable_gravity: bool,
    enable_magnetics: bool,
    linear_solver: str,
) -> ForwardKinematicsEngine:
    calib_file, _reason = _get_calib_file()
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=parse_m_list(m_list),
        L1_min=1e-6,
        flex_d_outer=(0.0015,),
        flex_E=(1.8e6,),
        flex_G=(0.6e6,),
        flex_rho=(970.0,),
        rigid_d_outer=(0.0015,),
        rigid_rho=(7500.0,),
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=bool(enable_gravity),
        enable_magnetics=bool(enable_magnetics),
        calib_file=calib_file,
        actuation_table_pkl=str(
            REPO_ROOT
            / "supiee_auto_diff"
            / "offline_interpolation_data"
            / "actuation_tables"
            / "actuation_table.pkl"
        ),
        m_body_list=parse_m_body_list(m_body_list),
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=0.1,
        jac_method="fwd",
        linear_solver=str(linear_solver),
        damping_mode="identity",
        dense_solve_method="cholesky",
        step_norm_clip=None,
        step_scale=1.0,
        backtrack_max=0,
        backtrack_factor=0.5,
        trust_region_eta=None,
        max_damping_attempts=1,
        reject_lam_gain=10.0,
        scale_mode="optimized",
        scale_include_external=True,
        L_protrude_max=0.11,
    )


def final_norm(z_bar: Any, params: Any) -> tuple[float, float]:
    E = residual_bar(z_bar, params)
    _block(E)
    return float(jnp.linalg.norm(E)), float(jnp.max(jnp.abs(E)))


def make_initial_state(engine: ForwardKinematicsEngine, target: jnp.ndarray, *, L_protrude: float) -> tuple[Any, Any]:
    engine._ensure_static_initialized(L_protrude=float(L_protrude), coil_currents=target)
    flex_lengths = engine._compute_flex_lengths(L_protrude=float(L_protrude))
    L1_dim = float(flex_lengths[0])
    if engine._params_static_base is None or engine._meshes_ref is None:
        raise RuntimeError("FK engine static initialization failed")
    params = with_coil_currents(engine._params_static_base, target)
    params = with_L1_dim(params, jnp.asarray(L1_dim, dtype=jnp.float64))
    meshes = list(engine._meshes_ref)
    meshes[0] = build_uniform_mesh(float(L1_dim), int(engine.M_list[0]))
    z0_bar, *_ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=params.scales,
        p0_dim=engine.p0_dim,
        Q0=engine.Q0,
        axis_body=engine.axis_body,
    )
    return z0_bar, params


def save_checkpoint_npz(case_dir: Path, checkpoints: list[dict[str, Any]]) -> None:
    payload = {
        f"iter_{int(item['iter']):05d}": item["centerline"]
        for item in checkpoints
    }
    payload["iters"] = np.asarray([int(item["iter"]) for item in checkpoints], dtype=np.int32)
    payload["normE"] = np.asarray([float(item["normE"]) for item in checkpoints], dtype=np.float64)
    np.savez_compressed(case_dir / "centerline_checkpoints_every20.npz", **payload)


def run_checkpoint_case(
    *,
    out_dir: Path,
    case_name: str,
    condition_label: str,
    magnet_key: str,
    currents: str,
    m_list: str,
    tol: float,
    max_total_iter: int,
    checkpoint_iter: int,
    enable_gravity: bool,
    enable_magnetics: bool,
    linear_solver: str,
) -> dict[str, Any]:
    case_dir = out_dir / "cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    magnet = MAGNETS[magnet_key]
    target = jnp.asarray(parse_vector(currents, expected=8), dtype=jnp.float64)
    engine = build_engine(
        m_body_list=str(magnet["m_body_list"]),
        m_list=m_list,
        max_iter=0,
        tol=tol,
        enable_gravity=enable_gravity,
        enable_magnetics=enable_magnetics,
        linear_solver=linear_solver,
    )

    config = {
        "case_name": case_name,
        "condition_label": condition_label,
        "magnet_key": magnet_key,
        "magnet_label": magnet["label"],
        "active_magnets": magnet["active_magnets"],
        "coil_currents": currents,
        "M_list": m_list,
        "tol": tol,
        "max_total_iter": max_total_iter,
        "checkpoint_iter": checkpoint_iter,
        "enable_gravity": enable_gravity,
        "enable_magnetics": enable_magnetics,
        "linear_solver": linear_solver,
    }
    (case_dir / "case_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_name": case_name, "event": "start"}, ensure_ascii=False), flush=True)

    start = time.perf_counter()
    z_bar, params = make_initial_state(engine, target, L_protrude=0.11)
    engine._last_z_bar = z_bar
    engine._last_scales = params.scales
    engine._last_flex_lengths = engine._compute_flex_lengths(L_protrude=0.11)
    engine._last_rigid_lengths = engine.rigid_lengths
    _block(z_bar)
    checkpoints: list[dict[str, Any]] = [
        {
            "iter": 0,
            "normE": float("nan"),
            "max_abs_E": float("nan"),
            "centerline": _extract_centerline(z_bar, params, rigid_sample_count=10),
            "stop_reason": "initial_guess",
            "block_iter": 0,
        }
    ]

    total_iter = 0
    ok = False
    stop_reason = "not_started"
    while total_iter < int(max_total_iter):
        engine.max_iter = int(checkpoint_iter)
        engine.tol = float(tol)
        z_bar, params, _meshes, ok, stats = engine.solve_with_stats(
            coil_currents=target,
            L_protrude=0.11,
            warm_start=True,
            return_stats=True,
            stats_include_jacobian=False,
            verbose=False,
        )
        _block(z_bar)
        block_iter = 0 if stats is None else int(stats.n_iter)
        total_iter += block_iter if block_iter > 0 else int(checkpoint_iter)
        normE, max_abs = final_norm(z_bar, params)
        stop_reason = "unknown" if stats is None else str(stats.stop_reason)
        checkpoints.append(
            {
                "iter": int(total_iter),
                "normE": normE,
                "max_abs_E": max_abs,
                "centerline": _extract_centerline(z_bar, params, rigid_sample_count=10),
                "stop_reason": stop_reason,
                "block_iter": block_iter,
            }
        )
        print(
            json.dumps(
                {
                    "case_name": case_name,
                    "iter": int(total_iter),
                    "normE": normE,
                    "stop_reason": stop_reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if bool(ok) and normE < float(tol):
            break
        if block_iter == 0 or stop_reason in {"gtol", "xtol", "patience", "reject_patience", "lam_max"}:
            break

    centerline = checkpoints[-1]["centerline"]
    np.save(case_dir / "centerline.npy", centerline)
    save_checkpoint_npz(case_dir, checkpoints)

    trace_rows = [
        {
            "iter": int(item["iter"]),
            "normE": float(item["normE"]),
            "max_abs_E": float(item["max_abs_E"]),
            "stop_reason": str(item["stop_reason"]),
            "block_iter": int(item["block_iter"]),
        }
        for item in checkpoints
    ]
    with (case_dir / "convergence_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iter", "normE", "max_abs_E", "stop_reason", "block_iter"])
        writer.writeheader()
        writer.writerows(trace_rows)

    summary = {
        **config,
        "m_body_list": str(magnet["m_body_list"]),
        "ok": bool(ok and float(checkpoints[-1]["normE"]) < float(tol)),
        "stop_reason": stop_reason,
        "n_iter": int(checkpoints[-1]["iter"]),
        "wall_s": float(time.perf_counter() - start),
        "final_normE": float(checkpoints[-1]["normE"]),
        "max_abs_E": float(checkpoints[-1]["max_abs_E"]),
        "checkpoint_count": len(checkpoints),
        "centerline_shape": list(centerline.shape),
        "centerline_end_m": centerline[-1].tolist(),
        "rss_mb": _rss_mb(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "case_dir": str(case_dir)}


def set_equal_3d_axes_mm(ax: Any, points_mm: np.ndarray) -> None:
    mins = np.min(points_mm, axis=0)
    maxs = np.max(points_mm, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def plot_case_evolution(case_dir: Path, out_path: Path, title: str) -> None:
    data = np.load(case_dir / "centerline_checkpoints_every20.npz")
    iters = data["iters"]
    normE = data["normE"]
    keys = [f"iter_{int(i):05d}" for i in iters]
    centerlines = [np.asarray(data[key], dtype=np.float64) * 1000.0 for key in keys]
    all_points = np.concatenate(centerlines, axis=0)

    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("viridis")
    denom = max(len(centerlines) - 1, 1)
    tips = []
    for idx, (iteration, centerline) in enumerate(zip(iters, centerlines)):
        color = cmap(idx / denom)
        lw = 0.7
        alpha = 0.22
        label = None
        if idx == 0:
            color = "0.25"
            lw = 2.0
            alpha = 1.0
            label = "初始位姿"
        elif idx == len(centerlines) - 1:
            color = "#D55E00"
            lw = 2.3
            alpha = 1.0
            label = "收敛位姿"
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color=color, lw=lw, alpha=alpha, label=label)
        tips.append(centerline[-1])
    tips_np = np.asarray(tips)
    ax.plot(tips_np[:, 0], tips_np[:, 1], tips_np[:, 2], color="black", lw=1.4, marker=".", ms=2.4, label="末端轨迹")
    ax.scatter(centerlines[0][0, 0], centerlines[0][0, 1], centerlines[0][0, 2], marker="^", s=34, color="black")
    set_equal_3d_axes_mm(ax, all_points)
    ax.set_title(title, fontproperties=en_font(9.5))
    ax.set_xlabel("x (mm)", fontproperties=en_font(8.5))
    ax.set_ylabel("y (mm)", fontproperties=en_font(8.5))
    ax.set_zlabel("z (mm)", fontproperties=en_font(8.5))
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(en_font(7.5))
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", frameon=False, prop=zh_font(8.0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=float(iters.min()), vmax=float(iters.max())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.66, pad=0.06)
    cbar.set_label("累计迭代步", fontproperties=zh_font(8.0))
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(en_font(7.5))
    fig.text(0.16, 0.04, f"最终残差 ||E||={float(normE[-1]):.2e}", fontproperties=zh_font(8.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_evolution_panels(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    selected = [item for item in summaries if item["condition_label"] == "single_coil_c6"]
    if not selected:
        return
    fig = plt.figure(figsize=(10.5, 3.7))
    axes = [fig.add_subplot(1, len(selected), idx + 1, projection="3d") for idx in range(len(selected))]
    cmap = plt.get_cmap("viridis")
    for ax, summary in zip(axes, selected):
        case_dir = Path(summary["case_dir"])
        data = np.load(case_dir / "centerline_checkpoints_every20.npz")
        iters = data["iters"]
        centerlines = [np.asarray(data[f"iter_{int(i):05d}"], dtype=np.float64) * 1000.0 for i in iters]
        all_points = np.concatenate(centerlines, axis=0)
        denom = max(len(centerlines) - 1, 1)
        for idx, centerline in enumerate(centerlines):
            if idx == 0:
                ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="0.2", lw=1.6, label="初始")
            elif idx == len(centerlines) - 1:
                ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="#D55E00", lw=2.0, label="收敛")
            else:
                ax.plot(
                    centerline[:, 0],
                    centerline[:, 1],
                    centerline[:, 2],
                    color=cmap(idx / denom),
                    lw=0.55,
                    alpha=0.2,
                )
        tips = np.asarray([line[-1] for line in centerlines])
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color="black", lw=1.0)
        set_equal_3d_axes_mm(ax, all_points)
        ax.set_title(f"{summary['magnet_label']}，{summary['n_iter']}步", fontproperties=zh_font(9.2), pad=6)
        ax.set_xlabel("x (mm)", fontproperties=en_font(8.0))
        ax.set_ylabel("y (mm)", fontproperties=en_font(8.0))
        ax.set_zlabel("z (mm)", fontproperties=en_font(8.0))
        for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick.set_fontproperties(en_font(7.2))
        ax.view_init(elev=22, azim=-58)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, prop=zh_font(8.5))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out_dir / "fig_321_single_coil_evolution_panels.png")
    fig.savefig(out_dir / "fig_321_single_coil_evolution_panels.pdf")
    plt.close(fig)


def plot_m555_checkpoint_panels(out_dir: Path, summaries: list[dict[str, Any]]) -> bool:
    selected = [item for item in summaries if item.get("condition_label") == "m555_random_current"]
    if len(selected) < 5:
        return False
    selected.sort(key=lambda item: item["case_name"])

    fig = plt.figure(figsize=(12.6, 7.1))
    gs = fig.add_gridspec(2, 3, left=0.02, right=0.99, bottom=0.06, top=0.93, wspace=0.06, hspace=0.24)
    axes = []
    for idx in range(5):
        axes.append(fig.add_subplot(gs[idx // 3, idx % 3], projection="3d"))
    legend_ax = fig.add_subplot(gs[1, 2])
    legend_ax.axis("off")
    cmap = plt.get_cmap("viridis")

    for panel_idx, (ax, summary) in enumerate(zip(axes, selected), start=1):
        case_dir = Path(summary["case_dir"])
        data = np.load(case_dir / "centerline_checkpoints_every20.npz")
        iters = data["iters"]
        centerlines = [np.asarray(data[f"iter_{int(i):05d}"], dtype=np.float64) * 1000.0 for i in iters]
        all_points = np.concatenate(centerlines, axis=0)
        denom = max(len(centerlines) - 1, 1)
        keep_stride = max(1, int(np.ceil(len(centerlines) / 28)))
        shown_mid = 0
        for idx, centerline in enumerate(centerlines):
            if idx == 0:
                ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="0.25", lw=1.9, label="初始位姿")
            elif idx == len(centerlines) - 1:
                ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="#D55E00", lw=2.2, label="收敛位姿")
            elif idx % keep_stride == 0:
                shown_mid += 1
                ax.plot(
                    centerline[:, 0],
                    centerline[:, 1],
                    centerline[:, 2],
                    color=cmap(idx / denom),
                    lw=0.65,
                    alpha=0.25,
                )
        tips = np.asarray([line[-1] for line in centerlines])
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color="black", lw=1.0, alpha=0.9, label="末端轨迹")
        ax.scatter(centerlines[0][0, 0], centerlines[0][0, 1], centerlines[0][0, 2], color="black", marker="^", s=24)
        set_equal_3d_axes_mm(ax, all_points)
        ax.view_init(elev=21, azim=-58)
        ax.set_title(f"({chr(64 + panel_idx)}) {summary['magnet_label']}", fontproperties=zh_font(9.2, "bold"), pad=4)
        ax.set_xlabel("x (mm)", fontproperties=en_font(8.0))
        ax.set_ylabel("y (mm)", fontproperties=en_font(8.0))
        ax.set_zlabel("z (mm)", fontproperties=en_font(8.0))
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(en_font(7.2))
        info = (
            f"I={current_short(summary['coil_currents'])}\n"
            f"{magnet_short(summary['m_body_list'])}\n"
            f"M={summary['M_list']}  n={summary['n_iter']}\n"
            f"||E||={summary['final_normE']:.2e}"
        )
        ax.text2D(
            0.02,
            0.98,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            fontproperties=en_font(6.6),
            bbox={"facecolor": "white", "edgecolor": "0.80", "alpha": 0.78, "pad": 2.0},
        )
        ax.text2D(
            0.02,
            0.05,
            f"中间态 {shown_mid} 条",
            transform=ax.transAxes,
            fontproperties=zh_font(7.2),
            color="0.25",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, prop=zh_font(9.0))
    legend_ax.text(
        0.02,
        0.28,
        "注：中间态为每20个LM迭代步\n保存一次的中心线检查点；\n曲线透明度用于避免遮挡。",
        fontproperties=zh_font(9.0),
        va="top",
    )
    fig.savefig(out_dir / "fig_321_m555_checkpoint_evolution_panels.png")
    fig.savefig(out_dir / "fig_321_m555_checkpoint_evolution_panels.pdf")
    fig.savefig(out_dir / "fig_321_existing_stage_evolution_panels.png")
    fig.savefig(out_dir / "fig_321_existing_stage_evolution_panels.pdf")
    plt.close(fig)
    return True


def plot_existing_stage_panels(out_dir: Path, stage_base: Path) -> None:
    cases = []
    for case_name, label in EXISTING_STAGE_CASES.items():
        path = stage_base / case_name / "stage_centerlines.npz"
        if not path.exists():
            continue
        data = np.load(path)
        stage_names = [key for key in data.files if key != "tip_path"]
        stage_names.sort()
        centerlines = [np.asarray(data[key], dtype=np.float64) * 1000.0 for key in stage_names]
        if centerlines:
            cases.append((case_name, label, stage_names, centerlines))
    if not cases:
        return

    fig = plt.figure(figsize=(10.5, 3.7))
    axes = [fig.add_subplot(1, len(cases), idx + 1, projection="3d") for idx in range(len(cases))]
    colors = ["0.25", "#56B4E9", "#009E73", "#D55E00"]
    labels = ["初始位姿", "中间状态", "中间状态", "收敛位姿"]
    for ax, (_case_name, label, stage_names, centerlines) in zip(axes, cases):
        all_points = np.concatenate(centerlines, axis=0)
        for idx, centerline in enumerate(centerlines):
            color = colors[min(idx, len(colors) - 1)]
            lw = 2.1 if idx in (0, len(centerlines) - 1) else 1.5
            alpha = 1.0 if idx in (0, len(centerlines) - 1) else 0.75
            plot_label = labels[min(idx, len(labels) - 1)]
            if idx == 2:
                plot_label = None
            ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                centerline[:, 2],
                color=color,
                lw=lw,
                alpha=alpha,
                label=plot_label,
            )
            ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], color=color, s=18)
        tips = np.asarray([line[-1] for line in centerlines])
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color="black", lw=1.1, marker="o", ms=2.6)
        set_equal_3d_axes_mm(ax, all_points)
        ax.set_title(label, pad=8, fontproperties=zh_font(9.8))
        ax.set_xlabel("x (mm)", fontproperties=en_font(8.0))
        ax.set_ylabel("y (mm)", fontproperties=en_font(8.0))
        ax.set_zlabel("z (mm)", fontproperties=en_font(8.0))
        for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick.set_fontproperties(en_font(7.5))
        ax.view_init(elev=22, azim=-58)
        ax.text2D(0.03, 0.94, f"{len(stage_names)}个延续阶段", transform=ax.transAxes, fontproperties=zh_font(8.0))
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02), prop=zh_font(8.5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.18, top=0.88, wspace=0.08)
    fig.savefig(out_dir / "fig_321_existing_stage_evolution_panels.png")
    fig.savefig(out_dir / "fig_321_existing_stage_evolution_panels.pdf")
    plt.close(fig)


def active_magnets_from_m_body(text: str) -> int:
    return sum(1 for row in parse_m_body_list(text) if np.linalg.norm(row) > 0.0)


def read_complexity_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            m_values = parse_m_list(row["M_list"])
            active = active_magnets_from_m_body(row["m_body_list"])
            rows.append(
                {
                    "case_name": row["case_name"],
                    "M_list": row["M_list"],
                    "grid_intervals": int(sum(m_values)),
                    "active_magnets": active,
                    "wall_s": float(row["wall_s"]),
                    "n_iter": int(float(row["n_iter"])),
                    "ok": str(row["ok"]).lower() == "true",
                    "final_normE": float(row["final_normE"]),
                }
            )
    return rows


def plot_runtime_complexity(out_dir: Path, complexity_csv: Path) -> list[dict[str, Any]]:
    rows = read_complexity_rows(complexity_csv)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["active_magnets"], row["grid_intervals"]), []).append(row)

    summary_rows = []
    for (active, grid), items in sorted(grouped.items()):
        wall = np.asarray([item["wall_s"] for item in items], dtype=np.float64)
        ok_count = sum(1 for item in items if item["ok"])
        summary_rows.append(
            {
                "active_magnets": active,
                "grid_intervals": grid,
                "mean_wall_s": float(np.mean(wall)),
                "std_wall_s": float(np.std(wall)),
                "min_wall_s": float(np.min(wall)),
                "max_wall_s": float(np.max(wall)),
                "case_count": len(items),
                "strict_ok": ok_count,
            }
        )

    with (out_dir / "runtime_complexity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "active_magnets",
            "grid_intervals",
            "mean_wall_s",
            "std_wall_s",
            "min_wall_s",
            "max_wall_s",
            "case_count",
            "strict_ok",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    palette = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}
    markers = {1: "o", 2: "s", 3: "^"}
    fig, ax = plt.subplots(figsize=(6.7, 4.1))
    for active in sorted({row["active_magnets"] for row in summary_rows}):
        points = [row for row in summary_rows if row["active_magnets"] == active]
        points.sort(key=lambda item: item["grid_intervals"])
        x = np.asarray([item["grid_intervals"] for item in points], dtype=np.float64)
        y = np.asarray([item["mean_wall_s"] for item in points], dtype=np.float64)
        ymin = np.asarray([item["min_wall_s"] for item in points], dtype=np.float64)
        ymax = np.asarray([item["max_wall_s"] for item in points], dtype=np.float64)
        ax.fill_between(x, ymin, ymax, color=palette[active], alpha=0.13, linewidth=0)
        raw = [row for row in rows if row["active_magnets"] == active]
        ax.scatter(
            [item["grid_intervals"] + 0.18 * (active - 2) for item in raw],
            [item["wall_s"] for item in raw],
            color=palette[active],
            marker=markers[active],
            s=18,
            alpha=0.26,
            linewidths=0,
        )
        ax.plot(
            x,
            y,
            color=palette[active],
            marker=markers[active],
            markersize=5.2,
            lw=2.0,
            label=f"{active}个磁化点",
        )
    ax.set_xlabel("导管网格总区间数")
    ax.set_ylabel("稳态求解耗时 (s)")
    ax.set_xticks(sorted({row["grid_intervals"] for row in summary_rows}))
    ax.set_xlim(13.7, 31.3)
    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, prop=zh_font(8.5))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font(8.5))
    ax.xaxis.label.set_fontproperties(zh_font(9.5))
    ax.yaxis.label.set_fontproperties(zh_font(9.5))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_321_runtime_complexity.png")
    fig.savefig(out_dir / "fig_321_runtime_complexity.pdf")
    plt.close(fig)
    return summary_rows


def plot_convergence_curves(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    available_conditions = [
        condition
        for condition in ["gravity_only", "single_coil_c6"]
        if any(item["condition_label"] == condition for item in summaries)
    ]
    selected_summaries = [item for item in summaries if item["condition_label"] in set(available_conditions)]
    if not selected_summaries:
        return
    fig, axes_arr = plt.subplots(1, len(available_conditions), figsize=(4.8 * len(available_conditions), 3.6), sharey=True)
    axes = np.atleast_1d(axes_arr)
    palette = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}
    for ax, condition in zip(axes, available_conditions):
        for summary in summaries:
            if summary["condition_label"] != condition:
                continue
            trace_path = Path(summary["case_dir"]) / "convergence_trace.csv"
            iters = []
            residuals = []
            with trace_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    iters.append(int(row["iter"]))
                    residuals.append(float(row["normE"]))
            active = int(summary["active_magnets"])
            ax.plot(
                iters,
                residuals,
                marker=({1: "o", 2: "s", 3: "^"}[active]),
                markersize=3.0,
                linewidth=1.4,
                color=palette[active],
                label=f"{active}个磁化点",
            )
        ax.axhline(1e-5, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("累计迭代步", fontproperties=zh_font(9.0))
        ax.set_title("仅重力" if condition == "gravity_only" else "6号线圈单独通电", fontproperties=zh_font(9.2))
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(en_font(8.0))
    axes[0].set_ylabel("平衡残差 ||E||", fontproperties=zh_font(9.0))
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, frameon=False, loc="upper right", prop=zh_font(8.3))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_321_convergence_curves.png")
    fig.savefig(out_dir / "fig_321_convergence_curves.pdf")
    plt.close(fig)


def write_record(out_dir: Path, summaries: list[dict[str, Any]], complexity_rows: list[dict[str, Any]]) -> None:
    output_items = [
        "- `fig_321_existing_stage_evolution_panels.png/pdf`",
        "- `fig_321_runtime_complexity.png/pdf`",
    ]
    if summaries:
        output_items.append("- `fig_321_m555_checkpoint_evolution_panels.png/pdf`（当 M555 检查点工况不少于 5 组时生成）")
        if any(item["condition_label"] == "single_coil_c6" for item in summaries):
            output_items.append("- `fig_321_single_coil_evolution_panels.png/pdf`")
        if any(item["condition_label"] in {"gravity_only", "single_coil_c6"} for item in summaries):
            output_items.append("- `fig_321_convergence_curves.png/pdf`")
        output_items.append("- `cases/*/fig_evolution.png/pdf`")
    run_scope = [
        "- 复杂度耗时图复用既有 M5/M7/M10 批量扫参记录，不重新运行。",
    ]
    if summaries:
        run_scope.extend(
            [
                "- 姿态演化采用每 20 个 LM 迭代步重入求解并保存一次中心线检查点。",
                "- `m555_random_current`：复现 M555 批次中的 5 个代表性随机电流/磁化配置工况。",
            ]
        )
        if any(item["condition_label"] == "gravity_only" for item in summaries):
            run_scope.append("- `gravity_only`：开启重力，关闭磁场，八个线圈电流为 0。")
        if any(item["condition_label"] == "single_coil_c6" for item in summaries):
            run_scope.append("- `single_coil_c6`：开启重力和磁场，仅 6 号线圈通电。")
    else:
        run_scope.append("- 本次未完成新增重力/单线圈补跑，因此不输出每20步检查点图和收敛曲线。")

    lines = [
        "# 3.2.1 FK 报告图生成记录",
        "",
        "## 目的",
        "",
        "为中期报告 3.2.1 生成更适合论文排版的正运动学姿态演化图、复杂度耗时图和收敛曲线。",
        "",
        "## 口径",
        "",
        *run_scope,
        "",
        "## 输出图表",
        "",
        *output_items,
        "",
        "## 本轮运行摘要",
        "",
    ]
    for item in summaries:
        lines.append(
            f"- `{item['case_name']}`：ok={item['ok']}，iter={item['n_iter']}，"
            f"final_normE={item['final_normE']:.3e}，wall={item['wall_s']:.2f}s。"
        )
    lines.extend(["", "## 复杂度统计摘要", ""])
    for row in complexity_rows:
        lines.append(
            f"- {row['active_magnets']} 个磁化点，网格总区间 {row['grid_intervals']}："
            f"平均 {row['mean_wall_s']:.2f}s，严格收敛 {row['strict_ok']}/{row['case_count']}。"
        )
    lines.extend(
        [
            "",
            "## 元信息",
            "",
            f"- git_commit: `{_repo_commit()}`",
            f"- jax_backend: `{jax.default_backend()}`",
            f"- jax_devices: `{[str(device) for device in jax.devices()]}`",
            "- fonts: 中文优先使用 SimSun/Songti SC；本机未安装时使用随项目保存的 Source Han Serif CN。英文优先使用 Times New Roman；本机未安装时回退 Liberation Serif。",
        ]
    )
    (out_dir / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report_cases(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    zero_current = "0,0,0,0,0,0,0,0"
    c6_current = f"0,0,0,0,0,{float(args.single_coil_current):g},0,0"
    if args.case_family in {"gravity_single", "all"}:
        for magnet_key in GRAVITY_SINGLE_MAGNETS:
            for condition, currents, enable_magnetics in (
                ("gravity_only", zero_current, False),
                ("single_coil_c6", c6_current, True),
            ):
                case_name = f"{magnet_key}_{condition}"
                summary = run_checkpoint_case(
                    out_dir=out_dir,
                    case_name=case_name,
                    condition_label=condition,
                    magnet_key=magnet_key,
                    currents=currents,
                    m_list=args.M_list,
                    tol=float(args.tol),
                    max_total_iter=int(args.max_total_iter),
                    checkpoint_iter=int(args.checkpoint_iter),
                    enable_gravity=True,
                    enable_magnetics=enable_magnetics,
                    linear_solver=str(args.linear_solver),
                )
                summaries.append(summary)
                plot_case_evolution(Path(summary["case_dir"]), Path(summary["case_dir"]) / "fig_evolution", case_name)
                print(
                    json.dumps(
                        {
                            "case_name": case_name,
                            "ok": summary["ok"],
                            "n_iter": summary["n_iter"],
                            "final_normE": summary["final_normE"],
                            "wall_s": summary["wall_s"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    if args.case_family in {"m555", "all"}:
        for case_name, magnet_key, current_key in M555_REPORT_CASES:
            summary = run_checkpoint_case(
                out_dir=out_dir,
                case_name=case_name,
                condition_label="m555_random_current",
                magnet_key=magnet_key,
                currents=CURRENTS[current_key],
                m_list=args.M_list,
                tol=float(args.tol),
                max_total_iter=int(args.max_total_iter),
                checkpoint_iter=int(args.checkpoint_iter),
                enable_gravity=True,
                enable_magnetics=True,
                linear_solver=str(args.linear_solver),
            )
            summaries.append(summary)
            plot_case_evolution(Path(summary["case_dir"]), Path(summary["case_dir"]) / "fig_evolution", case_name)
            print(
                json.dumps(
                    {
                        "case_name": case_name,
                        "ok": summary["ok"],
                        "n_iter": summary["n_iter"],
                        "final_normE": summary["final_normE"],
                        "wall_s": summary["wall_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return summaries


def load_existing_summaries(out_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted((out_dir / "cases").glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["case_dir"] = str(path.parent)
        summaries.append(item)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FK figures for report section 3.2.1.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--complexity_csv", default=str(COMPLEXITY_CSV))
    parser.add_argument("--skip_runs", action="store_true", help="Only redraw figures from existing outputs and CSV files")
    parser.add_argument("--stage_base", default=str(EXISTING_STAGE_BASE))
    parser.add_argument("--M_list", default="5,5,5")
    parser.add_argument("--case_family", choices=["gravity_single", "m555", "all"], default="all")
    parser.add_argument("--single_coil_current", type=float, default=20.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_total_iter", type=int, default=5000)
    parser.add_argument("--checkpoint_iter", type=int, default=20)
    parser.add_argument("--linear_solver", choices=["dense", "dense_fused_cholesky"], default="dense_fused_cholesky")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_runs:
        summaries = load_existing_summaries(out_dir)
    else:
        summaries = run_report_cases(args, out_dir)

    complexity_rows = plot_runtime_complexity(out_dir, Path(args.complexity_csv))
    if not plot_m555_checkpoint_panels(out_dir, summaries):
        plot_existing_stage_panels(out_dir, Path(args.stage_base))
    plot_convergence_curves(out_dir, summaries)
    if summaries:
        plot_evolution_panels(out_dir, summaries)
    write_record(out_dir, summaries, complexity_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
