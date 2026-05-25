from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondimJAX, residual_bar, with_L1_dim, with_coil_currents
from experiments.efficiency_optimization.fk_under_60s_driver import _extract_centerline, _pointwise_metrics, _rss_mb
from fk_engine import ForwardKinematicsEngine
from rod_mesh_nondim import build_uniform_mesh
from utils_nondim import make_initial_guess_multi_bar_jax


TARGET_TOLS = (1e-5, 1e-6, 1e-7, 1e-4)
TARGET_CURRENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
TARGET_M_BODY = (
    (0.0, 0.0, -0.005301),
    (0.0, 0.0, 0.005301),
    (0.005301, 0.0, 0.0),
)


def block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tol_label(tol: float) -> str:
    return f"tol_{tol:.0e}"


def get_calib_file() -> tuple[Path, str]:
    requested = Path("/abs/path/calibration.json")
    if requested.exists():
        return requested, "requested_path_exists"
    return ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml", "requested_path_missing_used_repo_fallback"


def build_engine(calib_file: Path, tol: float) -> ForwardKinematicsEngine:
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(5, 5, 5),
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
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=str(calib_file),
        actuation_table_pkl=str(
            ROOT
            / "supiee_auto_diff"
            / "offline_interpolation_data"
            / "actuation_tables"
            / "actuation_table.pkl"
        ),
        m_body_list=TARGET_M_BODY,
        max_iter=100000,
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense",
        damping_mode="identity",
        dense_solve_method="solve",
        scale_mode="optimized",
        scale_include_external=True,
        L_protrude_max=0.11,
    )


def initial_state(engine: ForwardKinematicsEngine, currents: jnp.ndarray):
    engine._ensure_static_initialized(L_protrude=0.11, coil_currents=currents)
    assert engine._params_static_base is not None
    assert engine._meshes_ref is not None

    flex_lengths = engine._compute_flex_lengths(L_protrude=0.11)
    params = with_coil_currents(engine._params_static_base, currents)
    params = with_L1_dim(params, jnp.asarray(float(flex_lengths[0]), dtype=jnp.float64))

    meshes = list(engine._meshes_ref)
    meshes[0] = build_uniform_mesh(float(flex_lengths[0]), int(engine.M_list[0]))

    z0_bar, *_ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=params.scales,
        p0_dim=engine.p0_dim,
        Q0=engine.Q0,
        axis_body=engine.axis_body,
    )
    return z0_bar, params, meshes


def save_case_plot(case_dir: Path, summary: dict[str, Any]) -> None:
    data = np.load(case_dir / "trajectory_snapshots.npz")
    centerlines = data["centerlines"]
    iters = data["iters"]
    normE = data["normE"]
    if centerlines.size == 0:
        return

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("viridis")
    count = max(int(centerlines.shape[0]), 1)
    for idx, line in enumerate(centerlines):
        color = cmap(idx / max(count - 1, 1))
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, alpha=0.45, linewidth=1.0)

    final_line = centerlines[-1]
    tips = centerlines[:, -1, :]
    ax.plot(final_line[:, 0], final_line[:, 1], final_line[:, 2], color="black", linewidth=2.5, label="final")
    ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color="tab:red", linewidth=1.6, label="tip path")
    ax.scatter(final_line[0, 0], final_line[0, 1], final_line[0, 2], color="black", s=35, label="start")
    ax.scatter(tips[-1, 0], tips[-1, 1], tips[-1, 2], color="tab:red", s=45, label="final tip")
    ax.set_title(f"{summary['case_name']} convergence, snapshots every 50 accepted steps")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(case_dir / "convergence_every_50.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.semilogy(iters, normE, marker="o", markersize=2.5, linewidth=1.0)
    ax2.axhline(float(summary["tol"]), color="tab:red", linestyle="--", linewidth=1.0, label="tol")
    ax2.set_xlabel("accepted iteration")
    ax2.set_ylabel("||E||")
    ax2.set_title(f"{summary['case_name']} residual snapshots")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(case_dir / "residual_every_50.png", dpi=180)
    plt.close(fig2)


def run_case(
    solver: MultiSegmentEquilibriumSolverNondimJAX,
    z0_bar: Any,
    params: Any,
    out_dir: Path,
    tol: float,
    warmup: dict[str, float],
) -> dict[str, Any]:
    snapshots: list[np.ndarray] = []
    snapshot_iters: list[int] = []
    snapshot_norms: list[float] = []
    accepted_callback_count = {"value": 0}

    def callback(iter_num: int, z_bar: Any, normE: float) -> None:
        if int(iter_num) == 0:
            snapshots.append(_extract_centerline(z_bar, params, rigid_sample_count=10))
            snapshot_iters.append(int(iter_num))
            snapshot_norms.append(float(normE))
            return
        accepted_callback_count["value"] += 1
        if accepted_callback_count["value"] % 50 == 0:
            snapshots.append(_extract_centerline(z_bar, params, rigid_sample_count=10))
            snapshot_iters.append(int(iter_num))
            snapshot_norms.append(float(normE))

    start = time.perf_counter()
    z_star, ok, stats = solver.solve_lm(
        z0_bar,
        max_iter=100000,
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        callback=callback,
        return_stats=True,
    )
    block(z_star)
    wall_s = float(time.perf_counter() - start)
    E = residual_bar(z_star, params)
    block(E)
    final_normE = float(jnp.linalg.norm(E))
    final_centerline = _extract_centerline(z_star, params, rigid_sample_count=10)

    final_iter = int(stats.n_iter)
    if not snapshot_iters or snapshot_iters[-1] != final_iter:
        snapshots.append(final_centerline)
        snapshot_iters.append(final_iter)
        snapshot_norms.append(final_normE)

    case_dir = out_dir / tol_label(tol)
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "centerline.npy", final_centerline)
    np.savez_compressed(
        case_dir / "solution.npz",
        z_bar=np.asarray(jax.device_get(z_star), dtype=np.float64),
        E=np.asarray(jax.device_get(E), dtype=np.float64),
        centerline=final_centerline,
    )
    np.savez_compressed(
        case_dir / "trajectory_snapshots.npz",
        iters=np.asarray(snapshot_iters, dtype=np.int64),
        normE=np.asarray(snapshot_norms, dtype=np.float64),
        centerlines=np.asarray(snapshots, dtype=np.float64),
    )

    summary = {
        "case_name": tol_label(tol),
        "tol": float(tol),
        "ok": bool(ok and final_normE < float(tol)),
        "stop_reason": str(stats.stop_reason),
        "n_iter": final_iter,
        "wall_s": wall_s,
        "first_call_compile_s": float(warmup.get("total_s", 0.0)),
        "steady_solve_s": wall_s,
        "final_normE": final_normE,
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "centerline_shape": list(final_centerline.shape),
        "centerline_start_m": final_centerline[0].tolist(),
        "centerline_end_m": final_centerline[-1].tolist(),
        "snapshot_count": len(snapshot_iters),
        "scale": {
            "mode": "optimized",
            "include_external": True,
            "L_ref": float(params.scales.L_ref),
            "F_ref": float(params.scales.F_ref),
            "M_ref": float(params.scales.M_ref),
        },
        "rss_mb": _rss_mb(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_case_plot(case_dir, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def plot_final_overlay(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, summary in enumerate(summaries):
        case_dir = out_dir / summary["case_name"]
        centerline = np.load(case_dir / "centerline.npy")
        label = f"{summary['case_name']} n={summary['n_iter']} norm={summary['final_normE']:.2e}"
        color = colors[idx % len(colors)]
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], linewidth=2.0, color=color, label=label)
        ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], color=color, s=35)

    first = np.load(out_dir / summaries[0]["case_name"] / "centerline.npy")
    ax.scatter(first[0, 0], first[0, 1], first[0, 2], color="black", s=45, label="start")
    ax.set_title("New FK final pose comparison by tolerance")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "final_pose_tolerance_overlay.png", dpi=180)
    plt.close(fig)


def write_metrics(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    ref_summary = min(summaries, key=lambda item: item["tol"])
    ref_centerline = np.load(out_dir / ref_summary["case_name"] / "centerline.npy")
    for summary in summaries:
        centerline = np.load(out_dir / summary["case_name"] / "centerline.npy")
        metrics = _pointwise_metrics(ref_centerline, centerline, L_ref=0.11)
        row = {
            "case_name": summary["case_name"],
            "tol": summary["tol"],
            "ok": summary["ok"],
            "stop_reason": summary["stop_reason"],
            "n_iter": summary["n_iter"],
            "wall_s": summary["wall_s"],
            "final_normE": summary["final_normE"],
            "max_abs_E": summary["max_abs_E"],
            "end_x_m": summary["centerline_end_m"][0],
            "end_y_m": summary["centerline_end_m"][1],
            "end_z_m": summary["centerline_end_m"][2],
            "ref_case": ref_summary["case_name"],
            **metrics,
        }
        rows.append(row)

    fields = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_file, calib_reason = get_calib_file()
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(ROOT),
        "git_commit": repo_commit(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "calib_file": str(calib_file),
        "calib_reason": calib_reason,
        "actuation_table_pkl": str(
            ROOT
            / "supiee_auto_diff"
            / "offline_interpolation_data"
            / "actuation_tables"
            / "actuation_table.pkl"
        ),
        "fixed_case": {
            "flex_lengths": [0.03, 0.03],
            "rigid_lengths": [0.003, 0.003, 0.003],
            "L_protrude": 0.11,
            "M_list": [5, 5, 5],
            "enable_gravity": True,
            "enable_magnetics": True,
            "coil_currents": list(TARGET_CURRENTS),
            "m_body_list": [list(item) for item in TARGET_M_BODY],
            "p0": [0.0, 0.0, 0.0],
            "Q0": [0.70710678, 0.0, 0.70710678, 0.0],
            "axis_body": [0.0, 0.0, 1.0],
            "max_iter": 100000,
            "lm_damping": 1e-1,
            "jac_method": "fwd",
            "scale_mode": "optimized",
            "scale_include_external": True,
            "dense_solve_method": "solve",
        },
        "tolerances": list(TARGET_TOLS),
    }
    (out_dir / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    currents = jnp.asarray(TARGET_CURRENTS, dtype=jnp.float64)
    engine = build_engine(calib_file, TARGET_TOLS[0])
    z0_bar, params, _meshes = initial_state(engine, currents)
    solver = MultiSegmentEquilibriumSolverNondimJAX(params)

    warmup_start = time.perf_counter()
    t0 = time.perf_counter()
    E0 = solver.residual_jit(z0_bar)
    block(E0)
    t1 = time.perf_counter()
    J0 = solver.jacobian_jit(z0_bar, method="fwd")
    block(J0)
    t2 = time.perf_counter()
    warmup = {
        "residual_s": float(t1 - t0),
        "jacobian_s": float(t2 - t1),
        "total_s": float(t2 - warmup_start),
    }
    warmup["total_wall_s"] = float(time.perf_counter() - warmup_start)
    (out_dir / "warmup.json").write_text(json.dumps(warmup, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries = []
    for tol in TARGET_TOLS:
        summaries.append(run_case(solver, z0_bar, params, out_dir, tol, warmup))

    write_metrics(out_dir, summaries)
    plot_final_overlay(out_dir, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
