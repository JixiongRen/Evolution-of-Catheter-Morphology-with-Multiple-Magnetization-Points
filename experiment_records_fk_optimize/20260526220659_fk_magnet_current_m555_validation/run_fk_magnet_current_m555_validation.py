"""Run fixed-grid FK validation across magnet and current configurations."""

from __future__ import annotations

import csv
import json
import os
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

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.efficiency_optimization.fk_under_60s_driver import (
    _build_engine,
    _extract_centerline,
    _get_calib_file,
    _repo_commit,
    _rss_mb,
    _solve_once,
)


OUT_DIR = Path(__file__).resolve().parent
M_LIST = "5,5,5"
TOL = 1e-5
MAX_ITER = 100000

CURRENTS = {
    "rand_a": "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4",
    "rand_b": "-19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2",
}

MAGNETS = {
    "n1_tip_z": {
        "segment_count": 1,
        "m_body_list": "0,0,0; 0,0,0; 0,0,0.005301",
    },
    "n1_base_x": {
        "segment_count": 1,
        "m_body_list": "0.005301,0,0; 0,0,0; 0,0,0",
    },
    "n2_z_pair": {
        "segment_count": 2,
        "m_body_list": "0,0,-0.005301; 0,0,0.005301; 0,0,0",
    },
    "n2_xz_pair": {
        "segment_count": 2,
        "m_body_list": "0.005301,0,0; 0,0,0; 0,0,0.005301",
    },
    "n3_mixed": {
        "segment_count": 3,
        "m_body_list": "0,0,-0.005301; 0,0,0.005301; 0.005301,0,0",
    },
    "n3_xyz": {
        "segment_count": 3,
        "m_body_list": "0.005301,0,0; 0,0.005301,0; 0,0,0.005301",
    },
}


def parse_currents(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 8:
        raise ValueError(f"Expected 8 coil currents, got {len(values)}")
    if any(value < -20.0 or value > 20.0 for value in values):
        raise ValueError(f"Current out of [-20, 20] A: {values}")
    return values


def make_args(case_name: str, *, currents: str, magnets: str) -> Any:
    return SimpleNamespace(
        out_dir=str(OUT_DIR / case_name),
        case_name=case_name,
        mode="direct",
        classification="fixed M555 magnet-current validation",
        coil_currents=currents,
        m_body_list=magnets,
        M_list=M_LIST,
        max_iter=MAX_ITER,
        tol=TOL,
        lm_damping=0.1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
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
        warmup=True,
        current_factors="",
        coarse_M_list=M_LIST,
        stage_max_iter=0,
        stage_tol=TOL,
        final_max_iter=MAX_ITER,
        final_tol=TOL,
        rigid_sample_count=10,
        reference_new_centerline="",
        reference_old_centerline="",
    )


def set_equal_axes(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_centerline(case_dir: Path, centerline: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], linewidth=2.3, color="tab:green")
    ax.scatter(centerline[0, 0], centerline[0, 1], centerline[0, 2], color="black", marker="^", s=55, label="start")
    ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], color="tab:red", s=55, label="tip")
    set_equal_axes(ax, centerline)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "final_centerline.png", dpi=180)
    plt.close(fig)


def run_case(case_name: str, *, current_name: str, currents: str, magnet_name: str, magnet_info: dict[str, Any]) -> dict[str, Any]:
    parse_currents(currents)
    case_dir = OUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(case_name, currents=currents, magnets=str(magnet_info["m_body_list"]))

    case_config = {
        "case_name": case_name,
        "current_name": current_name,
        "coil_currents": currents,
        "magnet_name": magnet_name,
        "magnet_segment_count": int(magnet_info["segment_count"]),
        "m_body_list": str(magnet_info["m_body_list"]),
        "M_list": M_LIST,
        "tol": TOL,
        "max_iter": MAX_ITER,
        "linear_solver": "dense_fused_cholesky",
    }
    (case_dir / "case_config.json").write_text(json.dumps(case_config, ensure_ascii=False, indent=2), encoding="utf-8")

    engine = _build_engine(args, max_iter=MAX_ITER, tol=TOL, M_list=(5, 5, 5))
    target = jax.numpy.asarray(parse_currents(currents), dtype=jax.numpy.float64)
    warmup = engine.warm_up(coil_currents=target, L_protrude=0.11)
    z_star, params, _meshes, ok, stats, row = _solve_once(
        engine,
        currents=target,
        L_protrude=0.11,
        warm_start=False,
        stage_name="direct",
        current_factor=1.0,
    )
    centerline = _extract_centerline(z_star, params, rigid_sample_count=int(args.rigid_sample_count))
    np.save(case_dir / "centerline.npy", centerline)
    plot_centerline(case_dir, centerline, case_name)

    summary = {
        **case_config,
        "ok": bool(ok and float(row["normE"]) < TOL),
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "wall_s": float(row["wall_s"]),
        "first_call_compile_s": float(warmup.get("total_s", 0.0)),
        "wall_s_including_warmup": float(row["wall_s"] + warmup.get("total_s", 0.0)),
        "final_normE": float(row["normE"]),
        "max_abs_E": float(row["max_abs_E"]),
        "centerline_start_m": centerline[0].tolist(),
        "centerline_end_m": centerline[-1].tolist(),
        "centerline_shape": list(centerline.shape),
        "warmup": warmup,
        "stage": row,
        "rss_mb": _rss_mb(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def plot_batch(rows: list[dict[str, Any]]) -> None:
    labels = [str(row["case_name"]) for row in rows]
    ok_colors = ["tab:green" if row.get("ok") else "tab:red" for row in rows]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(np.arange(len(rows)), [float(row.get("wall_s", np.nan)) for row in rows], color=ok_colors)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("steady solve wall time (s)")
    ax.set_title("FK wall time, M=[5,5,5], tol=1e-5")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_wall_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(np.arange(len(rows)), [float(row.get("final_normE", np.nan)) for row in rows], color=ok_colors)
    ax.axhline(TOL, color="black", linestyle="--", linewidth=1.2, label="tol=1e-5")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("final normE")
    ax.set_title("FK convergence residual, M=[5,5,5]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_residual.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    all_points = []
    for row in rows:
        path = OUT_DIR / str(row["case_name"]) / "centerline.npy"
        centerline = np.load(path)
        all_points.append(centerline)
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], linewidth=1.4, label=str(row["case_name"]))
        ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], s=20)
    all_concat = np.concatenate(all_points, axis=0)
    ax.scatter(all_concat[0, 0], all_concat[0, 1], all_concat[0, 2], color="black", marker="^", s=60, label="start")
    set_equal_axes(ax, all_concat)
    ax.set_title("Final FK centerlines across magnet/current cases")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_final_centerlines.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_name",
        "ok",
        "magnet_segment_count",
        "magnet_name",
        "current_name",
        "wall_s",
        "first_call_compile_s",
        "wall_s_including_warmup",
        "n_iter",
        "final_normE",
        "max_abs_E",
        "centerline_end_m",
        "stop_reason",
    ]
    with (OUT_DIR / "batch_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "batch_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    calib_file, calib_reason = _get_calib_file()
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "git_commit": _repo_commit(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "calib_file": calib_file,
        "calib_reason": calib_reason,
        "M_list": M_LIST,
        "tol": TOL,
        "max_iter": MAX_ITER,
        "case_count": len(MAGNETS) * len(CURRENTS),
        "current_rule": "Two deterministic multi-coil current vectors; every coil is within [-20, 20] A.",
        "magnet_rule": "Six m_body_list cases covering one, two, and three nonzero magnetized segments.",
    }
    (OUT_DIR / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for magnet_name, magnet_info in MAGNETS.items():
        for current_name, currents in CURRENTS.items():
            case_name = f"{magnet_name}_{current_name}"
            try:
                summary = run_case(
                    case_name,
                    current_name=current_name,
                    currents=currents,
                    magnet_name=magnet_name,
                    magnet_info=magnet_info,
                )
            except Exception as exc:
                summary = {
                    "case_name": case_name,
                    "ok": False,
                    "magnet_name": magnet_name,
                    "current_name": current_name,
                    "M_list": M_LIST,
                    "tol": TOL,
                    "error": repr(exc),
                }
                case_dir = OUT_DIR / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "error.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(summary)
            write_summary(rows)
            print(
                json.dumps(
                    {
                        "case_name": case_name,
                        "ok": summary.get("ok"),
                        "wall_s": summary.get("wall_s"),
                        "n_iter": summary.get("n_iter"),
                        "final_normE": summary.get("final_normE"),
                        "error": summary.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    plot_batch(rows)
    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
