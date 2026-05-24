"""Run FK validation for M-list and current-limited magnet configurations."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.efficiency_optimization.fk_under_60s_driver import (
    _build_engine,
    _extract_centerline,
    _get_calib_file,
    _load_reference_metrics,
    _repo_commit,
    _rss_mb,
    _solve_once,
)
from equilibrium_solver_nondim import residual_bar


OUT_DIR = Path(__file__).resolve().parent

M_LISTS = {
    "M555": "5,5,5",
    "M777": "7,7,7",
    "M101010": "10,10,10",
}

CURRENTS = {
    "rand_a": "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4",
    "rand_b": "-19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2",
    "rand_c": "4.5,14.9,-17.6,8.8,-2.1,-11.4,19.2,-5.7",
}

MAGNETS = {
    "single_tip": "0,0,0; 0,0,0; 0,0,0.005301",
    "double_z": "0,0,-0.005301; 0,0,0.005301; 0,0,0",
    "triple_mixed": "0,0,-0.005301; 0,0,0.005301; 0.005301,0,0",
}


def parse_currents(text: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if len(values) != 8:
        raise ValueError(f"coil currents must have 8 values, got {len(values)}")
    if any(value < -20.0 or value > 20.0 for value in values):
        raise ValueError(f"coil current out of [-20, 20] A: {values}")
    return values


def parse_m_list(text: str) -> tuple[int, int, int]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if len(values) != 3:
        raise ValueError(f"M_list must have 3 values, got {len(values)}")
    if any(value < 5 or value > 10 for value in values):
        raise ValueError(f"M_list values must be in [5, 10], got {values}")
    return values  # type: ignore[return-value]


def make_args(case_name: str, *, currents: str, magnets: str, m_list: str) -> Any:
    return SimpleNamespace(
        out_dir=str(OUT_DIR / case_name),
        case_name=case_name,
        mode="continuation",
        classification="M5-to-M10 current-limited validation",
        coil_currents=currents,
        m_body_list=magnets,
        M_list=m_list,
        max_iter=5000,
        tol=1e-5,
        lm_damping=0.1,
        jac_method="fwd",
        damping_mode="identity",
        dense_solve_method="pos",
        step_norm_clip=None,
        backtrack_max=0,
        backtrack_factor=0.5,
        trust_region_eta=None,
        max_damping_attempts=1,
        reject_lam_gain=10.0,
        scale_mode="optimized",
        scale_include_external=True,
        warmup=True,
        current_factors="0,0.5,1.0",
        coarse_M_list="5,5,5",
        stage_max_iter=40,
        stage_tol=1e-3,
        final_max_iter=1600,
        final_tol=1e-5,
        rigid_sample_count=10,
        reference_new_centerline="",
        reference_old_centerline="",
    )


def block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def set_equal_axes(ax: Any, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_stage_evolution(case_dir: Path, stage_centerlines: list[np.ndarray], stage_names: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    all_points = np.concatenate(stage_centerlines, axis=0)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(stage_centerlines)))

    tips = []
    for idx, (name, centerline) in enumerate(zip(stage_names, stage_centerlines)):
        color = colors[idx]
        tips.append(centerline[-1])
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color=color, linewidth=1.8, label=name)
        ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], color=color, s=35)

    tip_path = np.asarray(tips, dtype=np.float64)
    ax.plot(tip_path[:, 0], tip_path[:, 1], tip_path[:, 2], color="black", linewidth=2.4, marker="o", label="tip path")
    start = stage_centerlines[0][0]
    ax.scatter(start[0], start[1], start[2], color="red", s=55, marker="^", label="start")
    set_equal_axes(ax, all_points)
    ax.set_title("FK stage morphology evolution")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "stage_morphology_evolution.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        case_dir / "stage_centerlines.npz",
        **{name: centerline for name, centerline in zip(stage_names, stage_centerlines)},
        tip_path=tip_path,
    )


def run_case(case_name: str, *, currents: str, magnets: str, m_list: str) -> dict[str, Any]:
    parse_currents(currents)
    parse_m_list(m_list)
    args = make_args(case_name, currents=currents, magnets=magnets, m_list=m_list)
    case_dir = OUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    case_config = {
        "case_name": case_name,
        "coil_currents": currents,
        "m_body_list": magnets,
        "M_list": m_list,
    }
    (case_dir / "case_config.json").write_text(json.dumps(case_config, ensure_ascii=False, indent=2), encoding="utf-8")

    target = jnp.asarray(parse_currents(currents), dtype=jnp.float64)
    engine = _build_engine(args, max_iter=int(args.stage_max_iter), tol=float(args.stage_tol), M_list=parse_m_list(m_list))
    warmup = engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}

    stages: list[dict[str, Any]] = []
    stage_names: list[str] = []
    stage_centerlines: list[np.ndarray] = []
    z_star = params = stats = None
    ok = False
    start = time.perf_counter()

    for idx, factor in enumerate([0.0, 0.5, 1.0]):
        engine.max_iter = int(args.stage_max_iter)
        engine.tol = float(args.stage_tol)
        z_star, params, _meshes, ok, stats, row = _solve_once(
            engine,
            currents=target * float(factor),
            L_protrude=0.11,
            warm_start=(idx > 0),
            stage_name=f"continuation_{idx:02d}",
            current_factor=float(factor),
        )
        stages.append(row)
        stage_names.append(row["stage"])
        stage_centerlines.append(_extract_centerline(z_star, params, rigid_sample_count=int(args.rigid_sample_count)))

    engine.max_iter = int(args.final_max_iter)
    engine.tol = float(args.final_tol)
    z_star, params, _meshes, ok, stats, row = _solve_once(
        engine,
        currents=target,
        L_protrude=0.11,
        warm_start=True,
        stage_name="final_refine",
        current_factor=1.0,
    )
    stages.append(row)
    stage_names.append(row["stage"])
    stage_centerlines.append(_extract_centerline(z_star, params, rigid_sample_count=int(args.rigid_sample_count)))
    block(z_star)

    if z_star is None or params is None:
        raise RuntimeError("FK solve did not produce a final state")
    E = residual_bar(z_star, params)
    block(E)
    centerline = _extract_centerline(z_star, params, rigid_sample_count=int(args.rigid_sample_count))
    refs = _load_reference_metrics(centerline, args)
    plot_stage_evolution(case_dir, stage_centerlines, stage_names)
    np.save(case_dir / "centerline.npy", centerline)
    np.savez_compressed(
        case_dir / "solution.npz",
        z_bar=np.asarray(jax.device_get(z_star), dtype=np.float64),
        E=np.asarray(jax.device_get(E), dtype=np.float64),
        centerline=centerline,
    )

    stage_wall_s = float(sum(float(item["wall_s"]) for item in stages))
    final_norm = float(jnp.linalg.norm(E))
    summary = {
        **case_config,
        "ok": bool(ok and final_norm < float(args.final_tol)),
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "wall_s": stage_wall_s,
        "first_call_compile_s": float(warmup.get("total_s", 0.0)) if warmup else 0.0,
        "wall_s_including_warmup": stage_wall_s + (float(warmup.get("total_s", 0.0)) if warmup else 0.0),
        "process_wall_s": float(time.perf_counter() - start),
        "final_normE": final_norm,
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "centerline_end_m": centerline[-1].tolist(),
        "stage_count": len(stage_centerlines),
        "warmup": warmup,
        "stages": stages,
        "references": refs,
        "rss_mb": _rss_mb(),
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def plot_batch(rows: list[dict[str, Any]]) -> None:
    labels = [row["case_name"] for row in rows]
    wall = [float(row["wall_s"]) for row in rows]
    norm = [float(row["final_normE"]) for row in rows]
    colors = ["tab:green" if row["ok"] else "tab:red" for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(rows)), wall, color=colors)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("steady solve (s)")
    ax.set_title("M5-M10 current-limited FK runtime")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_runtime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(rows)), norm, color=colors)
    ax.axhline(1e-5, color="black", linestyle="--", linewidth=1.2, label="tol=1e-5")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("final normE")
    ax.set_title("M5-M10 current-limited FK residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "batch_residual.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_name",
        "ok",
        "M_list",
        "coil_currents",
        "m_body_list",
        "wall_s",
        "first_call_compile_s",
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
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "calib_file": calib_file,
        "calib_reason": calib_reason,
        "case_count": len(M_LISTS) * len(CURRENTS) * len(MAGNETS),
        "current_rule": "Deterministic random multi-coil vectors in [-20, 20] A; no single-coil-only case.",
    }
    (OUT_DIR / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for m_name, m_list in M_LISTS.items():
        for mag_name, magnets in MAGNETS.items():
            for current_name, currents in CURRENTS.items():
                case_name = f"{m_name}_{mag_name}_{current_name}"
                try:
                    summary = run_case(case_name, currents=currents, magnets=magnets, m_list=m_list)
                except Exception as exc:
                    summary = {
                        "case_name": case_name,
                        "ok": False,
                        "M_list": m_list,
                        "coil_currents": currents,
                        "m_body_list": magnets,
                        "error": repr(exc),
                    }
                    (OUT_DIR / case_name).mkdir(parents=True, exist_ok=True)
                    (OUT_DIR / case_name / "error.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(summary)
                write_summary(rows)
                print(json.dumps({
                    "case_name": case_name,
                    "ok": summary.get("ok"),
                    "wall_s": summary.get("wall_s"),
                    "final_normE": summary.get("final_normE"),
                    "error": summary.get("error"),
                }, ensure_ascii=False), flush=True)
    plot_batch(rows)
    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
