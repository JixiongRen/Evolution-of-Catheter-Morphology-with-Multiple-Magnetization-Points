"""Run gravity-only FK stiffness sweep for section 3.2.1 validation.

The script keeps geometry and loading consistent with the report figure cases,
but sweeps flexible Young's modulus to check whether the gravity sag is
physically plausible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fk_engine import ForwardKinematicsEngine
from experiments.efficiency_optimization.fk_under_60s_driver import _block, _extract_centerline, _rss_mb
from experiments.report_figures.generate_fk_321_report_figures import (
    final_norm,
    make_initial_state,
    parse_m_list,
    parse_vector,
    save_checkpoint_npz,
    setup_plot_style,
)


DEFAULT_OUT = REPO_ROOT / "experiment_records_fk_optimize" / "20260615_fk_gravity_E_sweep"


def build_engine(
    *,
    flex_E: float,
    m_list: str,
    max_iter: int,
    tol: float,
    linear_solver: str,
) -> ForwardKinematicsEngine:
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=parse_m_list(m_list),
        L1_min=1e-6,
        flex_d_outer=(0.0015,),
        flex_E=(float(flex_E),),
        flex_G=(0.6e6,),
        flex_rho=(970.0,),
        rigid_d_outer=(0.0015,),
        rigid_rho=(7500.0,),
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        g_world=(0.0, 0.0, -9.81),
        enable_magnetics=False,
        calib_file=None,
        actuation_table_pkl=None,
        m_body_list=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.005301)),
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


def centerline_metrics(centerline: np.ndarray) -> dict[str, float]:
    if centerline.size == 0:
        return {
            "tip_x_m": float("nan"),
            "tip_y_m": float("nan"),
            "tip_z_m": float("nan"),
            "tip_sag_m": float("nan"),
            "polyline_length_m": float("nan"),
            "terminal_angle_deg": float("nan"),
        }
    seg = np.diff(centerline, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    tip = centerline[-1]
    terminal_angle = float("nan")
    if len(seg) > 0 and lengths[-1] > 0:
        tangent = seg[-1] / lengths[-1]
        terminal_angle = float(np.degrees(np.arctan2(-tangent[2], max(tangent[0], 1e-12))))
    return {
        "tip_x_m": float(tip[0]),
        "tip_y_m": float(tip[1]),
        "tip_z_m": float(tip[2]),
        "tip_sag_m": float(centerline[0, 2] - tip[2]),
        "polyline_length_m": float(np.sum(lengths)),
        "terminal_angle_deg": terminal_angle,
    }


def write_trace_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "normE",
                "max_abs_E",
                "tip_x_m",
                "tip_y_m",
                "tip_z_m",
                "tip_sag_m",
                "polyline_length_m",
                "terminal_angle_deg",
                "stop_reason",
                "block_iter",
                "wall_s",
            ],
        )
        writer.writeheader()


def append_trace(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "normE",
                "max_abs_E",
                "tip_x_m",
                "tip_y_m",
                "tip_z_m",
                "tip_sag_m",
                "polyline_length_m",
                "terminal_angle_deg",
                "stop_reason",
                "block_iter",
                "wall_s",
            ],
        )
        writer.writerow(row)


def run_case(
    *,
    out_dir: Path,
    flex_E: float,
    m_list: str,
    tol: float,
    max_total_iter: int,
    checkpoint_iter: int,
    linear_solver: str,
    save_npz_every: int,
    force: bool,
) -> dict[str, Any]:
    case_name = f"E_{flex_E / 1e6:.1f}MPa".replace(".", "p")
    case_dir = out_dir / "cases" / case_name
    summary_path = case_dir / "summary.json"
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["case_dir"] = str(case_dir)
        print(json.dumps({"case_name": case_name, "event": "skip_existing"}, ensure_ascii=False), flush=True)
        return summary

    case_dir.mkdir(parents=True, exist_ok=True)
    target = jnp.asarray(parse_vector("0,0,0,0,0,0,0,0", expected=8), dtype=jnp.float64)
    engine = build_engine(
        flex_E=float(flex_E),
        m_list=m_list,
        max_iter=0,
        tol=tol,
        linear_solver=linear_solver,
    )
    config = {
        "case_name": case_name,
        "condition_label": "gravity_only_E_sweep",
        "flex_E": float(flex_E),
        "flex_E_MPa": float(flex_E / 1e6),
        "flex_G": 0.6e6,
        "flex_d_outer": 0.0015,
        "flex_rho": 970.0,
        "rigid_d_outer": 0.0015,
        "rigid_rho": 7500.0,
        "flex_lengths_tail": [0.03, 0.03],
        "rigid_lengths": [0.003, 0.003, 0.003],
        "L_protrude": 0.11,
        "M_list": m_list,
        "tol": float(tol),
        "max_total_iter": int(max_total_iter),
        "checkpoint_iter": int(checkpoint_iter),
        "linear_solver": str(linear_solver),
    }
    (case_dir / "case_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path = case_dir / "convergence_trace.csv"
    write_trace_header(trace_path)

    print(json.dumps({"case_name": case_name, "event": "start", "flex_E": flex_E}, ensure_ascii=False), flush=True)
    start = time.perf_counter()
    z_bar, params = make_initial_state(engine, target, L_protrude=0.11)
    engine._last_z_bar = z_bar
    engine._last_scales = params.scales
    engine._last_flex_lengths = engine._compute_flex_lengths(L_protrude=0.11)
    engine._last_rigid_lengths = engine.rigid_lengths
    _block(z_bar)

    centerline0 = _extract_centerline(z_bar, params, rigid_sample_count=10)
    checkpoints: list[dict[str, Any]] = [
        {
            "iter": 0,
            "normE": float("nan"),
            "max_abs_E": float("nan"),
            "centerline": centerline0,
            "stop_reason": "initial_guess",
            "block_iter": 0,
        }
    ]
    row0 = {
        "iter": 0,
        "normE": float("nan"),
        "max_abs_E": float("nan"),
        **centerline_metrics(centerline0),
        "stop_reason": "initial_guess",
        "block_iter": 0,
        "wall_s": 0.0,
    }
    append_trace(trace_path, row0)

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
        block_iter = int(stats.n_iter) if stats is not None else int(checkpoint_iter)
        total_iter += block_iter if block_iter > 0 else int(checkpoint_iter)
        normE, max_abs = final_norm(z_bar, params)
        stop_reason = str(stats.stop_reason) if stats is not None else "unknown"
        centerline = _extract_centerline(z_bar, params, rigid_sample_count=10)
        item = {
            "iter": int(total_iter),
            "normE": normE,
            "max_abs_E": max_abs,
            "centerline": centerline,
            "stop_reason": stop_reason,
            "block_iter": block_iter,
        }
        checkpoints.append(item)
        append_trace(
            trace_path,
            {
                "iter": int(total_iter),
                "normE": normE,
                "max_abs_E": max_abs,
                **centerline_metrics(centerline),
                "stop_reason": stop_reason,
                "block_iter": block_iter,
                "wall_s": time.perf_counter() - start,
            },
        )
        if len(checkpoints) % max(int(save_npz_every), 1) == 0:
            save_checkpoint_npz(case_dir, checkpoints)
        print(
            json.dumps(
                {
                    "case_name": case_name,
                    "iter": int(total_iter),
                    "normE": normE,
                    "tip_z_mm": centerline[-1, 2] * 1000.0,
                    "stop_reason": stop_reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if ok or normE < float(tol):
            break

    final_centerline = checkpoints[-1]["centerline"]
    np.save(case_dir / "centerline.npy", final_centerline)
    save_checkpoint_npz(case_dir, checkpoints)
    summary = {
        **config,
        "ok": bool(ok or checkpoints[-1]["normE"] < float(tol)),
        "stop_reason": stop_reason,
        "n_iter": int(total_iter),
        "wall_s": float(time.perf_counter() - start),
        "final_normE": float(checkpoints[-1]["normE"]),
        "max_abs_E": float(checkpoints[-1]["max_abs_E"]),
        "checkpoint_count": len(checkpoints),
        "centerline_shape": list(final_centerline.shape),
        **centerline_metrics(final_centerline),
        "rss_mb": _rss_mb(),
        "case_dir": str(case_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_name": case_name,
                "event": "done",
                "ok": summary["ok"],
                "n_iter": summary["n_iter"],
                "tip_z_mm": summary["tip_z_m"] * 1000.0,
                "final_normE": summary["final_normE"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def write_sweep_summary(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = sorted(summaries, key=lambda item: float(item["flex_E"]))
    csv_path = out_dir / "sweep_summary.csv"
    fields = [
        "case_name",
        "flex_E_MPa",
        "ok",
        "stop_reason",
        "n_iter",
        "wall_s",
        "final_normE",
        "max_abs_E",
        "tip_x_m",
        "tip_z_m",
        "tip_sag_m",
        "terminal_angle_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    xs = [float(row["flex_E_MPa"]) for row in rows]
    zs = [-1000.0 * float(row["tip_z_m"]) for row in rows]
    ax.plot(xs, zs, marker="o", linewidth=1.8)
    ax.set_xlabel("Young's modulus E (MPa)")
    ax.set_ylabel("Tip sag -z (mm)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title("Gravity-only FK stiffness sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tip_sag_vs_E.png", dpi=300)
    fig.savefig(out_dir / "fig_tip_sag_vs_E.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gravity-only FK Young's modulus sweep.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--E_values", default="0.2e6,0.5e6,1.0e6,1.8e6")
    parser.add_argument("--M_list", default="5,5,5")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_total_iter", type=int, default=20000)
    parser.add_argument("--checkpoint_iter", type=int, default=20)
    parser.add_argument("--save_npz_every", type=int, default=25)
    parser.add_argument("--linear_solver", choices=["dense", "dense_fused_cholesky"], default="dense")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    e_values = parse_vector(args.E_values)
    summaries = []
    for value in e_values:
        summaries.append(
            run_case(
                out_dir=out_dir,
                flex_E=float(value),
                m_list=str(args.M_list),
                tol=float(args.tol),
                max_total_iter=int(args.max_total_iter),
                checkpoint_iter=int(args.checkpoint_iter),
                linear_solver=str(args.linear_solver),
                save_npz_every=int(args.save_npz_every),
                force=bool(args.force),
            )
        )
    write_sweep_summary(out_dir, summaries)
    (out_dir / "record.md").write_text(
        "\n".join(
            [
                "# Gravity-only FK Young's modulus sweep",
                "",
                f"- E values: {', '.join(str(float(v)) for v in e_values)} Pa",
                f"- tol: {float(args.tol):.3g}",
                f"- max_total_iter: {int(args.max_total_iter)}",
                f"- checkpoint_iter: {int(args.checkpoint_iter)}",
                f"- linear_solver: {args.linear_solver}",
                "",
                "See `sweep_summary.csv` and per-case directories under `cases/`.",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
