#!/usr/bin/env python3
"""Run angle-continuation probes for A1 supervised IK."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import run_a1_axial_circle_formal_experiment as formal


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _parse_currents(text: str | None) -> list[float] | None:
    """Parse an optional comma-separated current vector."""
    if text is None or str(text).strip() == "":
        return None
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(values) != 8:
        raise ValueError("--initial-I-A must contain exactly 8 currents")
    return values


def _target_from_circle_index(index_float: float, *, center_x_m: float, radius_m: float) -> dict[str, Any]:
    """Return a target on the standard A1 delivery circle."""
    theta = 2.0 * math.pi * float(index_float) / 20.0
    return {
        "index": int(round(float(index_float) * 100.0)),
        "index_float": float(index_float),
        "theta_rad": float(theta),
        "p_des_m": [
            float(center_x_m),
            float(radius_m * math.cos(theta)),
            float(radius_m * math.sin(theta)),
        ],
    }


def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous rows while preserving scalar provenance fields."""
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--catheter", default="A1")
    parser.add_argument("--start-index", type=float, default=13.0)
    parser.add_argument("--end-index", type=float, default=12.0)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--center-x-m", type=float, default=0.06)
    parser.add_argument("--radius-m", type=float, default=0.1)
    parser.add_argument("--eps-p-mm", type=float, default=5.0)
    parser.add_argument("--outer-max-iter", type=int, default=25)
    parser.add_argument("--per-point-timeout-s", type=float, default=1800.0)
    parser.add_argument("--step-I-max", type=float, default=5.0)
    parser.add_argument("--step-x-max", type=float, default=0.02)
    parser.add_argument("--initial-I-A", type=str, default=None)
    parser.add_argument("--initial-x-m", type=float, default=None)
    parser.add_argument("--initial-branch-state-npz", type=str, default=None)
    parser.add_argument("--channel-gate-file", type=str, default=None)
    parser.add_argument("--channel-gate-missing-policy", choices=("block_all", "allow_x_only", "allow_all"), default="block_all")
    parser.add_argument("--channel-gate-stage", choices=("refine", "basin"), default="refine")
    parser.add_argument("--channel-gate-allow-limited", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.segments) <= 0:
        raise ValueError("--segments must be positive")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = formal._load_config()
    if args.catheter not in config["catheter_types"]:
        raise ValueError(f"Unknown catheter key: {args.catheter}")
    eps_p_m = float(args.eps_p_mm) / 1000.0
    targets = [
        _target_from_circle_index(
            float(args.start_index) + (float(args.end_index) - float(args.start_index)) * float(k) / float(args.segments),
            center_x_m=float(args.center_x_m),
            radius_m=float(args.radius_m),
        )
        for k in range(1, int(args.segments) + 1)
    ]
    targets_path = out_dir / "targets.json"
    targets_path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")

    sensitivity_config = {
        "x_sensitivity_method": "implicit",
        "sensitivity_mode": "gn_svd",
        "sensitivity_solve_mode": "svd",
        "sensitivity_svd_rcond": 1e-9,
        "sensitivity_spectral_filter": "hard",
        "sensitivity_spectral_damping": 0.0,
        "ridge_adjoint": 1e-10,
        "sensitivity_conditioning": False,
    }
    p2_config = {
        "trust_region": True,
        "terminal_error_first": True,
        "reliable_subspace": True,
        "reliable_task_rcond": 0.001,
        "min_full_task_rank": 2,
        "max_backtracks": 3,
        "backtrack_shrink": 0.5,
        "disable_Ix_jacobian_below_dx": 0.001,
    }
    channel_gate_config = {
        "file": None if args.channel_gate_file is None else str(Path(args.channel_gate_file).resolve()),
        "missing_policy": str(args.channel_gate_missing_policy),
        "allow_limited": bool(args.channel_gate_allow_limited),
        "stage": str(args.channel_gate_stage),
    }

    phase_dir = out_dir / "angle_continuation"
    current_I = _parse_currents(args.initial_I_A)
    current_x = 0.103 if args.initial_x_m is None else float(args.initial_x_m)
    current_branch = None if args.initial_branch_state_npz is None else str(Path(args.initial_branch_state_npz).resolve())
    warm_source = "manual_branch_state" if current_branch is not None else "manual_initial_state"
    rows: list[dict[str, Any]] = []
    metadata = {
        "catheter": args.catheter,
        "start_index": float(args.start_index),
        "end_index": float(args.end_index),
        "segments": int(args.segments),
        "center_x_m": float(args.center_x_m),
        "radius_m": float(args.radius_m),
        "eps_p_m": eps_p_m,
        "outer_max_iter": int(args.outer_max_iter),
        "per_point_timeout_s": float(args.per_point_timeout_s),
        "step_I_max": float(args.step_I_max),
        "step_x_max": float(args.step_x_max),
        "initial_branch_state_npz": current_branch,
        "channel_gate_config": channel_gate_config,
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for target in targets:
        row = formal._run_one_target(
            catheter_key=str(args.catheter),
            cfg=formal._catheter_solver_cfg(config, str(args.catheter)),
            target=target,
            phase_dir=phase_dir,
            phase="angle_continuation",
            experiment_type="angle_continuation",
            experiment_type_zh="角度分段延拓",
            formal_statistics_included=False,
            initial_I_A=current_I,
            initial_x_m=current_x,
            x_search_max_m=0.203,
            enable_gravity=False,
            warm_start_source=warm_source,
            outer_max_iter=int(args.outer_max_iter),
            per_point_timeout_s=float(args.per_point_timeout_s),
            eps_p_m=eps_p_m,
            step_I_max=float(args.step_I_max),
            step_x_max=float(args.step_x_max),
            sensitivity_config=sensitivity_config,
            p2_config=p2_config,
            channel_gate_config=channel_gate_config,
            initial_branch_state_npz=current_branch,
        )
        row["target_index_float"] = float(target["index_float"])
        row["target_theta_rad"] = float(target["theta_rad"])
        rows.append(row)
        print(
            f"A1 angle {float(target['index_float']):.3f}: "
            f"err={row['err_true_mm']:.3f} mm success={row['success_eps']} cls={row['cls']} wall={row['wall_s']:.1f}s",
            flush=True,
        )
        if row.get("I_A") is not None and row.get("x_m") is not None and row.get("branch_npz_path"):
            current_I = [float(v) for v in row["I_A"]]
            current_x = float(row["x_m"])
            current_branch = str(row["branch_npz_path"])
            warm_source = f"angle_prev_{float(target['index_float']):.3f}"
        if bool(args.stop_on_failure) and not bool(row.get("success_eps")):
            break

    _write_union_csv(out_dir / "angle_continuation_summary.csv", rows)
    formal._plot_phase_overview(out_dir, rows, targets, "angle_continuation_overview")
    result = {
        "metadata": metadata,
        "targets": targets,
        "rows": rows,
        "success_count": sum(1 for row in rows if bool(row.get("success_eps"))),
        "final_error_mm": rows[-1].get("err_true_mm") if rows else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
