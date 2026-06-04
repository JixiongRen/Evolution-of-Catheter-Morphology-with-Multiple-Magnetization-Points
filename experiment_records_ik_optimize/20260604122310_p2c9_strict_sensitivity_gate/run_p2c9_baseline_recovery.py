"""Probe baseline recovery options for failed P2c9 cases."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    case = next(item for item in p2c4._cases() if item.name == "n3_mixed_rand20")
    engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")

    best_success: dict[str, Any] | None = None
    best_stationary: dict[str, Any] | None = None
    for continuity_mu in (0.01, 0.1, 1.0, 10.0):
        residual_jit, _task_jit, output_jit, continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_oracle,
            reg_mu=1.0e-4,
            continuity_mu=float(continuity_mu),
            steps_per_flex=4,
        )
        _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = Jy_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        y_start = np.asarray(y_oracle, dtype=float)
        for max_iter in (24, 80, 160):
            solve = p2c7._solve_branch(
                y0=y_start,
                I=I0,
                residual_jit=residual_jit,
                output_jit=output_jit,
                Jy_jit=Jy_jit,
                max_iter=max_iter,
            )
            cont_norm = float(
                np.linalg.norm(
                    np.asarray(jax.device_get(continuity_jit(jnp.asarray(solve.y), jnp.asarray(I0))), dtype=float)
                )
            )
            row = {
                "case": case.name,
                "continuity_mu": continuity_mu,
                "max_iter": max_iter,
                "success": solve.success,
                "steady_s": solve.steady_s,
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "continuity_norm": cont_norm,
                "y_norm": float(np.linalg.norm(solve.y)),
            }
            rows.append(row)
            if row["success"] and (
                best_success is None
                or (row["stationarity_norm"], row["residual_norm"]) < (best_success["stationarity_norm"], best_success["residual_norm"])
            ):
                best_success = row
            if best_stationary is None or (row["stationarity_norm"], row["residual_norm"]) < (
                best_stationary["stationarity_norm"],
                best_stationary["residual_norm"],
            ):
                best_stationary = row
            y_start = solve.y
    summary = {
        "case": case.name,
        "total_wall_s": float(time.perf_counter() - t0),
        "best_success": best_success,
        "best_stationary": best_stationary,
        "rows": rows,
        "classification": "recovered"
        if best_success and best_success["stationarity_norm"] <= 1.0e-2
        else "not_recovered",
    }
    (OUT_DIR / "baseline_recovery_summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "baseline_recovery_summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
