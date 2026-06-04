"""Run P2c9e soft-to-hard continuation and coupling audits."""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import os
import subprocess
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

from forward_kinematics_optimized import segments_nondim

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260604)


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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _active_direction(I0: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    d = np.zeros(8)
    d[idx] = 1.0
    return d


def _rigid_coupling_audit() -> dict[str, Any]:
    source = inspect.getsource(segments_nondim.rigid_state_along_dim)
    tree = ast.parse(source)
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = ast.unparse(node.value)
    p_expr = assignments.get("p_sigma", "")
    q_expr = assignments.get("Q_sigma", "")
    return {
        "function": "forward_kinematics_optimized.segments_nondim.rigid_state_along_dim",
        "p_sigma_expr": p_expr,
        "Q_sigma_expr": q_expr,
        "pose_depends_on_force": "f_ext_total_dim" in p_expr or "f_ext_total_dim" in q_expr,
        "pose_depends_on_torque": "tau_ext_total_dim" in p_expr or "tau_ext_total_dim" in q_expr,
        "source_excerpt": source,
        "interpretation": "Rigid external wrench updates distal force/moment but does not update rigid pose in this implementation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n3_mixed_base_i4,n3_mixed_rand20")
    parser.add_argument("--continuity_mu", default="0,1e-4,1e-2,1e0,1e2")
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=80)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    case_names = {x.strip() for x in args.cases.split(",") if x.strip()}
    mu_values = [float(x) for x in args.continuity_mu.split(",") if x.strip()]
    continuation_rows: list[dict[str, Any]] = []
    explicit_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in [item for item in p2c4._cases() if item.name in case_names]:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
        y_start = np.asarray(y_ref, dtype=float).reshape(-1)
        best_success = None
        best_stationary = None
        for mu in mu_values:
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "stage", "continuity_mu": mu})
            residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, _JI_jit, _Ty_jit, TI_jit = p2c8._make_functions(
                params_base=params_base,
                target_p=target_p,
                target_q=target_q,
                y_ref=y_ref,
                reg_mu=float(args.reg_mu),
                continuity_mu=mu,
                steps_per_flex=4,
            )
            _ = residual_jit(jnp.asarray(y_start), jnp.asarray(I0)).block_until_ready()
            _ = Jy_jit(jnp.asarray(y_start), jnp.asarray(I0)).block_until_ready()
            solve = p2c7._solve_branch(
                y0=y_start,
                I=I0,
                residual_jit=residual_jit,
                output_jit=output_jit,
                Jy_jit=Jy_jit,
                max_iter=int(args.max_iter),
            )
            y_start = solve.y
            cont_norm = float(
                np.linalg.norm(
                    np.asarray(jax.device_get(continuity_jit(jnp.asarray(solve.y), jnp.asarray(I0))), dtype=float)
                )
            )
            row = {
                "case": case.name,
                "continuity_mu": mu,
                "success": solve.success,
                "steady_s": solve.steady_s,
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "continuity_norm": cont_norm,
                "y_norm": float(np.linalg.norm(solve.y)),
            }
            continuation_rows.append(row)
            if solve.success and (best_success is None or row["stationarity_norm"] < best_success["stationarity_norm"]):
                best_success = row
            if best_stationary is None or row["stationarity_norm"] < best_stationary["stationarity_norm"]:
                best_stationary = row

            active = _active_direction(I0)
            explicit = np.asarray(jax.device_get(TI_jit(jnp.asarray(solve.y), jnp.asarray(I0))), dtype=float) @ active
            explicit_rows.append(
                {
                    "case": case.name,
                    "continuity_mu": mu,
                    "active_coil": int(np.argmax(np.abs(active))),
                    "fixed_y_explicit_task_response_norm": float(np.linalg.norm(explicit)),
                    "fixed_y_explicit_task_response": explicit.tolist(),
                }
            )
        if best_success is None:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "case": case.name,
                    "failure_class": "F1_baseline_unstable",
                    "evidence": {"best_stationary": best_stationary},
                    "recovery_action": "inspect rigid-load pose coupling or use pseudo-arclength continuation",
                },
            )

    coupling = _rigid_coupling_audit()
    hard_recovered = [row for row in continuation_rows if row["success"] and row["continuity_norm"] <= 1.0e-3]
    explicit_nonzero = [row for row in explicit_rows if row["fixed_y_explicit_task_response_norm"] > 1.0e-12]
    classification = "recovered" if hard_recovered else "model-coupling-suspect"
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "cases": sorted(case_names),
            "continuity_mu_values": mu_values,
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "hard_recovered_count": len(hard_recovered),
        "explicit_nonzero_count": len(explicit_nonzero),
        "continuation_rows": continuation_rows,
        "explicit_response_rows": explicit_rows,
        "rigid_coupling_audit": coupling,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "continuation_summary.csv", continuation_rows)
    _write_csv(OUT_DIR / "explicit_response_summary.csv", explicit_rows)
    (OUT_DIR / "rigid_coupling_audit.json").write_text(json.dumps(_jsonable(coupling), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
