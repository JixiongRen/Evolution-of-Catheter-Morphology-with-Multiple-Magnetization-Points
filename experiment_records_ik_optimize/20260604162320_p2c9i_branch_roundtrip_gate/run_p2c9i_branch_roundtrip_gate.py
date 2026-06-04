"""Audit branch consistency and current roundtrip for shooting IK."""

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


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


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


def _case_map() -> dict[str, Any]:
    return {case.name: case for case in p2c4._cases()}


def _build_functions(case: Any, *, continuity_mu: float, reg_mu: float) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, _continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    return {
        "I0": I0,
        "y_oracle": y_oracle,
        "residual_jit": residual_jit,
        "task_jit": task_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
    }


def _active_direction(I0: np.ndarray) -> np.ndarray:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0.0 else 5
    direction = np.zeros(8)
    direction[active] = 1.0
    return direction


def _solve_from_init(bundle: dict[str, Any], y0: np.ndarray, max_iter: int) -> Any:
    return p2c7._solve_branch(
        y0=y0,
        I=bundle["I0"],
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n3_mixed_base_i4,n2_z_pair_rand12,n1_tip_z_rand15,n3_mixed_rand20")
    parser.add_argument("--neighbor", default="n3_mixed_base_i4")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    cases = _case_map()
    t_all = time.perf_counter()

    neighbor_y = None
    if args.neighbor in cases:
        neighbor_bundle = _build_functions(cases[args.neighbor], continuity_mu=args.continuity_mu, reg_mu=args.reg_mu)
        neighbor_solve = _solve_from_init(neighbor_bundle, neighbor_bundle["y_oracle"], args.max_iter)
        if neighbor_solve.success:
            neighbor_y = neighbor_solve.y

    for case_name in [x.strip() for x in args.cases.split(",") if x.strip()]:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case_name, "phase": "build"})
        bundle = _build_functions(cases[case_name], continuity_mu=args.continuity_mu, reg_mu=args.reg_mu)
        y_oracle = bundle["y_oracle"]
        init_map = {
            "oracle": y_oracle,
            "zero": np.zeros_like(y_oracle),
            "half_oracle": 0.5 * y_oracle,
        }
        if neighbor_y is not None and neighbor_y.shape == y_oracle.shape:
            init_map["neighbor"] = neighbor_y

        solves: dict[str, Any] = {}
        tasks: dict[str, np.ndarray] = {}
        for init_name, y0 in init_map.items():
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case_name, "phase": "baseline", "init": init_name})
            solve = _solve_from_init(bundle, y0, args.max_iter)
            solves[init_name] = solve
            task = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(solve.y), jnp.asarray(bundle["I0"]))), dtype=float)
            tasks[init_name] = task
            row = {
                "case": case_name,
                "init": init_name,
                "success": solve.success,
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "steady_s": solve.steady_s,
                "branch_delta_from_oracle": float(np.linalg.norm(solve.y - solves["oracle"].y)) if "oracle" in solves else 0.0,
                "task_delta_from_oracle": float(np.linalg.norm(task - tasks["oracle"])) if "oracle" in tasks else 0.0,
            }
            rows.append(row)

        oracle_solve = solves["oracle"]
        successful = [name for name, solve in solves.items() if solve.success]
        branch_task_spread = 0.0
        branch_state_spread = 0.0
        if len(successful) >= 2:
            branch_task_spread = max(float(np.linalg.norm(tasks[a] - tasks[b])) for a in successful for b in successful)
            branch_state_spread = max(float(np.linalg.norm(solves[a].y - solves[b].y)) for a in successful for b in successful)

        branch_consistent = bool(oracle_solve.success and branch_task_spread <= 1.0e-3 and branch_state_spread <= 5.0)
        case_row = {
            "case": case_name,
            "oracle_success": oracle_solve.success,
            "successful_init_count": len(successful),
            "init_count": len(init_map),
            "branch_task_spread": branch_task_spread,
            "branch_state_spread": branch_state_spread,
            "branch_consistent": branch_consistent,
            "classification": "branch-consistent" if branch_consistent else ("baseline-unstable" if not oracle_solve.success else "branch-ambiguous"),
        }
        case_rows.append(case_row)
        if not branch_consistent:
            _append_jsonl(failures, {"ts": time.time(), "case": case_name, "failure_class": "F4_branch_jump" if oracle_solve.success else "F1_baseline_unstable", "evidence": case_row})

        if oracle_solve.success:
            direction = _active_direction(bundle["I0"])
            I_plus = bundle["I0"] + float(args.step) * direction
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case_name, "phase": "roundtrip_plus"})
            plus = p2c7._solve_branch(
                y0=oracle_solve.y,
                I=I_plus,
                residual_jit=bundle["residual_jit"],
                output_jit=bundle["output_jit"],
                Jy_jit=bundle["Jy_jit"],
                max_iter=args.max_iter,
            )
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case_name, "phase": "roundtrip_back"})
            back = p2c7._solve_branch(
                y0=plus.y,
                I=bundle["I0"],
                residual_jit=bundle["residual_jit"],
                output_jit=bundle["output_jit"],
                Jy_jit=bundle["Jy_jit"],
                max_iter=args.max_iter,
            )
            task_back = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(back.y), jnp.asarray(bundle["I0"]))), dtype=float)
            roundtrip_row = {
                "case": case_name,
                "plus_success": plus.success,
                "back_success": back.success,
                "roundtrip_state_error": float(np.linalg.norm(back.y - oracle_solve.y)),
                "roundtrip_task_error": float(np.linalg.norm(task_back - tasks["oracle"])),
                "plus_boundary": plus.boundary_norm,
                "back_boundary": back.boundary_norm,
                "roundtrip_consistent": bool(plus.success and back.success and np.linalg.norm(task_back - tasks["oracle"]) <= 1.0e-3 and np.linalg.norm(back.y - oracle_solve.y) <= 5.0),
            }
            roundtrip_rows.append(roundtrip_row)
            if not roundtrip_row["roundtrip_consistent"]:
                _append_jsonl(failures, {"ts": time.time(), "case": case_name, "failure_class": "F4_roundtrip_branch_jump", "evidence": roundtrip_row})

    branch_ok = sum(1 for row in case_rows if row["branch_consistent"])
    roundtrip_ok = sum(1 for row in roundtrip_rows if row["roundtrip_consistent"])
    classification = "usable" if case_rows and branch_ok == len(case_rows) and roundtrip_ok == len(roundtrip_rows) else ("partially-usable" if branch_ok or roundtrip_ok else "diagnostic-only")
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "branch_consistent_count": branch_ok,
        "case_count": len(case_rows),
        "roundtrip_consistent_count": roundtrip_ok,
        "roundtrip_count": len(roundtrip_rows),
        "case_rows": case_rows,
        "rows": rows,
        "roundtrip_rows": roundtrip_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "case_summary.csv", case_rows)
    _write_csv(OUT_DIR / "roundtrip_summary.csv", roundtrip_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
