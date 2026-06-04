"""Run strict P2c9 gates for shooting task sensitivity."""

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


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    directions: list[tuple[str, np.ndarray]] = []
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    directions.append((f"active_coil_{active}", e))
    for i in range(int(random_count)):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        directions.append((f"random_{i}", d))
    return directions


def _seed_rows(
    *,
    case_name: str,
    y_oracle: np.ndarray,
    I0: np.ndarray,
    residual_jit: Any,
    output_jit: Any,
    Jy_jit: Any,
    continuity_jit: Any,
    max_iter: int,
) -> tuple[p2c7.AuditSolve | None, list[dict[str, Any]]]:
    seed_specs = [
        ("oracle_y0", np.asarray(y_oracle, dtype=float)),
        ("zero_y0", np.zeros_like(y_oracle, dtype=float)),
        ("neighbor_y0", np.asarray(y_oracle, dtype=float) + 0.1 * RNG.normal(size=np.asarray(y_oracle).shape)),
    ]
    rows: list[dict[str, Any]] = []
    solves: dict[str, p2c7.AuditSolve] = {}
    for seed_name, y0 in seed_specs:
        solve = p2c7._solve_branch(
            y0=y0,
            I=I0,
            residual_jit=residual_jit,
            output_jit=output_jit,
            Jy_jit=Jy_jit,
            max_iter=max_iter,
        )
        solves[seed_name] = solve
        cont_norm = float(
            np.linalg.norm(np.asarray(jax.device_get(continuity_jit(jnp.asarray(solve.y), jnp.asarray(I0))), dtype=float))
        )
        rows.append(
            {
                "case": case_name,
                "seed": seed_name,
                "success": solve.success,
                "steady_s": solve.steady_s,
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "continuity_norm": cont_norm,
                "y_norm": float(np.linalg.norm(solve.y)),
            }
        )
    base = solves.get("oracle_y0")
    if base is None or not base.success:
        return None, rows
    base_tip = np.asarray(base.out["tip_p"], dtype=float)
    for row in rows:
        solve = solves[str(row["seed"])]
        row["task_jump_from_oracle_seed"] = float(np.linalg.norm(np.asarray(solve.out["tip_p"], dtype=float) - base_tip))
        row["y_jump_from_oracle_seed"] = float(np.linalg.norm(solve.y - base.y))
        row["branch_consistent_with_oracle_seed"] = bool(row["task_jump_from_oracle_seed"] <= 1.0e-3 and row["y_jump_from_oracle_seed"] <= 5.0)
    return base, rows


def _eps_plateau(values: list[np.ndarray]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    norms = [float(np.linalg.norm(v)) for v in values]
    mean_norm = float(np.mean(norms))
    rel_spread = (max(norms) - min(norms)) / max(mean_norm, 1.0e-12)
    stack = np.stack(values, axis=0)
    mean_vec = np.mean(stack, axis=0)
    vec_spread = float(np.max(np.linalg.norm(stack - mean_vec, axis=1)) / max(np.linalg.norm(mean_vec), 1.0e-12))
    return rel_spread, vec_spread


def _classify_row(
    *,
    rel_err: float,
    cosine: float,
    fd_norm: float,
    chain_norm: float,
    branch_jump: bool,
    stationarity_norm: float,
    plus_success: bool,
    minus_success: bool,
    plus_boundary: float,
    minus_boundary: float,
    condition_number: float,
) -> tuple[str, str]:
    if not plus_success or not minus_success:
        return "untrusted", "F7_reopt_solver_instability"
    if stationarity_norm > 1.0e-2:
        return "nonstationary-diagnostic", "F1_nonstationary_inner_solve"
    if branch_jump:
        return "untrusted", "F4_branch_jump"
    if plus_boundary > 0.2 or minus_boundary > 0.2:
        return "untrusted", "F3_physics_leak"
    if condition_number > 1.0e8:
        return "untrusted", "F5_ill_conditioned_linear_system"
    if fd_norm <= 1.0e-9 and chain_norm <= 1.0e-9:
        return "diagnostic-flat", "F10_observability_zero_task"
    if rel_err <= 0.3 and cosine > 0.0:
        return "usable", "PASS"
    return "untrusted", "F5_sensitivity_mismatch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", default="0.02,0.05,0.10")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--random_count", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    seed_rows_all: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    cases = p2c4._cases()[:2] if args.quick else p2c4._cases()

    for case in cases:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")

        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "compile"})
        residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, JI_jit, Ty_jit, TI_jit = p2c8._make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_oracle,
            reg_mu=float(args.reg_mu),
            continuity_mu=float(args.continuity_mu),
            steps_per_flex=4,
        )
        _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = Jy_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = JI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = Ty_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = TI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()

        base, seed_rows = _seed_rows(
            case_name=case.name,
            y_oracle=y_oracle,
            I0=I0,
            residual_jit=residual_jit,
            output_jit=output_jit,
            Jy_jit=Jy_jit,
            continuity_jit=continuity_jit,
            max_iter=int(args.max_iter),
        )
        seed_rows_all.extend(seed_rows)
        solve_rows.extend([{**row, "mode": "cont"} for row in seed_rows])
        if base is None:
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "failure_class": "F1_baseline_unstable", "evidence": "oracle_y0 branch failed"})
            continue

        for dir_name, direction in _directions(I0, args.random_count):
            fd_vectors: list[np.ndarray] = []
            per_eps_indices: list[int] = []
            chain = p2c7._implicit_chain(
                y=base.y,
                I=I0,
                direction=direction,
                residual_jit=residual_jit,
                task_jit=task_jit,
                Jy_jit=Jy_jit,
                JI_jit=JI_jit,
                Ty_jit=Ty_jit,
                TI_jit=TI_jit,
            )
            for eps in eps_values:
                plus = p2c7._solve_branch(
                    y0=base.y,
                    I=I0 + eps * direction,
                    residual_jit=residual_jit,
                    output_jit=output_jit,
                    Jy_jit=Jy_jit,
                    max_iter=int(args.max_iter),
                )
                minus = p2c7._solve_branch(
                    y0=base.y,
                    I=I0 - eps * direction,
                    residual_jit=residual_jit,
                    output_jit=output_jit,
                    Jy_jit=Jy_jit,
                    max_iter=int(args.max_iter),
                )
                task_plus = np.asarray(jax.device_get(task_jit(jnp.asarray(plus.y), jnp.asarray(I0 + eps * direction))), dtype=float)
                task_minus = np.asarray(jax.device_get(task_jit(jnp.asarray(minus.y), jnp.asarray(I0 - eps * direction))), dtype=float)
                fd_total = (task_plus - task_minus) / (2.0 * eps)
                fd_vectors.append(fd_total)
                rel_total, cos_total, fd_norm = p2c7._safe_rel(np.asarray(chain["total"], dtype=float), fd_total)
                back = p2c7._solve_branch(
                    y0=plus.y,
                    I=I0,
                    residual_jit=residual_jit,
                    output_jit=output_jit,
                    Jy_jit=Jy_jit,
                    max_iter=int(args.max_iter),
                )
                loop_task_error = float(np.linalg.norm(np.asarray(back.out["tip_p"], dtype=float) - np.asarray(base.out["tip_p"], dtype=float)))
                loop_y_error = float(np.linalg.norm(back.y - base.y))
                branch_jump = bool(
                    loop_task_error > 1.0e-3
                    or loop_y_error > 5.0
                    or np.linalg.norm(plus.y - base.y) > 5.0
                    or np.linalg.norm(minus.y - base.y) > 5.0
                    or plus.boundary_norm > 0.2
                    or minus.boundary_norm > 0.2
                )
                status, failure_class = _classify_row(
                    rel_err=rel_total,
                    cosine=cos_total,
                    fd_norm=fd_norm,
                    chain_norm=float(np.linalg.norm(chain["total"])),
                    branch_jump=branch_jump,
                    stationarity_norm=float(chain["stationarity_norm"]),
                    plus_success=plus.success,
                    minus_success=minus.success,
                    plus_boundary=plus.boundary_norm,
                    minus_boundary=minus.boundary_norm,
                    condition_number=float(chain["condition_number"]),
                )
                row = {
                    "case": case.name,
                    "direction": dir_name,
                    "direction_vector": direction.tolist(),
                    "eps": eps,
                    "status": status,
                    "failure_class": failure_class,
                    "rel_err_total": rel_total,
                    "cosine_total": cos_total,
                    "fd_norm": fd_norm,
                    "rank": chain["rank"],
                    "condition_number": chain["condition_number"],
                    "nullspace_size": chain["nullspace_size"],
                    "stationarity_norm": chain["stationarity_norm"],
                    "base_boundary": base.boundary_norm,
                    "plus_boundary": plus.boundary_norm,
                    "minus_boundary": minus.boundary_norm,
                    "branch_jump": branch_jump,
                    "loop_task_error": loop_task_error,
                    "loop_y_error": loop_y_error,
                    "plus_success": plus.success,
                    "minus_success": minus.success,
                    "chain_norm": float(np.linalg.norm(chain["chain"])),
                    "frozen_norm": float(np.linalg.norm(chain["frozen"])),
                    "total_norm": float(np.linalg.norm(chain["total"])),
                    "dy_dI_direction_norm": chain["dy_dI_direction_norm"],
                    "fd_total_vector": fd_total.tolist(),
                    "implicit_total_vector": np.asarray(chain["total"], dtype=float).tolist(),
                }
                rows.append(row)
                per_eps_indices.append(len(rows) - 1)
                if failure_class != "PASS":
                    _append_jsonl(
                        failures,
                        {
                            "ts": time.time(),
                            "case": case.name,
                            "direction": dir_name,
                            "eps": eps,
                            "failure_class": failure_class,
                            "evidence": {
                                "rel_err": rel_total,
                                "cosine": cos_total,
                                "stationarity_norm": chain["stationarity_norm"],
                                "condition_number": chain["condition_number"],
                                "branch_jump": branch_jump,
                            },
                            "recovery_action": "use hard continuity KKT or true continuous shooting before production IK",
                        },
                    )
                _append_jsonl(
                    heartbeat,
                    {
                        "ts": time.time(),
                        "case": case.name,
                        "phase": "audit",
                        "direction": dir_name,
                        "eps": eps,
                        "status": status,
                        "rel_err": rel_total,
                    },
                )
            norm_spread, vec_spread = _eps_plateau(fd_vectors)
            eps_plateau_ok = bool(norm_spread <= 0.5 and vec_spread <= 0.5)
            for idx in per_eps_indices:
                rows[idx]["eps_norm_rel_spread"] = norm_spread
                rows[idx]["eps_vector_rel_spread"] = vec_spread
                rows[idx]["eps_plateau_ok"] = eps_plateau_ok
                if not eps_plateau_ok and rows[idx]["status"] == "usable":
                    rows[idx]["status"] = "untrusted"
                    rows[idx]["failure_class"] = "F6_eps_inconsistency"
            dir_items = [rows[idx] for idx in per_eps_indices]
            direction_rows.append(
                {
                    "case": case.name,
                    "direction": dir_name,
                    "usable_eps_count": sum(1 for row in dir_items if row["status"] == "usable"),
                    "row_count": len(dir_items),
                    "median_rel_err": float(np.median([row["rel_err_total"] for row in dir_items])),
                    "median_cosine": float(np.median([row["cosine_total"] for row in dir_items])),
                    "eps_norm_rel_spread": norm_spread,
                    "eps_vector_rel_spread": vec_spread,
                    "eps_plateau_ok": eps_plateau_ok,
                    "direction_usable": bool(all(row["status"] == "usable" for row in dir_items)),
                }
            )

    usable = [row for row in rows if row["status"] == "usable"]
    untrusted = [row for row in rows if row["status"] not in {"usable", "diagnostic-flat"}]
    direction_usable = [row for row in direction_rows if row["direction_usable"]]
    if rows and len(usable) == len(rows):
        classification = "usable"
    elif direction_usable:
        classification = "partially-usable"
    elif any(row["status"] == "diagnostic-flat" for row in rows):
        classification = "diagnostic-only"
    else:
        classification = "untrusted"
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "eps_values": eps_values,
            "continuity_mu": float(args.continuity_mu),
            "reg_mu": float(args.reg_mu),
            "quick": bool(args.quick),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "untrusted_count": len(untrusted),
        "row_count": len(rows),
        "direction_usable_count": len(direction_usable),
        "direction_count": len(direction_rows),
        "median_rel_err_total": float(np.median([row["rel_err_total"] for row in rows])) if rows else float("nan"),
        "rows": rows,
        "direction_rows": direction_rows,
        "seed_rows": seed_rows_all,
        "solve_rows": solve_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "direction_summary.csv", direction_rows)
    _write_csv(OUT_DIR / "seed_summary.csv", seed_rows_all)
    _write_csv(OUT_DIR / "solve_summary.csv", solve_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
