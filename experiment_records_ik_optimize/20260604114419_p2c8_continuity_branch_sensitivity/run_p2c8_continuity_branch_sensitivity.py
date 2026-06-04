"""Run P2c8 continuity-regularized shooting sensitivity audits."""

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
P2C5_DIR = ROOT / "experiment_records_ik_optimize" / "20260603204650_p2c5_jax_clean_shooting_speedup"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C5_DIR, P2C7_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_terminal_shooting_jax import (
    CleanShootingWeights,
    _advance_rigid,
    _integrate_flex,
    clean_terminal_residual,
    forward_segment_wrench,
    initial_state,
    normalize_state,
    output_to_numpy,
)

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
WEIGHTS = CleanShootingWeights(tip_weight=120.0, so3_weight=2.0, boundary_weight=0.5)
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


def interface_wrench_residual(params: Any, y: jnp.ndarray, *, steps_per_flex: int = 4) -> jnp.ndarray:
    """Return segment-interface wrench jumps in normalized shooting units."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    residuals = []
    f_ref = jnp.maximum(float(params.scales.F_ref), 1.0e-12)
    m_ref = jnp.maximum(float(params.scales.M_ref), 1.0e-12)
    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = normalize_state(x)
        x, _flex_points = _integrate_flex(params, x, i, steps_per_flex)
        x, _mid_p, _mid_q = _advance_rigid(params, x, i)
        if i + 1 < len(params.flex):
            dist_block = jnp.concatenate([x[7:10] / f_ref, x[10:13] / m_ref], axis=0)
            residuals.append(blocks[i + 1] - dist_block)
    if not residuals:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(residuals, axis=0)


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    y_ref: np.ndarray,
    reg_mu: float,
    continuity_mu: float,
    steps_per_flex: int,
):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))
    cont_scale = jnp.sqrt(jnp.asarray(continuity_mu, dtype=jnp.float64))

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        clean = clean_terminal_residual(
            params,
            y,
            target_p=target_p_j,
            target_q=target_q_j,
            weights=WEIGHTS,
            steps_per_flex=steps_per_flex,
            so3_scale=1.0,
        )
        reg = reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        continuity = cont_scale * interface_wrench_residual(params, y, steps_per_flex=steps_per_flex)
        return jnp.concatenate([clean.reshape(-1), reg.reshape(-1), continuity.reshape(-1)], axis=0)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = forward_segment_wrench(params, y, steps_per_flex=steps_per_flex)
        return jnp.concatenate([out.tip_p.reshape(-1), p2c7._so3_vec(out.tip_q, target_q_j).reshape(-1)], axis=0)

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)

    def continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return interface_wrench_residual(params_for_I(I), y, steps_per_flex=steps_per_flex)

    residual_jit = jax.jit(residual_yI)
    task_jit = jax.jit(task_yI)
    output_jit = jax.jit(output_yI)
    continuity_jit = jax.jit(continuity_yI)
    Jy_jit = jax.jit(jax.jacfwd(residual_yI, argnums=0))
    JI_jit = jax.jit(jax.jacfwd(residual_yI, argnums=1))
    Ty_jit = jax.jit(jax.jacfwd(task_yI, argnums=0))
    TI_jit = jax.jit(jax.jacfwd(task_yI, argnums=1))
    return residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, JI_jit, Ty_jit, TI_jit


def _case_subset(args: argparse.Namespace) -> list[Any]:
    cases = p2c4._cases()
    if args.quick:
        return [cases[0], cases[2]]
    return cases


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


def _mode_specs(modes: str, reg_mu: float) -> list[tuple[str, float, float]]:
    specs = []
    for raw in modes.split(","):
        name = raw.strip()
        if not name:
            continue
        if name == "reg_only":
            specs.append((name, float(reg_mu), 0.0))
        elif name.startswith("cont_"):
            specs.append((name, float(reg_mu), float(name.split("_", 1)[1])))
        else:
            raise ValueError(f"Unsupported mode: {name}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a two-case audit.")
    parser.add_argument("--eps", default="0.05")
    parser.add_argument("--modes", default="reg_only,cont_0.01,cont_0.1,cont_1.0")
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--random_count", type=int, default=2)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    mode_specs = _mode_specs(args.modes, args.reg_mu)
    rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in _case_subset(args):
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")

        for mode_name, reg_mu, continuity_mu in mode_specs:
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "mode": mode_name, "phase": "compile"})
            residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, JI_jit, Ty_jit, TI_jit = _make_functions(
                params_base=params_base,
                target_p=target_p,
                target_q=target_q,
                y_ref=y_oracle,
                reg_mu=reg_mu,
                continuity_mu=continuity_mu,
                steps_per_flex=4,
            )
            _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
            _ = Jy_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
            _ = JI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
            _ = Ty_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
            _ = TI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()

            base = p2c7._solve_branch(
                y0=y_oracle,
                I=I0,
                residual_jit=residual_jit,
                output_jit=output_jit,
                Jy_jit=Jy_jit,
                max_iter=int(args.max_iter),
            )
            cont_norm = float(
                np.linalg.norm(
                    np.asarray(jax.device_get(continuity_jit(jnp.asarray(base.y), jnp.asarray(I0))), dtype=float)
                )
            )
            solve_rows.append(
                {
                    "case": case.name,
                    "mode": mode_name,
                    "success": base.success,
                    "steady_s": base.steady_s,
                    "residual_norm": base.residual_norm,
                    "stationarity_norm": base.stationarity_norm,
                    "boundary_residual_norm": base.boundary_norm,
                    "continuity_norm": cont_norm,
                }
            )
            if not base.success:
                _append_jsonl(
                    failures,
                    {
                        "ts": time.time(),
                        "case": case.name,
                        "mode": mode_name,
                        "failure_class": "F2_numerical_fail",
                        "evidence": "base branch solve failed",
                    },
                )
                continue

            for dir_name, direction in _directions(I0, args.random_count):
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
                    task_plus = np.asarray(
                        jax.device_get(task_jit(jnp.asarray(plus.y), jnp.asarray(I0 + eps * direction))),
                        dtype=float,
                    )
                    task_minus = np.asarray(
                        jax.device_get(task_jit(jnp.asarray(minus.y), jnp.asarray(I0 - eps * direction))),
                        dtype=float,
                    )
                    fd_total = (task_plus - task_minus) / (2.0 * eps)
                    rel_total, cos_total, fd_norm = p2c7._safe_rel(np.asarray(chain["total"], dtype=float), fd_total)
                    branch_jump = bool(
                        np.linalg.norm(plus.y - base.y) > 5.0
                        or np.linalg.norm(minus.y - base.y) > 5.0
                        or plus.boundary_norm > 0.2
                        or minus.boundary_norm > 0.2
                    )
                    status = "untrusted"
                    if base.stationarity_norm > 1.0e-2:
                        status = "nonstationary-diagnostic"
                    if rel_total <= 0.3 and cos_total > 0.0 and not branch_jump and fd_norm > 1.0e-9 and base.stationarity_norm <= 1.0e-2:
                        status = "usable"
                    elif fd_norm <= 1.0e-9 and float(np.linalg.norm(chain["total"])) <= 1.0e-9 and not branch_jump:
                        status = "diagnostic-only"
                    if status in {"untrusted", "nonstationary-diagnostic"}:
                        _append_jsonl(
                            failures,
                            {
                                "ts": time.time(),
                                "case": case.name,
                                "mode": mode_name,
                                "direction": dir_name,
                                "eps": eps,
                                "failure_class": "F6_nonstationary_branch" if status == "nonstationary-diagnostic" else "F5_sensitivity_mismatch",
                                "evidence": {
                                    "rel_err": rel_total,
                                    "cosine": cos_total,
                                    "stationarity_norm": base.stationarity_norm,
                                    "branch_jump": branch_jump,
                                },
                                "recovery_action": "increase physical branch constraints or switch to continuous initial-value shooting sensitivity",
                            },
                        )
                    row = {
                        "case": case.name,
                        "mode": mode_name,
                        "reg_mu": reg_mu,
                        "continuity_mu": continuity_mu,
                        "direction": dir_name,
                        "direction_vector": direction.tolist(),
                        "eps": eps,
                        "status": status,
                        "rel_err_total": rel_total,
                        "cosine_total": cos_total,
                        "fd_norm": fd_norm,
                        "rank": chain["rank"],
                        "condition_number": chain["condition_number"],
                        "nullspace_size": chain["nullspace_size"],
                        "stationarity_norm": chain["stationarity_norm"],
                        "base_boundary": base.boundary_norm,
                        "base_continuity_norm": cont_norm,
                        "plus_boundary": plus.boundary_norm,
                        "minus_boundary": minus.boundary_norm,
                        "branch_jump": branch_jump,
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
                    _append_jsonl(
                        heartbeat,
                        {
                            "ts": time.time(),
                            "case": case.name,
                            "mode": mode_name,
                            "phase": "audit",
                            "direction": dir_name,
                            "eps": eps,
                            "status": status,
                            "rel_err": rel_total,
                        },
                    )

    usable = [row for row in rows if row["status"] == "usable"]
    untrusted = [row for row in rows if row["status"] in {"untrusted", "nonstationary-diagnostic"}]
    median_rel_by_mode = {
        mode: float(np.median([row["rel_err_total"] for row in rows if row["mode"] == mode]))
        for mode, _reg, _cont in mode_specs
        if any(row["mode"] == mode for row in rows)
    }
    best_mode = min(median_rel_by_mode, key=median_rel_by_mode.get) if median_rel_by_mode else None
    if rows and len(usable) == len(rows):
        classification = "usable"
    elif usable and untrusted:
        classification = "partially-usable"
    elif usable:
        classification = "diagnostic-partial"
    else:
        classification = "diagnostic-only"
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "eps_values": eps_values,
            "modes": [name for name, _reg, _cont in mode_specs],
            "quick": bool(args.quick),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "untrusted_count": len(untrusted),
        "row_count": len(rows),
        "median_rel_err_by_mode": median_rel_by_mode,
        "best_mode_by_median_rel": best_mode,
        "rows": rows,
        "solve_rows": solve_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "solve_summary.csv", solve_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
