"""Run P2c7 shooting sensitivity audits.

The audit checks internal chain consistency for a branch-conditioned shooting
problem. FK oracle is not used as the primary truth signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C5_DIR = ROOT / "experiment_records_ik_optimize" / "20260603204650_p2c5_jax_clean_shooting_speedup"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C5_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import (
    CleanShootingWeights,
    clean_terminal_residual,
    forward_segment_wrench,
    output_to_numpy,
)

import run_clean_multi_case as p2c4
import run_jax_clean_shooting as p2c5
import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
WEIGHTS = CleanShootingWeights(tip_weight=120.0, so3_weight=2.0, boundary_weight=0.5)
RNG = np.random.default_rng(20260604)


@dataclass
class AuditSolve:
    """Branch-conditioned solve result."""

    y: np.ndarray
    out: dict[str, np.ndarray]
    success: bool
    n_iter: int
    reject_count: int
    lambda_final: float
    steady_s: float
    residual_norm: float
    stationarity_norm: float
    boundary_norm: float


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


def _so3_vec(current_q: Any, target_q: Any) -> jnp.ndarray:
    return quaternion_so3_residual(jnp.asarray(current_q), jnp.asarray(target_q), sigma_angle=1.0)


def _task_vector(params: Any, y: jnp.ndarray, target_q: jnp.ndarray, *, steps_per_flex: int) -> jnp.ndarray:
    out = forward_segment_wrench(params, y, steps_per_flex=steps_per_flex)
    return jnp.concatenate([out.tip_p.reshape(-1), _so3_vec(out.tip_q, target_q).reshape(-1)], axis=0)


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    y_ref: np.ndarray,
    reg_mu: float,
    steps_per_flex: int,
):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))

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
        return jnp.concatenate([clean.reshape(-1), reg.reshape(-1)], axis=0)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return _task_vector(params_for_I(I), y, target_q_j, steps_per_flex=steps_per_flex)

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)

    residual_jit = jax.jit(residual_yI)
    task_jit = jax.jit(task_yI)
    output_jit = jax.jit(output_yI)
    Jy_jit = jax.jit(jax.jacfwd(residual_yI, argnums=0))
    JI_jit = jax.jit(jax.jacfwd(residual_yI, argnums=1))
    Ty_jit = jax.jit(jax.jacfwd(task_yI, argnums=0))
    TI_jit = jax.jit(jax.jacfwd(task_yI, argnums=1))
    return residual_jit, task_jit, output_jit, Jy_jit, JI_jit, Ty_jit, TI_jit


def _solve_branch(
    *,
    y0: np.ndarray,
    I: np.ndarray,
    residual_jit: Any,
    output_jit: Any,
    Jy_jit: Any,
    max_iter: int = 24,
    step_cap: float = 5.0,
) -> AuditSolve:
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(residual_jit(jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(Jy_jit(jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=int(max_iter),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(residual_jit(y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(Jy_jit(y_j, I_j)), dtype=float)
    out = output_to_numpy(output_jit(y_j, I_j))
    stationarity = J.T @ r
    boundary = float(np.linalg.norm(out["boundary"]))
    success = bool(np.all(np.isfinite(y)) and boundary <= 0.3 and np.linalg.norm(r) <= 1.0e-2)
    return AuditSolve(
        y=y,
        out=out,
        success=success,
        n_iter=int(opt.nfev),
        reject_count=0,
        lambda_final=0.0,
        steady_s=float(steady_s),
        residual_norm=float(np.linalg.norm(r)),
        stationarity_norm=float(np.linalg.norm(stationarity)),
        boundary_norm=boundary,
    )


def _implicit_chain(
    *,
    y: np.ndarray,
    I: np.ndarray,
    direction: np.ndarray,
    residual_jit: Any,
    task_jit: Any,
    Jy_jit: Any,
    JI_jit: Any,
    Ty_jit: Any,
    TI_jit: Any,
) -> dict[str, Any]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64)
    d_j = jnp.asarray(direction, dtype=jnp.float64)
    r = np.asarray(jax.device_get(residual_jit(y_j, I_j)), dtype=float)
    Jy = np.asarray(jax.device_get(Jy_jit(y_j, I_j)), dtype=float)
    JI = np.asarray(jax.device_get(JI_jit(y_j, I_j)), dtype=float)
    Ty = np.asarray(jax.device_get(Ty_jit(y_j, I_j)), dtype=float)
    TI = np.asarray(jax.device_get(TI_jit(y_j, I_j)), dtype=float)
    H = Jy.T @ Jy
    D = np.clip(np.diag(H), 1e-12, None)
    H_reg = H + 1e-8 * np.diag(D)
    rhs = Jy.T @ (JI @ np.asarray(direction, dtype=float))
    try:
        dy = -np.linalg.solve(H_reg, rhs)
    except np.linalg.LinAlgError:
        dy = -np.linalg.lstsq(H_reg, rhs, rcond=None)[0]
    frozen = TI @ np.asarray(direction, dtype=float)
    chain = Ty @ dy
    total = frozen + chain
    svals = np.linalg.svd(Jy, compute_uv=False)
    tol = max(Jy.shape) * np.finfo(float).eps * (float(svals[0]) if svals.size else 0.0)
    rank = int(np.sum(svals > tol))
    cond = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    return {
        "r_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(Jy.T @ r)),
        "rank": rank,
        "condition_number": cond,
        "nullspace_size": int(Jy.shape[1] - rank),
        "frozen": frozen,
        "chain": chain,
        "total": total,
        "dy_dI_direction_norm": float(np.linalg.norm(dy)),
        "singular_values": svals.tolist(),
    }


def _safe_rel(pred: np.ndarray, ref: np.ndarray, floor: float = 1e-9) -> tuple[float, float, float]:
    ref_norm = float(np.linalg.norm(ref))
    err = float(np.linalg.norm(pred - ref))
    rel = err / max(ref_norm, floor)
    denom = max(float(np.linalg.norm(pred)) * ref_norm, floor)
    cosine = float(np.dot(pred, ref) / denom)
    return rel, cosine, ref_norm


def _case_subset(args: argparse.Namespace) -> list[Any]:
    cases = p2c4._cases()
    if args.quick:
        return [cases[0], cases[2]]
    return cases


def _sampled_jacobian_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sampled-Jacobian fallback summaries in the sampled subspace."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["case"]), float(row["eps"])), []).append(row)
    summaries = []
    for (case, eps), items in groups.items():
        if len(items) < 2:
            continue
        D = np.stack([np.asarray(item["direction_vector"], dtype=float) for item in items], axis=1)
        F = np.stack([np.asarray(item["fd_total_vector"], dtype=float) for item in items], axis=1)
        J_sample = F @ np.linalg.pinv(D)
        pred = J_sample @ D
        rel = float(np.linalg.norm(pred - F) / max(float(np.linalg.norm(F)), 1e-12))
        fd_norms = [float(item["fd_norm"]) for item in items]
        summaries.append(
            {
                "case": case,
                "eps": eps,
                "sample_count": len(items),
                "sampled_subspace_rel_err": rel,
                "fd_norm_min": float(np.min(fd_norms)),
                "fd_norm_max": float(np.max(fd_norms)),
                "rank_directions": int(np.linalg.matrix_rank(D)),
                "fallback_status": "sampled_jacobian_available" if rel <= 1e-9 else "sampled_jacobian_diagnostic",
                "note": "This is a local sampled fallback in the measured direction subspace, not a production gradient.",
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a two-case smoke audit.")
    parser.add_argument("--eps", default="0.02,0.05,0.10")
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=20)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
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
        residual_jit, task_jit, output_jit, Jy_jit, JI_jit, Ty_jit, TI_jit = _make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_oracle,
            reg_mu=float(args.reg_mu),
            steps_per_flex=4,
        )
        # Compile on the oracle branch.
        _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = Jy_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = JI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = Ty_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = TI_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()

        base = _solve_branch(
            y0=y_oracle,
            I=I0,
            residual_jit=residual_jit,
            output_jit=output_jit,
            Jy_jit=Jy_jit,
            max_iter=int(args.max_iter),
        )
        solve_rows.append(
            {
                "case": case.name,
                "solve": "base_oracle_y0",
                "success": base.success,
                "steady_s": base.steady_s,
                "residual_norm": base.residual_norm,
                "stationarity_norm": base.stationarity_norm,
                "boundary_residual_norm": base.boundary_norm,
            }
        )
        if not base.success:
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "failure_class": "F2_numerical_fail", "evidence": "base branch solve failed"})
            continue

        directions: list[tuple[str, np.ndarray]] = []
        active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
        e = np.zeros(8)
        e[active] = 1.0
        directions.append((f"active_coil_{active}", e))
        for i in range(2):
            d = RNG.normal(size=8)
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            directions.append((f"random_{i}", d))

        for dir_name, direction in directions:
            chain = _implicit_chain(
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
                plus = _solve_branch(
                    y0=base.y,
                    I=I0 + eps * direction,
                    residual_jit=residual_jit,
                    output_jit=output_jit,
                    Jy_jit=Jy_jit,
                    max_iter=int(args.max_iter),
                )
                minus = _solve_branch(
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
                rel_total, cos_total, fd_norm = _safe_rel(np.asarray(chain["total"], dtype=float), fd_total)
                rel_frozen, cos_frozen, _ = _safe_rel(np.asarray(chain["frozen"], dtype=float), fd_total)
                branch_jump = bool(
                    np.linalg.norm(plus.y - base.y) > 5.0
                    or np.linalg.norm(minus.y - base.y) > 5.0
                    or plus.boundary_norm > 0.2
                    or minus.boundary_norm > 0.2
                )
                total_norm = float(np.linalg.norm(chain["total"]))
                status = "untrusted"
                if rel_total <= 0.3 and cos_total > 0.0 and not branch_jump and fd_norm > 1e-9:
                    status = "usable"
                elif fd_norm <= 1e-9 and total_norm <= 1e-9 and not branch_jump:
                    status = "diagnostic-only"
                if status == "untrusted":
                    failure_class = "F4_branch_jump" if branch_jump else "F5_sensitivity_mismatch"
                    _append_jsonl(
                        failures,
                        {
                            "ts": time.time(),
                            "case": case.name,
                            "direction": dir_name,
                            "eps": eps,
                            "failure_class": failure_class,
                            "evidence": {"rel_err": rel_total, "cosine": cos_total, "branch_jump": branch_jump},
                            "recovery_action": "try branch regularization/continuation or sampled Jacobian fallback",
                        },
                    )
                row = {
                    "case": case.name,
                    "direction": dir_name,
                    "direction_vector": direction.tolist(),
                    "eps": eps,
                    "status": status,
                    "rel_err_total": rel_total,
                    "cosine_total": cos_total,
                    "rel_err_frozen": rel_frozen,
                    "cosine_frozen": cos_frozen,
                    "fd_norm": fd_norm,
                    "rank": chain["rank"],
                    "condition_number": chain["condition_number"],
                    "nullspace_size": chain["nullspace_size"],
                    "stationarity_norm": chain["stationarity_norm"],
                    "base_boundary": base.boundary_norm,
                    "plus_boundary": plus.boundary_norm,
                    "minus_boundary": minus.boundary_norm,
                    "branch_jump": branch_jump,
                    "plus_success": plus.success,
                    "minus_success": minus.success,
                    "chain_norm": float(np.linalg.norm(chain["chain"])),
                    "frozen_norm": float(np.linalg.norm(chain["frozen"])),
                    "total_norm": total_norm,
                    "dy_dI_direction_norm": chain["dy_dI_direction_norm"],
                    "fd_total_vector": fd_total.tolist(),
                    "implicit_total_vector": np.asarray(chain["total"], dtype=float).tolist(),
                    "frozen_vector": np.asarray(chain["frozen"], dtype=float).tolist(),
                    "chain_vector": np.asarray(chain["chain"], dtype=float).tolist(),
                }
                rows.append(row)
                _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "audit", "direction": dir_name, "eps": eps, "status": status, "rel_err": rel_total})

    usable = [row for row in rows if row["status"] == "usable"]
    diagnostic = [row for row in rows if row["status"] == "diagnostic-only"]
    median_rel = float(np.median([row["rel_err_total"] for row in rows])) if rows else float("nan")
    classification = "usable" if usable and median_rel <= 0.3 else ("diagnostic-only" if diagnostic else "untrusted")
    sampled_summaries = _sampled_jacobian_summaries(rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "reg_mu": float(args.reg_mu),
            "eps_values": eps_values,
            "quick": bool(args.quick),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "diagnostic_count": len(diagnostic),
        "row_count": len(rows),
        "median_rel_err_total": median_rel,
        "sampled_jacobian_summaries": sampled_summaries,
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
