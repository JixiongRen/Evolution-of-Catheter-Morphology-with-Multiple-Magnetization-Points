"""Probe true continuous initial-value shooting sensitivity."""

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
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import (
    CleanShootingOutput,
    CleanShootingWeights,
    _advance_rigid,
    _integrate_flex,
    initial_state,
    output_to_numpy,
)

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
WEIGHTS = CleanShootingWeights(tip_weight=120.0, so3_weight=2.0, boundary_weight=0.5)
RNG = np.random.default_rng(20260604)


@dataclass
class SolveResult:
    """Continuous-shooting solve result."""

    y: np.ndarray
    out: dict[str, np.ndarray]
    success: bool
    n_iter: int
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


def forward_continuous_wrench(params: Any, y: jnp.ndarray, *, steps_per_flex: int = 4) -> CleanShootingOutput:
    """Integrate a single continuous base-wrench shooting state."""
    x = initial_state(params, jnp.asarray(y, dtype=jnp.float64).reshape(6))
    centerline_parts = [x[0:3].reshape(1, 3)]
    rigid_mid_p = []
    rigid_mid_q = []
    rigid_dist_p = []
    rigid_dist_q = []
    for i in range(len(params.flex)):
        x, flex_points = _integrate_flex(params, x, i, steps_per_flex)
        centerline_parts.append(flex_points)
        x, mid_p, mid_q = _advance_rigid(params, x, i)
        rigid_mid_p.append(mid_p)
        rigid_mid_q.append(mid_q)
        rigid_dist_p.append(x[0:3])
        rigid_dist_q.append(x[3:7])
        centerline_parts.append(x[0:3].reshape(1, 3))
    boundary = jnp.concatenate(
        [
            x[7:10] / jnp.maximum(float(params.scales.F_ref), 1.0e-12),
            x[10:13] / jnp.maximum(float(params.scales.M_ref), 1.0e-12),
        ],
        axis=0,
    )
    return CleanShootingOutput(
        centerline=jnp.concatenate(centerline_parts, axis=0),
        rigid_mid_p=jnp.stack(rigid_mid_p, axis=0),
        rigid_mid_q=jnp.stack(rigid_mid_q, axis=0),
        rigid_dist_p=jnp.stack(rigid_dist_p, axis=0),
        rigid_dist_q=jnp.stack(rigid_dist_q, axis=0),
        tip_p=x[0:3],
        tip_q=x[3:7],
        boundary=boundary,
    )


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
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(6)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_continuous_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        r_p = float(WEIGHTS.tip_weight) * (out.tip_p - target_p_j)
        r_q = float(WEIGHTS.so3_weight) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        r_b = float(WEIGHTS.boundary_weight) * out.boundary
        r_reg = reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(6) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_q.reshape(-1), r_b.reshape(-1), r_reg.reshape(-1)], axis=0)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_continuous_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        return jnp.concatenate(
            [out.tip_p.reshape(-1), quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0).reshape(-1)],
            axis=0,
        )

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return forward_continuous_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)

    residual_jit = jax.jit(residual_yI)
    task_jit = jax.jit(task_yI)
    output_jit = jax.jit(output_yI)
    Jy_jit = jax.jit(jax.jacfwd(residual_yI, argnums=0))
    JI_jit = jax.jit(jax.jacfwd(residual_yI, argnums=1))
    Ty_jit = jax.jit(jax.jacfwd(task_yI, argnums=0))
    TI_jit = jax.jit(jax.jacfwd(task_yI, argnums=1))
    return residual_jit, task_jit, output_jit, Jy_jit, JI_jit, Ty_jit, TI_jit


def _solve(
    *,
    y0: np.ndarray,
    I: np.ndarray,
    residual_jit: Any,
    output_jit: Any,
    Jy_jit: Any,
    max_iter: int,
) -> SolveResult:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(residual_jit(jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float),
        np.asarray(y0, dtype=float).reshape(6),
        jac=lambda yy: np.asarray(jax.device_get(Jy_jit(jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones(6),
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
    success = bool(np.all(np.isfinite(y)) and np.linalg.norm(r) <= 1.0e-2 and boundary <= 0.2 and np.linalg.norm(stationarity) <= 1.0e-2)
    return SolveResult(
        y=y,
        out=out,
        success=success,
        n_iter=int(opt.nfev),
        steady_s=float(steady_s),
        residual_norm=float(np.linalg.norm(r)),
        stationarity_norm=float(np.linalg.norm(stationarity)),
        boundary_norm=boundary,
    )


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    out = []
    e = np.zeros(8)
    e[active] = 1.0
    out.append((f"active_coil_{active}", e))
    for i in range(random_count):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        out.append((f"random_{i}", d))
    return out


def _implicit_chain(
    *,
    y: np.ndarray,
    I: np.ndarray,
    direction: np.ndarray,
    residual_jit: Any,
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
    D = np.clip(np.diag(H), 1.0e-12, None)
    rhs = Jy.T @ (JI @ np.asarray(direction, dtype=float))
    dy = -np.linalg.solve(H + 1.0e-8 * np.diag(D), rhs)
    frozen = TI @ np.asarray(direction, dtype=float)
    total = frozen + Ty @ dy
    svals = np.linalg.svd(Jy, compute_uv=False)
    rank = int(np.sum(svals > max(Jy.shape) * np.finfo(float).eps * max(float(svals[0]), 1.0e-12)))
    cond = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    return {
        "r_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(Jy.T @ r)),
        "rank": rank,
        "condition_number": cond,
        "nullspace_size": int(Jy.shape[1] - rank),
        "frozen": frozen,
        "total": total,
        "dy_dI_direction_norm": float(np.linalg.norm(dy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", default="0.05")
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--random_count", type=int, default=1)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in p2c4._cases():
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_segment = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
        y_ref = np.asarray(y_segment, dtype=float).reshape((-1, 6))[0]
        funcs = _make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_ref,
            reg_mu=float(args.reg_mu),
            steps_per_flex=4,
        )
        residual_jit, task_jit, output_jit, Jy_jit, JI_jit, Ty_jit, TI_jit = funcs
        _ = residual_jit(jnp.asarray(y_ref), jnp.asarray(I0)).block_until_ready()
        _ = Jy_jit(jnp.asarray(y_ref), jnp.asarray(I0)).block_until_ready()
        base = _solve(y0=y_ref, I=I0, residual_jit=residual_jit, output_jit=output_jit, Jy_jit=Jy_jit, max_iter=args.max_iter)
        solve_rows.append(
            {
                "case": case.name,
                "success": base.success,
                "steady_s": base.steady_s,
                "residual_norm": base.residual_norm,
                "stationarity_norm": base.stationarity_norm,
                "boundary_residual_norm": base.boundary_norm,
            }
        )
        if not base.success:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "case": case.name,
                    "failure_class": "F1_baseline_unstable",
                    "evidence": {
                        "residual_norm": base.residual_norm,
                        "stationarity_norm": base.stationarity_norm,
                        "boundary": base.boundary_norm,
                    },
                    "recovery_action": "add continuation, low-order slack, or switch to hard multiple shooting",
                },
            )
            continue
        for dir_name, direction in _directions(I0, args.random_count):
            chain = _implicit_chain(
                y=base.y,
                I=I0,
                direction=direction,
                residual_jit=residual_jit,
                Jy_jit=Jy_jit,
                JI_jit=JI_jit,
                Ty_jit=Ty_jit,
                TI_jit=TI_jit,
            )
            for eps in eps_values:
                plus = _solve(y0=base.y, I=I0 + eps * direction, residual_jit=residual_jit, output_jit=output_jit, Jy_jit=Jy_jit, max_iter=args.max_iter)
                minus = _solve(y0=base.y, I=I0 - eps * direction, residual_jit=residual_jit, output_jit=output_jit, Jy_jit=Jy_jit, max_iter=args.max_iter)
                task_plus = np.asarray(jax.device_get(task_jit(jnp.asarray(plus.y), jnp.asarray(I0 + eps * direction))), dtype=float)
                task_minus = np.asarray(jax.device_get(task_jit(jnp.asarray(minus.y), jnp.asarray(I0 - eps * direction))), dtype=float)
                fd = (task_plus - task_minus) / (2.0 * eps)
                rel, cosine, fd_norm = p2c7._safe_rel(np.asarray(chain["total"], dtype=float), fd)
                branch_jump = bool(np.linalg.norm(plus.y - base.y) > 5.0 or np.linalg.norm(minus.y - base.y) > 5.0)
                status = "usable" if rel <= 0.3 and cosine > 0 and not branch_jump and plus.success and minus.success else "untrusted"
                failure_class = "PASS" if status == "usable" else ("F7_reopt_solver_instability" if not plus.success or not minus.success else "F5_sensitivity_mismatch")
                row = {
                    "case": case.name,
                    "direction": dir_name,
                    "eps": eps,
                    "status": status,
                    "failure_class": failure_class,
                    "rel_err_total": rel,
                    "cosine_total": cosine,
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
                    "frozen_norm": float(np.linalg.norm(chain["frozen"])),
                    "total_norm": float(np.linalg.norm(chain["total"])),
                    "fd_total_vector": fd.tolist(),
                    "implicit_total_vector": np.asarray(chain["total"], dtype=float).tolist(),
                }
                rows.append(row)
                if failure_class != "PASS":
                    _append_jsonl(failures, {"ts": time.time(), "case": case.name, "direction": dir_name, "eps": eps, "failure_class": failure_class, "evidence": {"rel_err": rel, "cosine": cosine}})
                _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "audit", "direction": dir_name, "eps": eps, "status": status, "rel_err": rel})

    usable = [row for row in rows if row["status"] == "usable"]
    classification = "usable" if rows and len(usable) == len(rows) else ("partially-usable" if usable else "untrusted")
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "eps_values": eps_values,
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "row_count": len(rows),
        "median_rel_err_total": float(np.median([row["rel_err_total"] for row in rows])) if rows else float("nan"),
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
