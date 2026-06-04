"""Probe hard-continuity KKT sensitivity for shooting IK."""

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
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_terminal_shooting_jax import CleanShootingWeights, clean_terminal_residual, forward_segment_wrench, output_to_numpy

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
WEIGHTS = CleanShootingWeights(tip_weight=120.0, so3_weight=2.0, boundary_weight=0.5)
RNG = np.random.default_rng(20260604)


@dataclass
class KktSolve:
    """Hard-continuity KKT solve result."""

    v: np.ndarray
    y: np.ndarray
    lam: np.ndarray
    out: dict[str, np.ndarray]
    success: bool
    n_iter: int
    steady_s: float
    kkt_norm: float
    objective_norm: float
    constraint_norm: float
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


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    y_ref: np.ndarray,
    anchor_mu: float,
    constraint_scale: float,
    steps_per_flex: int,
):
    n_y = int(np.asarray(y_ref).size)
    n_c = n_y - 6
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(n_y)
    anchor_scale = jnp.sqrt(jnp.asarray(anchor_mu, dtype=jnp.float64))
    c_scale = jnp.asarray(constraint_scale, dtype=jnp.float64)

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def clean_residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        clean = clean_terminal_residual(
            params_for_I(I),
            y,
            target_p=target_p_j,
            target_q=target_q_j,
            weights=WEIGHTS,
            steps_per_flex=steps_per_flex,
            so3_scale=1.0,
        )
        anchor = anchor_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(n_y) - y_ref_j)
        return jnp.concatenate([clean.reshape(-1), anchor.reshape(-1)], axis=0)

    def constraint_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return c_scale * p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=steps_per_flex)

    def objective(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        r = clean_residual_yI(y, I)
        return 0.5 * jnp.dot(r, r)

    grad_obj = jax.grad(objective, argnums=0)
    jac_c_y = jax.jacfwd(constraint_yI, argnums=0)

    def kkt_residual_vI(v: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64).reshape(n_y + n_c)
        y = v[:n_y]
        lam = v[n_y:]
        c = constraint_yI(y, I)
        stationarity = grad_obj(y, I) + jac_c_y(y, I).T @ lam
        return jnp.concatenate([stationarity.reshape(-1), c.reshape(-1)], axis=0)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        return p2c7._task_vector(params_for_I(I), y, target_q_j, steps_per_flex=steps_per_flex)

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)

    return {
        "n_y": n_y,
        "n_c": n_c,
        "residual_jit": jax.jit(clean_residual_yI),
        "constraint_jit": jax.jit(constraint_yI),
        "kkt_jit": jax.jit(kkt_residual_vI),
        "kkt_v_jit": jax.jit(jax.jacfwd(kkt_residual_vI, argnums=0)),
        "kkt_I_jit": jax.jit(jax.jacfwd(kkt_residual_vI, argnums=1)),
        "task_jit": jax.jit(task_yI),
        "task_y_jit": jax.jit(jax.jacfwd(task_yI, argnums=0)),
        "task_I_jit": jax.jit(jax.jacfwd(task_yI, argnums=1)),
        "output_jit": jax.jit(output_yI),
    }


def _solve_kkt(*, v0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], max_iter: int) -> KktSolve:
    n_y = int(funcs["n_y"])
    n_c = int(funcs["n_c"])
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    t0 = time.perf_counter()
    opt = least_squares(
        lambda vv: np.asarray(jax.device_get(funcs["kkt_jit"](jnp.asarray(vv, dtype=jnp.float64), I_j)), dtype=float),
        np.asarray(v0, dtype=float).reshape(n_y + n_c),
        jac=lambda vv: np.asarray(jax.device_get(funcs["kkt_v_jit"](jnp.asarray(vv, dtype=jnp.float64), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones(n_y + n_c),
        loss="linear",
        max_nfev=int(max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    v = np.asarray(opt.x, dtype=float)
    y = v[:n_y]
    lam = v[n_y:]
    v_j = jnp.asarray(v, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    kkt = np.asarray(jax.device_get(funcs["kkt_jit"](v_j, I_j)), dtype=float)
    r = np.asarray(jax.device_get(funcs["residual_jit"](y_j, I_j)), dtype=float)
    c = np.asarray(jax.device_get(funcs["constraint_jit"](y_j, I_j)), dtype=float)
    out = output_to_numpy(funcs["output_jit"](y_j, I_j))
    stationarity = kkt[:n_y]
    boundary = float(np.linalg.norm(out["boundary"]))
    success = bool(
        np.all(np.isfinite(v))
        and np.linalg.norm(kkt) <= 1.0e-4
        and np.linalg.norm(c) <= 1.0e-5
        and np.linalg.norm(r) <= 1.0e-2
        and boundary <= 0.2
    )
    return KktSolve(
        v=v,
        y=y,
        lam=lam,
        out=out,
        success=success,
        n_iter=int(opt.nfev),
        steady_s=float(steady_s),
        kkt_norm=float(np.linalg.norm(kkt)),
        objective_norm=float(np.linalg.norm(r)),
        constraint_norm=float(np.linalg.norm(c)),
        stationarity_norm=float(np.linalg.norm(stationarity)),
        boundary_norm=boundary,
    )


def _directions(I0: np.ndarray) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    d = RNG.normal(size=8)
    d = d / max(float(np.linalg.norm(d)), 1.0e-12)
    return [(f"active_coil_{active}", e), ("random_0", d)]


def _ift_chain(*, solve: KktSolve, I: np.ndarray, direction: np.ndarray, funcs: dict[str, Any]) -> dict[str, Any]:
    n_y = int(funcs["n_y"])
    v_j = jnp.asarray(solve.v, dtype=jnp.float64)
    y_j = jnp.asarray(solve.y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64)
    d = np.asarray(direction, dtype=float)
    F_v = np.asarray(jax.device_get(funcs["kkt_v_jit"](v_j, I_j)), dtype=float)
    F_I = np.asarray(jax.device_get(funcs["kkt_I_jit"](v_j, I_j)), dtype=float)
    rhs = F_I @ d
    try:
        dv = -np.linalg.solve(F_v, rhs)
    except np.linalg.LinAlgError:
        dv = -np.linalg.lstsq(F_v, rhs, rcond=None)[0]
    dy = dv[:n_y]
    T_y = np.asarray(jax.device_get(funcs["task_y_jit"](y_j, I_j)), dtype=float)
    T_I = np.asarray(jax.device_get(funcs["task_I_jit"](y_j, I_j)), dtype=float)
    total = T_y @ dy + T_I @ d
    svals = np.linalg.svd(F_v, compute_uv=False)
    return {
        "total": total,
        "dy_dI_direction_norm": float(np.linalg.norm(dy)),
        "kkt_matrix_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "kkt_matrix_rank": int(np.linalg.matrix_rank(F_v)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n3_mixed_base_i4,n3_mixed_rand20")
    parser.add_argument("--eps", default="0.02,0.05,0.10")
    parser.add_argument("--anchor_mu", type=float, default=1.0e-6)
    parser.add_argument("--constraint_scale", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=80)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    case_names = {x.strip() for x in args.cases.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in [item for item in p2c4._cases() if item.name in case_names]:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
        funcs = _make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_ref,
            anchor_mu=float(args.anchor_mu),
            constraint_scale=float(args.constraint_scale),
            steps_per_flex=4,
        )
        v0 = np.concatenate([np.asarray(y_ref, dtype=float).reshape(-1), np.zeros(int(funcs["n_c"]), dtype=float)])
        _ = funcs["kkt_jit"](jnp.asarray(v0), jnp.asarray(I0)).block_until_ready()
        _ = funcs["kkt_v_jit"](jnp.asarray(v0), jnp.asarray(I0)).block_until_ready()
        base = _solve_kkt(v0=v0, I=I0, funcs=funcs, max_iter=int(args.max_iter))
        baseline_row = {
            "case": case.name,
            "success": base.success,
            "steady_s": base.steady_s,
            "kkt_norm": base.kkt_norm,
            "objective_norm": base.objective_norm,
            "constraint_norm": base.constraint_norm,
            "stationarity_norm": base.stationarity_norm,
            "boundary_residual_norm": base.boundary_norm,
        }
        baseline_rows.append(baseline_row)
        if not base.success:
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "failure_class": "F1_baseline_unstable", "evidence": baseline_row})
            continue
        for dir_name, direction in _directions(I0):
            chain = _ift_chain(solve=base, I=I0, direction=direction, funcs=funcs)
            for eps in eps_values:
                plus = _solve_kkt(v0=base.v, I=I0 + eps * direction, funcs=funcs, max_iter=int(args.max_iter))
                minus = _solve_kkt(v0=base.v, I=I0 - eps * direction, funcs=funcs, max_iter=int(args.max_iter))
                task_plus = np.asarray(jax.device_get(funcs["task_jit"](jnp.asarray(plus.y), jnp.asarray(I0 + eps * direction))), dtype=float)
                task_minus = np.asarray(jax.device_get(funcs["task_jit"](jnp.asarray(minus.y), jnp.asarray(I0 - eps * direction))), dtype=float)
                fd = (task_plus - task_minus) / (2.0 * eps)
                rel, cosine, fd_norm = p2c7._safe_rel(np.asarray(chain["total"], dtype=float), fd)
                branch_jump = bool(np.linalg.norm(plus.y - base.y) > 5.0 or np.linalg.norm(minus.y - base.y) > 5.0)
                status = "usable" if rel <= 0.3 and cosine > 0.0 and not branch_jump and plus.success and minus.success else "untrusted"
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
                    "kkt_matrix_condition": chain["kkt_matrix_condition"],
                    "kkt_matrix_rank": chain["kkt_matrix_rank"],
                    "base_boundary": base.boundary_norm,
                    "base_constraint": base.constraint_norm,
                    "plus_success": plus.success,
                    "minus_success": minus.success,
                    "branch_jump": branch_jump,
                    "total_norm": float(np.linalg.norm(chain["total"])),
                    "dy_dI_direction_norm": chain["dy_dI_direction_norm"],
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
            "cases": sorted(case_names),
            "eps_values": eps_values,
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "row_count": len(rows),
        "median_rel_err_total": float(np.median([row["rel_err_total"] for row in rows])) if rows else float("nan"),
        "baseline_rows": baseline_rows,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "baseline_summary.csv", baseline_rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
