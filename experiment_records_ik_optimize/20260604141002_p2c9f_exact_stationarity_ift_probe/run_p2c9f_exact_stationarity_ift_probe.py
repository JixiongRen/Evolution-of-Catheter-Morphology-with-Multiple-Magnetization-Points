"""Audit exact stationarity IFT sensitivity for soft-continuity shooting."""

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
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    out = [(f"active_coil_{active}", e)]
    for i in range(random_count):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        out.append((f"random_{i}", d))
    return out


def _make_exact_functions(*, residual_fun: Any, task_fun: Any):
    def objective_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        r = residual_fun(y, I)
        return 0.5 * jnp.dot(r, r)

    grad_fun = jax.grad(objective_yI, argnums=0)
    hess_y_fun = jax.jacfwd(grad_fun, argnums=0)
    grad_I_fun = jax.jacfwd(grad_fun, argnums=1)
    task_y_fun = jax.jacfwd(task_fun, argnums=0)
    task_I_fun = jax.jacfwd(task_fun, argnums=1)
    return (
        jax.jit(grad_fun),
        jax.jit(hess_y_fun),
        jax.jit(grad_I_fun),
        jax.jit(task_y_fun),
        jax.jit(task_I_fun),
    )


def _exact_ift_chain(
    *,
    y: np.ndarray,
    I: np.ndarray,
    direction: np.ndarray,
    grad_jit: Any,
    H_jit: Any,
    GI_jit: Any,
    Ty_jit: Any,
    TI_jit: Any,
) -> dict[str, Any]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64)
    d = np.asarray(direction, dtype=float)
    g = np.asarray(jax.device_get(grad_jit(y_j, I_j)), dtype=float)
    H = np.asarray(jax.device_get(H_jit(y_j, I_j)), dtype=float)
    GI = np.asarray(jax.device_get(GI_jit(y_j, I_j)), dtype=float)
    Ty = np.asarray(jax.device_get(Ty_jit(y_j, I_j)), dtype=float)
    TI = np.asarray(jax.device_get(TI_jit(y_j, I_j)), dtype=float)
    rhs = GI @ d
    diag = np.clip(np.abs(np.diag(H)), 1.0e-12, None)
    try:
        dy = -np.linalg.solve(H + 1.0e-8 * np.diag(diag), rhs)
    except np.linalg.LinAlgError:
        dy = -np.linalg.lstsq(H + 1.0e-8 * np.diag(diag), rhs, rcond=None)[0]
    total = TI @ d + Ty @ dy
    svals = np.linalg.svd(H, compute_uv=False)
    return {
        "stationarity_norm": float(np.linalg.norm(g)),
        "hessian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "hessian_rank": int(np.linalg.matrix_rank(H)),
        "dy_dI_direction_norm": float(np.linalg.norm(dy)),
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n3_mixed_base_i4,n2_z_pair_rand12,n1_tip_z_rand15,n3_mixed_rand20")
    parser.add_argument("--eps", default="0.02,0.05,0.10")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--random_count", type=int, default=1)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    case_names = {x.strip() for x in args.cases.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in [item for item in p2c4._cases() if item.name in case_names]:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "oracle"})
        engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
        I0 = np.asarray(case.currents, dtype=float)
        oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
        target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
        params_base = oracle["params"]
        y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
        residual_jit, task_jit, output_jit, _continuity_jit, Jy_jit, JI_jit, Ty_gn_jit, TI_gn_jit = p2c8._make_functions(
            params_base=params_base,
            target_p=target_p,
            target_q=target_q,
            y_ref=y_oracle,
            reg_mu=float(args.reg_mu),
            continuity_mu=float(args.continuity_mu),
            steps_per_flex=4,
        )
        grad_jit, H_jit, GI_jit, Ty_jit, TI_jit = _make_exact_functions(residual_fun=residual_jit, task_fun=task_jit)
        _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        _ = H_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
        base = p2c7._solve_branch(
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
                "success": base.success,
                "steady_s": base.steady_s,
                "residual_norm": base.residual_norm,
                "stationarity_norm": base.stationarity_norm,
                "boundary_residual_norm": base.boundary_norm,
            }
        )
        if not base.success:
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "failure_class": "F1_baseline_unstable", "evidence": solve_rows[-1]})
            continue
        for dir_name, direction in _directions(I0, int(args.random_count)):
            exact = _exact_ift_chain(
                y=base.y,
                I=I0,
                direction=direction,
                grad_jit=grad_jit,
                H_jit=H_jit,
                GI_jit=GI_jit,
                Ty_jit=Ty_jit,
                TI_jit=TI_jit,
            )
            gn = p2c7._implicit_chain(
                y=base.y,
                I=I0,
                direction=direction,
                residual_jit=residual_jit,
                task_jit=task_jit,
                Jy_jit=Jy_jit,
                JI_jit=JI_jit,
                Ty_jit=Ty_gn_jit,
                TI_jit=TI_gn_jit,
            )
            fd_vectors = []
            idxs = []
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
                fd = (task_plus - task_minus) / (2.0 * eps)
                fd_vectors.append(fd)
                rel_exact, cos_exact, fd_norm = p2c7._safe_rel(np.asarray(exact["total"], dtype=float), fd)
                rel_gn, cos_gn, _ = p2c7._safe_rel(np.asarray(gn["total"], dtype=float), fd)
                branch_jump = bool(np.linalg.norm(plus.y - base.y) > 5.0 or np.linalg.norm(minus.y - base.y) > 5.0)
                exact_status = "usable" if rel_exact <= 0.3 and cos_exact > 0 and not branch_jump and plus.success and minus.success else "untrusted"
                row = {
                    "case": case.name,
                    "direction": dir_name,
                    "eps": eps,
                    "status": exact_status,
                    "failure_class": "PASS" if exact_status == "usable" else ("F7_reopt_solver_instability" if not plus.success or not minus.success else "F5_sensitivity_mismatch"),
                    "rel_err_exact": rel_exact,
                    "cosine_exact": cos_exact,
                    "rel_err_gn": rel_gn,
                    "cosine_gn": cos_gn,
                    "fd_norm": fd_norm,
                    "stationarity_norm": exact["stationarity_norm"],
                    "hessian_condition": exact["hessian_condition"],
                    "hessian_rank": exact["hessian_rank"],
                    "gn_condition": gn["condition_number"],
                    "gn_rank": gn["rank"],
                    "base_boundary": base.boundary_norm,
                    "plus_success": plus.success,
                    "minus_success": minus.success,
                    "branch_jump": branch_jump,
                    "exact_total_norm": float(np.linalg.norm(exact["total"])),
                    "gn_total_norm": float(np.linalg.norm(gn["total"])),
                    "fd_total_vector": fd.tolist(),
                    "exact_total_vector": np.asarray(exact["total"], dtype=float).tolist(),
                    "gn_total_vector": np.asarray(gn["total"], dtype=float).tolist(),
                }
                rows.append(row)
                idxs.append(len(rows) - 1)
                if row["failure_class"] != "PASS":
                    _append_jsonl(failures, {"ts": time.time(), "case": case.name, "direction": dir_name, "eps": eps, "failure_class": row["failure_class"], "evidence": {"rel_exact": rel_exact, "rel_gn": rel_gn, "cos_exact": cos_exact}})
                _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "audit", "direction": dir_name, "eps": eps, "status": exact_status, "rel_exact": rel_exact, "rel_gn": rel_gn})
            norms = [float(np.linalg.norm(v)) for v in fd_vectors]
            norm_spread = (max(norms) - min(norms)) / max(float(np.mean(norms)), 1.0e-12)
            items = [rows[idx] for idx in idxs]
            direction_rows.append(
                {
                    "case": case.name,
                    "direction": dir_name,
                    "usable_eps_count": sum(1 for item in items if item["status"] == "usable"),
                    "row_count": len(items),
                    "median_rel_err_exact": float(np.median([item["rel_err_exact"] for item in items])),
                    "median_rel_err_gn": float(np.median([item["rel_err_gn"] for item in items])),
                    "median_cosine_exact": float(np.median([item["cosine_exact"] for item in items])),
                    "median_cosine_gn": float(np.median([item["cosine_gn"] for item in items])),
                    "eps_norm_rel_spread": norm_spread,
                    "direction_usable": bool(all(item["status"] == "usable" for item in items)),
                }
            )

    usable = [row for row in rows if row["status"] == "usable"]
    failed_baselines = [row for row in solve_rows if not row["success"]]
    classification = (
        "usable"
        if rows and len(usable) == len(rows) and not failed_baselines
        else ("partially-usable" if usable else "untrusted")
    )
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
        "failed_baseline_count": len(failed_baselines),
        "row_count": len(rows),
        "direction_usable_count": sum(1 for row in direction_rows if row["direction_usable"]),
        "direction_count": len(direction_rows),
        "median_rel_err_exact": float(np.median([row["rel_err_exact"] for row in rows])) if rows else float("nan"),
        "median_rel_err_gn": float(np.median([row["rel_err_gn"] for row in rows])) if rows else float("nan"),
        "rows": rows,
        "direction_rows": direction_rows,
        "solve_rows": solve_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "direction_summary.csv", direction_rows)
    _write_csv(OUT_DIR / "solve_summary.csv", solve_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
