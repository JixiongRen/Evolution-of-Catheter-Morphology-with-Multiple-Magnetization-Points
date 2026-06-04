"""Run exact-IFT shadow replay for shooting task sensitivity."""

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
P2C9F_DIR = ROOT / "experiment_records_ik_optimize" / "20260604141002_p2c9f_exact_stationarity_ift_probe"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, P2C9F_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8
import run_p2c9f_exact_stationarity_ift_probe as p2c9f

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260604145803)


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


def _relative_error_and_cosine(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    pred_norm = float(np.linalg.norm(pred))
    ref_norm = float(np.linalg.norm(ref))
    err = float(np.linalg.norm(pred - ref))
    rel = err / max(ref_norm, 1.0e-12)
    cosine = float(np.dot(pred, ref) / max(pred_norm * ref_norm, 1.0e-30))
    return rel, cosine, ref_norm


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    directions = [(f"active_coil_{active}", e)]
    for i in range(random_count):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        directions.append((f"random_{i}", d))
    return directions


def _exact_sensitivity_matrix(
    *,
    y: np.ndarray,
    I: np.ndarray,
    grad_jit: Any,
    H_jit: Any,
    GI_jit: Any,
    Ty_jit: Any,
    TI_jit: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64)
    g = np.asarray(jax.device_get(grad_jit(y_j, I_j)), dtype=float)
    H = np.asarray(jax.device_get(H_jit(y_j, I_j)), dtype=float)
    GI = np.asarray(jax.device_get(GI_jit(y_j, I_j)), dtype=float)
    Ty = np.asarray(jax.device_get(Ty_jit(y_j, I_j)), dtype=float)
    TI = np.asarray(jax.device_get(TI_jit(y_j, I_j)), dtype=float)
    diag = np.clip(np.abs(np.diag(H)), 1.0e-12, None)
    H_reg = H + 1.0e-8 * np.diag(diag)
    try:
        dydI = -np.linalg.solve(H_reg, GI)
    except np.linalg.LinAlgError:
        dydI = -np.linalg.lstsq(H_reg, GI, rcond=None)[0]
    sensitivity = TI + Ty @ dydI
    svals = np.linalg.svd(H, compute_uv=False)
    diagnostics = {
        "stationarity_norm": float(np.linalg.norm(g)),
        "hessian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "hessian_rank": int(np.linalg.matrix_rank(H)),
    }
    return sensitivity, diagnostics


def _build_case_functions(case: Any, continuity_mu: float, reg_mu: float, max_iter: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, _continuity_jit, Jy_jit, _JI_jit, _Ty_gn_jit, _TI_gn_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    grad_jit, H_jit, GI_jit, Ty_jit, TI_jit = p2c9f._make_exact_functions(residual_fun=residual_jit, task_fun=task_jit)
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    _ = H_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    base = p2c7._solve_branch(
        y0=y_oracle,
        I=I0,
        residual_jit=residual_jit,
        output_jit=output_jit,
        Jy_jit=Jy_jit,
        max_iter=max_iter,
    )
    return {
        "I0": I0,
        "base": base,
        "residual_jit": residual_jit,
        "task_jit": task_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
        "grad_jit": grad_jit,
        "H_jit": H_jit,
        "GI_jit": GI_jit,
        "Ty_jit": Ty_jit,
        "TI_jit": TI_jit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n3_mixed_base_i4,n2_z_pair_rand12,n1_tip_z_rand15,n3_mixed_rand20")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--random_count", type=int, default=1)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    case_names = {x.strip() for x in args.cases.split(",") if x.strip()}
    rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case in [item for item in p2c4._cases() if item.name in case_names]:
        _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "baseline"})
        bundle = _build_case_functions(case, args.continuity_mu, args.reg_mu, args.max_iter)
        I0 = bundle["I0"]
        base = bundle["base"]
        solve_row = {
            "case": case.name,
            "success": base.success,
            "steady_s": base.steady_s,
            "residual_norm": base.residual_norm,
            "stationarity_norm": base.stationarity_norm,
            "boundary_residual_norm": base.boundary_norm,
        }
        solve_rows.append(solve_row)
        if not base.success:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "case": case.name,
                    "failure_class": "F1_baseline_unstable",
                    "evidence": solve_row,
                },
            )
            continue

        sensitivity, diag = _exact_sensitivity_matrix(
            y=base.y,
            I=I0,
            grad_jit=bundle["grad_jit"],
            H_jit=bundle["H_jit"],
            GI_jit=bundle["GI_jit"],
            Ty_jit=bundle["Ty_jit"],
            TI_jit=bundle["TI_jit"],
        )
        task0 = np.asarray(jax.device_get(bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(I0))), dtype=float)

        for direction_name, direction in _directions(I0, args.random_count):
            delta_I = float(args.step) * direction
            predicted = sensitivity @ delta_I
            perturbed = p2c7._solve_branch(
                y0=base.y,
                I=I0 + delta_I,
                residual_jit=bundle["residual_jit"],
                output_jit=bundle["output_jit"],
                Jy_jit=bundle["Jy_jit"],
                max_iter=args.max_iter,
            )
            task1 = np.asarray(
                jax.device_get(bundle["task_jit"](jnp.asarray(perturbed.y), jnp.asarray(I0 + delta_I))),
                dtype=float,
            )
            actual = task1 - task0
            rel_err, cosine, actual_norm = _relative_error_and_cosine(predicted, actual)
            branch_jump = bool(np.linalg.norm(perturbed.y - base.y) > 5.0)
            status = (
                "usable"
                if perturbed.success and rel_err <= 0.3 and cosine > 0.0 and base.boundary_norm <= 0.2 and not branch_jump
                else "diagnostic-only"
            )
            failure_class = "PASS" if status == "usable" else (
                "F7_reopt_solver_instability" if not perturbed.success else (
                    "F4_branch_jump" if branch_jump else "F5_shadow_replay_mismatch"
                )
            )
            row = {
                "case": case.name,
                "direction": direction_name,
                "step_A": float(args.step),
                "status": status,
                "failure_class": failure_class,
                "rel_err": rel_err,
                "cosine": cosine,
                "actual_norm": actual_norm,
                "predicted_norm": float(np.linalg.norm(predicted)),
                "base_residual_norm": base.residual_norm,
                "base_stationarity_norm": base.stationarity_norm,
                "base_boundary": base.boundary_norm,
                "perturbed_success": perturbed.success,
                "perturbed_residual_norm": perturbed.residual_norm,
                "perturbed_stationarity_norm": perturbed.stationarity_norm,
                "perturbed_boundary": perturbed.boundary_norm,
                "branch_jump": branch_jump,
                "branch_delta_norm": float(np.linalg.norm(perturbed.y - base.y)),
                "hessian_condition": diag.get("hessian_condition"),
                "hessian_rank": diag.get("hessian_rank"),
                "predicted_delta": predicted.tolist(),
                "actual_delta": actual.tolist(),
                "delta_I": delta_I.tolist(),
            }
            rows.append(row)
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case.name, "phase": "shadow", "direction": direction_name, "status": status, "rel_err": rel_err, "cosine": cosine})
            if failure_class != "PASS":
                _append_jsonl(failures, {"ts": time.time(), "case": case.name, "direction": direction_name, "failure_class": failure_class, "evidence": row})

    usable = [row for row in rows if row["status"] == "usable"]
    failed_baselines = [row for row in solve_rows if not row["success"]]
    classification = (
        "usable"
        if rows and len(usable) == len(rows) and not failed_baselines
        else ("partially-usable" if usable else "diagnostic-only")
    )
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "cases": sorted(case_names),
            "step_A": args.step,
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": classification,
        "usable_count": len(usable),
        "row_count": len(rows),
        "failed_baseline_count": len(failed_baselines),
        "median_rel_err": float(np.median([row["rel_err"] for row in rows])) if rows else float("nan"),
        "median_cosine": float(np.median([row["cosine"] for row in rows])) if rows else float("nan"),
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
