"""Trace shooting branch along a current and target continuation path."""

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


def _case_by_name(name: str) -> Any:
    for case in p2c4._cases():
        if case.name == name:
            return case
    raise ValueError(f"Unknown case: {name}")


def _interpolated_case(base: Any, target: Any, alpha: float) -> Any:
    I_base = np.asarray(base.currents, dtype=float)
    I_target = np.asarray(target.currents, dtype=float)
    currents = tuple((1.0 - alpha) * I_base + alpha * I_target)
    return p2c4.CaseSpec(
        name=f"cont_alpha_{alpha:.3f}",
        m_body_list=base.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _build_functions(case: Any, *, continuity_mu: float, reg_mu: float) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, _continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=reg_mu,
        continuity_mu=continuity_mu,
        steps_per_flex=4,
    )
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I)).block_until_ready()
    return {
        "I": I,
        "y_oracle": y_oracle,
        "residual_jit": residual_jit,
        "task_jit": task_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
    }


def _solve(bundle: dict[str, Any], y0: np.ndarray, max_iter: int) -> Any:
    return p2c7._solve_branch(
        y0=y0,
        I=bundle["I"],
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    prev_y: np.ndarray | None = None
    first_failure_alpha: float | None = None
    t_all = time.perf_counter()

    for alpha in alphas:
        case = _interpolated_case(base, target, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": alpha})
        bundle = _build_functions(case, continuity_mu=args.continuity_mu, reg_mu=args.reg_mu)
        init_map = {"oracle": bundle["y_oracle"]}
        if prev_y is not None and prev_y.shape == bundle["y_oracle"].shape:
            init_map["continuation"] = prev_y
        init_map["half_oracle"] = 0.5 * bundle["y_oracle"]

        alpha_best = None
        for init_name, y0 in init_map.items():
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "init": init_name})
            solve = _solve(bundle, y0, args.max_iter)
            row = {
                "alpha": alpha,
                "init": init_name,
                "success": solve.success,
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "steady_s": solve.steady_s,
                "state_norm": float(np.linalg.norm(solve.y)),
            }
            rows.append(row)
            if alpha_best is None or (solve.success and not alpha_best.success) or solve.residual_norm < alpha_best.residual_norm:
                alpha_best = solve
        if alpha_best is not None and alpha_best.success:
            prev_y = alpha_best.y
        elif first_failure_alpha is None:
            first_failure_alpha = alpha
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "failure_class": "F1_continuation_break",
                    "alpha": alpha,
                    "evidence": [row for row in rows if row["alpha"] == alpha],
                },
            )

    per_alpha = []
    for alpha in alphas:
        items = [row for row in rows if row["alpha"] == alpha]
        best = min(items, key=lambda row: row["residual_norm"])
        per_alpha.append(
            {
                "alpha": alpha,
                "any_success": any(row["success"] for row in items),
                "best_init": best["init"],
                "best_residual_norm": best["residual_norm"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_boundary_residual_norm": best["boundary_residual_norm"],
            }
        )

    for item in per_alpha:
        item["any_linearizable"] = bool(
            item["any_success"]
            and item["best_residual_norm"] <= 1.0e-2
            and item["best_boundary_residual_norm"] <= 0.2
            and item["best_stationarity_norm"] <= 1.0e-3
        )

    recovered_target = bool(per_alpha and per_alpha[-1]["alpha"] == 1.0 and per_alpha[-1]["any_linearizable"])
    first_linearization_failure_alpha = next((item["alpha"] for item in per_alpha if not item["any_linearizable"]), None)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": "target-recovered" if recovered_target else "linearization-break",
        "first_failure_alpha": first_failure_alpha,
        "first_linearization_failure_alpha": first_linearization_failure_alpha,
        "recovered_target": recovered_target,
        "per_alpha": per_alpha,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "alpha_summary.csv", per_alpha)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
