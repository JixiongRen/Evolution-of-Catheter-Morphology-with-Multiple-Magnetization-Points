"""Feasibility-first layered shooting audit near the P2c break."""

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
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import forward_segment_wrench, output_to_numpy

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
GATE = {
    "task_norm_max": 1.0e-2,
    "boundary_norm_max": 2.0e-1,
    "continuity_norm_max": 1.0e-3,
    "stationarity_norm_max": 1.0e-3,
}


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
    currents = (1.0 - alpha) * np.asarray(base.currents, dtype=float) + alpha * np.asarray(target.currents, dtype=float)
    return p2c4.CaseSpec(
        name=f"alpha_{alpha:.3f}",
        m_body_list=base.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _build_functions(case: Any, *, continuity_scale: float, boundary_scale: float, reg_mu: float, steps_per_flex: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))

    def params_for_I(I_j: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I_j, dtype=jnp.float64).reshape(8))

    def output_y(y: jnp.ndarray) -> Any:
        return forward_segment_wrench(params_for_I(jnp.asarray(I)), y, steps_per_flex=steps_per_flex)

    def task_residual_y(y: jnp.ndarray) -> jnp.ndarray:
        out = output_y(y)
        r_p = float(p2c8.WEIGHTS.tip_weight) * (out.tip_p - target_p_j)
        r_q = float(p2c8.WEIGHTS.so3_weight) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        return jnp.concatenate([r_p.reshape(-1), r_q.reshape(-1)], axis=0)

    def boundary_residual_y(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(boundary_scale, dtype=jnp.float64) * output_y(y).boundary

    def continuity_residual_y(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(continuity_scale, dtype=jnp.float64) * p2c8.interface_wrench_residual(
            params_for_I(jnp.asarray(I)),
            y,
            steps_per_flex=steps_per_flex,
        )

    def full_residual_y(y: jnp.ndarray, task_scale: jnp.ndarray) -> jnp.ndarray:
        task = jnp.asarray(task_scale, dtype=jnp.float64) * task_residual_y(y)
        boundary = boundary_residual_y(y)
        continuity = continuity_residual_y(y)
        reg = reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([task.reshape(-1), boundary.reshape(-1), continuity.reshape(-1), reg.reshape(-1)], axis=0)

    return {
        "I": I,
        "y_ref": y_ref,
        "target_p": target_p,
        "target_q": target_q,
        "full_residual_jit": jax.jit(full_residual_y),
        "full_jac_jit": jax.jit(jax.jacfwd(full_residual_y, argnums=0)),
        "task_jit": jax.jit(task_residual_y),
        "boundary_jit": jax.jit(boundary_residual_y),
        "continuity_jit": jax.jit(continuity_residual_y),
        "output_jit": jax.jit(output_y),
    }


def _solve_stage(funcs: dict[str, Any], y0: np.ndarray, *, task_scale: float, max_iter: int) -> dict[str, Any]:
    scale_j = jnp.asarray(float(task_scale), dtype=jnp.float64)
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["full_residual_jit"](jnp.asarray(yy, dtype=jnp.float64), scale_j)), dtype=float),
        np.asarray(y0, dtype=float).reshape(-1),
        jac=lambda yy: np.asarray(jax.device_get(funcs["full_jac_jit"](jnp.asarray(yy, dtype=jnp.float64), scale_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(np.asarray(y0, dtype=float).reshape(-1)),
        loss="linear",
        max_nfev=int(max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["full_residual_jit"](y_j, scale_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["full_jac_jit"](y_j, scale_j)), dtype=float)
    task = np.asarray(jax.device_get(funcs["task_jit"](y_j)), dtype=float)
    boundary = np.asarray(jax.device_get(funcs["boundary_jit"](y_j)), dtype=float)
    continuity = np.asarray(jax.device_get(funcs["continuity_jit"](y_j)), dtype=float)
    stationarity = J.T @ r
    out = output_to_numpy(funcs["output_jit"](y_j))
    return {
        "y": y,
        "out": out,
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "full_residual_norm": float(np.linalg.norm(r)),
        "task_norm": float(np.linalg.norm(task)),
        "boundary_norm": float(np.linalg.norm(boundary)),
        "continuity_norm": float(np.linalg.norm(continuity)),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "success_flag": bool(opt.success),
    }


def _gate(row: dict[str, Any]) -> bool:
    return bool(
        row["task_norm"] <= GATE["task_norm_max"]
        and row["boundary_norm"] <= GATE["boundary_norm_max"]
        and row["continuity_norm"] <= GATE["continuity_norm_max"]
        and row["stationarity_norm"] <= GATE["stationarity_norm_max"]
    )


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    metrics = [
        ("task_norm", "terminal task residual"),
        ("boundary_norm", "terminal boundary residual"),
        ("continuity_norm", "interface continuity residual"),
        ("stationarity_norm", "first-order optimality residual"),
    ]
    for ax, alpha in zip(axes, sorted({row["alpha"] for row in rows})):
        sub = sorted([row for row in rows if row["alpha"] == alpha], key=lambda item: item["task_scale"])
        xs = [row["task_scale"] for row in sub]
        for key, label in metrics:
            ax.plot(xs, [row[key] for row in sub], marker="o", label=label)
        ax.axhline(GATE["stationarity_norm_max"], color="#c5221f", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("terminal task scale")
        ax.set_title(f"alpha={alpha:.2f}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("metric value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("P2c10f feasibility-first layered shooting")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feasibility_first_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.20,0.25")
    parser.add_argument("--task_scales", default="0,0.1,0.3,1.0")
    parser.add_argument("--continuity_scale", type=float, default=1000.0)
    parser.add_argument("--boundary_scale", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-6)
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    task_scales = [float(x) for x in args.task_scales.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for alpha in alphas:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": alpha})
        funcs = _build_functions(
            _interpolated_case(base, target, alpha),
            continuity_scale=args.continuity_scale,
            boundary_scale=args.boundary_scale,
            reg_mu=args.reg_mu,
            steps_per_flex=args.steps_per_flex,
        )
        y0 = funcs["y_ref"]
        _ = funcs["full_residual_jit"](jnp.asarray(y0), jnp.asarray(0.0)).block_until_ready()
        _ = funcs["full_jac_jit"](jnp.asarray(y0), jnp.asarray(0.0)).block_until_ready()
        for task_scale in task_scales:
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "task_scale": task_scale})
            try:
                result = _solve_stage(funcs, y0, task_scale=task_scale, max_iter=args.max_iter)
                y0 = result["y"]
                row = {
                    "alpha": alpha,
                    "task_scale": task_scale,
                    "success_flag": result["success_flag"],
                    "nfev": result["nfev"],
                    "steady_s": result["steady_s"],
                    "full_residual_norm": result["full_residual_norm"],
                    "task_norm": result["task_norm"],
                    "boundary_norm": result["boundary_norm"],
                    "continuity_norm": result["continuity_norm"],
                    "stationarity_norm": result["stationarity_norm"],
                    "tip_p": result["out"]["tip_p"].tolist(),
                    "tip_q": result["out"]["tip_q"].tolist(),
                }
            except Exception as exc:
                row = {
                    "alpha": alpha,
                    "task_scale": task_scale,
                    "success_flag": False,
                    "nfev": 0,
                    "steady_s": 0.0,
                    "full_residual_norm": float("inf"),
                    "task_norm": float("inf"),
                    "boundary_norm": float("inf"),
                    "continuity_norm": float("inf"),
                    "stationarity_norm": float("inf"),
                    "error": repr(exc),
                }
                _append_jsonl(failures, {"ts": time.time(), "failure_class": "分层打靶求解异常", "evidence": row})
            row["publishable"] = _gate(row)
            rows.append(row)
            if not row["publishable"]:
                _append_jsonl(failures, {"ts": time.time(), "failure_class": "分层打靶未通过发布门禁", "evidence": row})

    _plot(rows)
    best_by_alpha = []
    for alpha in alphas:
        sub = [row for row in rows if row["alpha"] == alpha]
        feasible = [row for row in sub if row["boundary_norm"] <= GATE["boundary_norm_max"] and row["continuity_norm"] <= GATE["continuity_norm_max"]]
        best_task = min(sub, key=lambda item: item["task_norm"])
        best_physics = min(sub, key=lambda item: item["boundary_norm"] + item["continuity_norm"])
        best_by_alpha.append(
            {
                "alpha": alpha,
                "any_publishable": any(row["publishable"] for row in sub),
                "any_physics_feasible": bool(feasible),
                "best_task_scale": best_task["task_scale"],
                "best_task_norm": best_task["task_norm"],
                "best_physics_task_scale": best_physics["task_scale"],
                "best_physics_boundary_norm": best_physics["boundary_norm"],
                "best_physics_continuity_norm": best_physics["continuity_norm"],
                "best_physics_stationarity_norm": best_physics["stationarity_norm"],
            }
        )

    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "gate": GATE,
        },
        "classification": "feasibility-first-layered-audited",
        "best_by_alpha": best_by_alpha,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "alpha_best_summary.csv", best_by_alpha)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
