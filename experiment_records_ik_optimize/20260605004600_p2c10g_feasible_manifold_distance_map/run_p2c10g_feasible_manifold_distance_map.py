"""Map terminal-pose distance to the feasibility-first shooting branch."""

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
import run_p2c8_continuity_branch_sensitivity as p2c8

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import forward_segment_wrench, output_to_numpy

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

    def raw_continuity_y(y: jnp.ndarray) -> jnp.ndarray:
        return p2c8.interface_wrench_residual(params_for_I(jnp.asarray(I)), y, steps_per_flex=steps_per_flex)

    def boundary_y(y: jnp.ndarray) -> jnp.ndarray:
        return output_y(y).boundary

    def feasibility_residual_y(y: jnp.ndarray) -> jnp.ndarray:
        boundary = jnp.asarray(boundary_scale, dtype=jnp.float64) * boundary_y(y)
        continuity = jnp.asarray(continuity_scale, dtype=jnp.float64) * raw_continuity_y(y)
        reg = reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([boundary.reshape(-1), continuity.reshape(-1), reg.reshape(-1)], axis=0)

    return {
        "I": I,
        "y_ref": y_ref,
        "target_p": target_p,
        "target_q": target_q,
        "feasibility_residual_jit": jax.jit(feasibility_residual_y),
        "feasibility_jac_jit": jax.jit(jax.jacfwd(feasibility_residual_y, argnums=0)),
        "boundary_jit": jax.jit(boundary_y),
        "raw_continuity_jit": jax.jit(raw_continuity_y),
        "output_jit": jax.jit(output_y),
    }


def _solve_feasible(funcs: dict[str, Any], y0: np.ndarray, *, max_iter: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["feasibility_residual_jit"](jnp.asarray(yy, dtype=jnp.float64))), dtype=float),
        np.asarray(y0, dtype=float).reshape(-1),
        jac=lambda yy: np.asarray(jax.device_get(funcs["feasibility_jac_jit"](jnp.asarray(yy, dtype=jnp.float64))), dtype=float),
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
    r = np.asarray(jax.device_get(funcs["feasibility_residual_jit"](y_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["feasibility_jac_jit"](y_j)), dtype=float)
    boundary = np.asarray(jax.device_get(funcs["boundary_jit"](y_j)), dtype=float)
    raw_cont = np.asarray(jax.device_get(funcs["raw_continuity_jit"](y_j)), dtype=float)
    out = output_to_numpy(funcs["output_jit"](y_j))
    stationarity = J.T @ r
    target_p = np.asarray(funcs["target_p"], dtype=float).reshape(3)
    target_q = jnp.asarray(funcs["target_q"], dtype=jnp.float64).reshape(4)
    tip_p = np.asarray(out["tip_p"], dtype=float).reshape(3)
    tip_q_j = jnp.asarray(out["tip_q"], dtype=jnp.float64).reshape(4)
    so3_vec = np.asarray(jax.device_get(quaternion_so3_residual(tip_q_j, target_q, sigma_angle=1.0)), dtype=float)
    return {
        "y": y,
        "out": out,
        "nfev": int(opt.nfev),
        "success_flag": bool(opt.success),
        "steady_s": float(steady_s),
        "feasibility_residual_norm": float(np.linalg.norm(r)),
        "boundary_norm": float(np.linalg.norm(boundary)),
        "raw_continuity_norm": float(np.linalg.norm(raw_cont)),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "terminal_position_error_m": float(np.linalg.norm(tip_p - target_p)),
        "terminal_orientation_error_rad": float(np.linalg.norm(so3_vec)),
        "terminal_orientation_error_deg": float(np.linalg.norm(so3_vec) * 180.0 / np.pi),
        "tip_p": tip_p.tolist(),
        "target_p": target_p.tolist(),
        "tip_q": np.asarray(out["tip_q"], dtype=float).reshape(4).tolist(),
        "target_q": np.asarray(funcs["target_q"], dtype=float).reshape(4).tolist(),
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(alpha, [row["terminal_position_error_m"] for row in rows], marker="o", label="position error")
    axes[0].set_xlabel("continuation alpha")
    axes[0].set_ylabel("terminal position error (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(alpha, [row["terminal_orientation_error_deg"] for row in rows], marker="o", label="orientation error")
    axes[1].plot(alpha, [row["boundary_norm"] for row in rows], marker="o", label="boundary residual")
    axes[1].plot(alpha, [row["raw_continuity_norm"] for row in rows], marker="o", label="raw continuity residual")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("continuation alpha")
    axes[1].set_ylabel("metric value")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    fig.suptitle("P2c10g distance from target to feasible shooting branch")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feasible_manifold_distance.png", dpi=180)
    plt.close(fig)


def _plot_tip_path(rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    tip = np.asarray([row["tip_p"] for row in rows], dtype=float)
    target = np.asarray([row["target_p"] for row in rows], dtype=float)
    ax.plot(tip[:, 0], tip[:, 1], tip[:, 2], marker="o", label="feasible branch terminal")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], marker="^", label="target terminal")
    for row, p, q in zip(rows, tip, target):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#999999", alpha=0.5)
        ax.text(p[0], p[1], p[2], f"{row['alpha']:.2f}", fontsize=7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Terminal path: feasible branch vs target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "terminal_path_distance_3d.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alpha_start", type=float, default=0.20)
    parser.add_argument("--alpha_stop", type=float, default=0.25)
    parser.add_argument("--alpha_step", type=float, default=0.01)
    parser.add_argument("--continuity_scale", type=float, default=1000.0)
    parser.add_argument("--boundary_scale", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-6)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    n_steps = int(round((args.alpha_stop - args.alpha_start) / args.alpha_step))
    alphas = [round(args.alpha_start + i * args.alpha_step, 10) for i in range(n_steps + 1)]
    rows: list[dict[str, Any]] = []
    y0: np.ndarray | None = None
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
        if y0 is None:
            y0 = funcs["y_ref"]
        _ = funcs["feasibility_residual_jit"](jnp.asarray(y0)).block_until_ready()
        _ = funcs["feasibility_jac_jit"](jnp.asarray(y0)).block_until_ready()
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha})
        result = _solve_feasible(funcs, y0, max_iter=args.max_iter)
        y0 = result["y"]
        row = {
            "alpha": alpha,
            "success_flag": result["success_flag"],
            "nfev": result["nfev"],
            "steady_s": result["steady_s"],
            "feasibility_residual_norm": result["feasibility_residual_norm"],
            "boundary_norm": result["boundary_norm"],
            "raw_continuity_norm": result["raw_continuity_norm"],
            "stationarity_norm": result["stationarity_norm"],
            "terminal_position_error_m": result["terminal_position_error_m"],
            "terminal_orientation_error_rad": result["terminal_orientation_error_rad"],
            "terminal_orientation_error_deg": result["terminal_orientation_error_deg"],
            "tip_p": result["tip_p"],
            "target_p": result["target_p"],
            "tip_q": result["tip_q"],
            "target_q": result["target_q"],
            "diagnostic_class": "feasible-branch-distance",
        }
        rows.append(row)
        if row["stationarity_norm"] > 1.0e-3:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "alpha": alpha,
                    "failure_class": "物理可行分支驻值性不足",
                    "evidence": {
                        "stationarity_norm": row["stationarity_norm"],
                        "position_error_m": row["terminal_position_error_m"],
                        "orientation_error_deg": row["terminal_orientation_error_deg"],
                    },
                },
            )

    _plot(rows)
    _plot_tip_path(rows)
    position_errors = [row["terminal_position_error_m"] for row in rows]
    orientation_errors = [row["terminal_orientation_error_deg"] for row in rows]
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "continuity_scale": args.continuity_scale,
            "boundary_scale": args.boundary_scale,
            "reg_mu": args.reg_mu,
        },
        "classification": "feasible-manifold-distance-map",
        "max_terminal_position_error_m": float(max(position_errors)) if position_errors else float("nan"),
        "max_terminal_orientation_error_deg": float(max(orientation_errors)) if orientation_errors else float("nan"),
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
