"""Audit residual-block contributions at the alpha break region."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8

from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import forward_segment_wrench
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

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


def _build_bundle(case: Any, *, reg_mu: float, continuity_mu: float, steps_per_flex: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    weights = p2c8.WEIGHTS
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))
    cont_scale = jnp.sqrt(jnp.asarray(continuity_mu, dtype=jnp.float64))

    def params_for_I(I_j: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I_j, dtype=jnp.float64).reshape(8))

    def block_position(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I_j), y, steps_per_flex=steps_per_flex)
        return float(weights.tip_weight) * (out.tip_p - target_p_j)

    def block_orientation(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I_j), y, steps_per_flex=steps_per_flex)
        return float(weights.so3_weight) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)

    def block_boundary(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I_j), y, steps_per_flex=steps_per_flex)
        return float(weights.boundary_weight) * out.boundary

    def block_regularization(y: jnp.ndarray, _I_j: jnp.ndarray) -> jnp.ndarray:
        return reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)

    def block_continuity(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        return cont_scale * p2c8.interface_wrench_residual(params_for_I(I_j), y, steps_per_flex=steps_per_flex)

    block_funs = {
        "末端位置残差": block_position,
        "末端姿态残差": block_orientation,
        "末端边界残差": block_boundary,
        "分支保持残差": block_regularization,
        "段间连续残差": block_continuity,
    }

    def total_residual(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([fn(y, I_j).reshape(-1) for fn in block_funs.values()], axis=0)

    def task_fun(y: jnp.ndarray, I_j: jnp.ndarray):
        out = forward_segment_wrench(params_for_I(I_j), y, steps_per_flex=steps_per_flex)
        return jnp.concatenate([out.tip_p.reshape(-1), p2c7._so3_vec(out.tip_q, target_q_j).reshape(-1)], axis=0)

    residual_jit = jax.jit(total_residual)
    output_jit = jax.jit(lambda y, I_j: forward_segment_wrench(params_for_I(I_j), y, steps_per_flex=steps_per_flex))
    Jy_jit = jax.jit(jax.jacfwd(total_residual, argnums=0))
    task_jit = jax.jit(task_fun)
    block_jits = {name: jax.jit(fn) for name, fn in block_funs.items()}
    block_jac_jits = {name: jax.jit(jax.jacfwd(fn, argnums=0)) for name, fn in block_funs.items()}
    _ = residual_jit(jnp.asarray(y_ref), jnp.asarray(I)).block_until_ready()
    return {
        "I": I,
        "y_ref": y_ref,
        "residual_jit": residual_jit,
        "output_jit": output_jit,
        "Jy_jit": Jy_jit,
        "task_jit": task_jit,
        "block_jits": block_jits,
        "block_jac_jits": block_jac_jits,
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


def _audit_blocks(bundle: dict[str, Any], y: np.ndarray, alpha: float, branch: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(bundle["I"], dtype=jnp.float64)
    total_r = np.asarray(jax.device_get(bundle["residual_jit"](y_j, I_j)), dtype=float)
    total_J = np.asarray(jax.device_get(bundle["Jy_jit"](y_j, I_j)), dtype=float)
    total_grad = total_J.T @ total_r
    total_grad_norm = float(np.linalg.norm(total_grad))
    total_residual_norm = float(np.linalg.norm(total_r))
    svals = np.linalg.svd(total_J, compute_uv=False)
    total_condition = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    rows = []
    contributions = {}
    for block_name, block_jit in bundle["block_jits"].items():
        r = np.asarray(jax.device_get(block_jit(y_j, I_j)), dtype=float)
        J = np.asarray(jax.device_get(bundle["block_jac_jits"][block_name](y_j, I_j)), dtype=float)
        g = J.T @ r
        g_norm = float(np.linalg.norm(g))
        residual_norm = float(np.linalg.norm(r))
        cosine = float(np.dot(g, total_grad) / max(g_norm * total_grad_norm, 1.0e-30))
        rows.append(
            {
                "alpha": alpha,
                "branch": branch,
                "block": block_name,
                "residual_norm": residual_norm,
                "residual_share": residual_norm / max(total_residual_norm, 1.0e-30),
                "stationarity_contribution_norm": g_norm,
                "stationarity_contribution_share": g_norm / max(total_grad_norm, 1.0e-30),
                "cosine_with_total_stationarity": cosine,
            }
        )
        contributions[block_name] = g
    summary = {
        "alpha": alpha,
        "branch": branch,
        "total_residual_norm": total_residual_norm,
        "total_stationarity_norm": total_grad_norm,
        "total_jacobian_rank": int(np.linalg.matrix_rank(total_J)),
        "total_condition_number": total_condition,
        "contribution_closure_error": float(np.linalg.norm(sum(contributions.values()) - total_grad)),
    }
    return rows, summary


def _plot(block_rows: list[dict[str, Any]]) -> None:
    alphas = sorted({row["alpha"] for row in block_rows})
    blocks = list(dict.fromkeys(row["block"] for row in block_rows))
    label_map = {
        "末端位置残差": "terminal position",
        "末端姿态残差": "terminal orientation",
        "末端边界残差": "terminal boundary",
        "分支保持残差": "branch regularization",
        "段间连续残差": "interface continuity",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, alpha in enumerate(alphas):
        rows = [row for row in block_rows if row["alpha"] == alpha]
        x = np.arange(len(blocks)) + idx * 0.35
        axes[0].bar(x, [next(row for row in rows if row["block"] == b)["residual_norm"] for b in blocks], width=0.35, label=f"alpha={alpha:.2f}")
        axes[1].bar(x, [next(row for row in rows if row["block"] == b)["stationarity_contribution_norm"] for b in blocks], width=0.35, label=f"alpha={alpha:.2f}")
    for ax in axes:
        ax.set_xticks(np.arange(len(blocks)) + 0.175)
        ax.set_xticklabels([label_map.get(block, block) for block in blocks], rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_title("Residual block norms")
    axes[1].set_title("Stationarity contribution norms")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_cause_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.20,0.25")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    block_rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for alpha in alphas:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "alpha": alpha})
        case = _interpolated_case(base, target, alpha)
        bundle = _build_bundle(case, reg_mu=args.reg_mu, continuity_mu=args.continuity_mu, steps_per_flex=args.steps_per_flex)
        branches = {"冷启动参考分支": bundle["y_ref"], "竞争分支": 0.5 * bundle["y_ref"]}
        for branch, y0 in branches.items():
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "branch": branch})
            solve = _solve(bundle, y0, args.max_iter)
            row = {
                "alpha": alpha,
                "branch": branch,
                "success_flag": bool(solve.success),
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "steady_s": solve.steady_s,
            }
            solve_rows.append(row)
            if branch == "冷启动参考分支":
                b_rows, t_row = _audit_blocks(bundle, solve.y, alpha, branch)
                block_rows.extend(b_rows)
                total_rows.append(t_row)
                if t_row["total_stationarity_norm"] > 1.0e-3:
                    _append_jsonl(failures, {"ts": time.time(), "alpha": alpha, "failure_class": "一阶最优性退化", "evidence": t_row})

    _plot(block_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": "break-cause-audited",
        "block_rows": block_rows,
        "solve_rows": solve_rows,
        "total_rows": total_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "stationarity_block_summary.csv", block_rows)
    _write_csv(OUT_DIR / "solve_summary.csv", solve_rows)
    _write_csv(OUT_DIR / "total_summary.csv", total_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
