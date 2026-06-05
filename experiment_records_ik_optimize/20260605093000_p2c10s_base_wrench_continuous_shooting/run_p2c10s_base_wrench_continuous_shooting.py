"""Probe continuous base-wrench shooting for branch reduction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
for path in (ROOT, FK_OPT_DIR, P2C7_DIR, P2C10I_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import (
    CleanShootingOutput,
    _advance_rigid,
    _integrate_flex,
    initial_state,
    output_to_numpy,
)

import run_p2c10i_boundary_candidate_strict_gate as p2c10i

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


def _parse_candidates(text: str) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def forward_base_wrench(params: Any, y: jnp.ndarray, *, steps_per_flex: int = 4) -> CleanShootingOutput:
    """Integrate all segments from one base wrench without interface resets."""
    y = jnp.asarray(y, dtype=jnp.float64).reshape(6)
    x = initial_state(params, y)
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
    position_scale: float,
    so3_scale: float,
    boundary_scale: float,
    steps_per_flex: int,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> CleanShootingOutput:
        return forward_base_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        q_res = quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        return jnp.concatenate([out.tip_p.reshape(-1), q_res.reshape(-1)], axis=0)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        r_p = float(position_scale) * (out.tip_p - target_p_j)
        r_q = float(so3_scale) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        r_b = float(boundary_scale) * out.boundary
        return jnp.concatenate([r_p.reshape(-1), r_q.reshape(-1), r_b.reshape(-1)], axis=0)

    return {
        "residual_jit": jax.jit(residual_yI),
        "task_jit": jax.jit(task_yI),
        "output_jit": jax.jit(output_yI),
        "Jy_jit": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
    }


def _solve_base_wrench(
    *,
    y0: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
    max_iter: int,
) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y0 = np.asarray(y0, dtype=float).reshape(6)
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(yy), I_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(yy), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=int(max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual_jit"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["Jy_jit"](y_j, I_j)), dtype=float)
    out = output_to_numpy(funcs["output_jit"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    condition = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf")
    task = np.asarray(jax.device_get(funcs["task_jit"](y_j, I_j)), dtype=float)
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "jacobian_condition": condition,
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "task": task,
        "message": str(opt.message),
    }


def _build_case(case_name: str, alpha: float) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    target_case = p2c10i._case_by_name(case_name)
    case = p2c10i._scaled_case(target_case, alpha)
    engine = p2c10i.p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c10i.p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c10i.p2c2._oracle_terminal_pose(oracle, engine)
    y_ref = p2c10i.p2c2.p2c1._y_guess_from_oracle(oracle["z"], oracle["params"], "segment_wrench")
    return (
        oracle["params"],
        I0,
        np.asarray(target_p, dtype=float),
        np.asarray(target_q, dtype=float),
        np.asarray(y_ref, dtype=float),
        oracle,
    )


def _plot_branches(case_name: str, alpha: float, oracle: dict[str, Any], solves: dict[str, dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.0, label="FK参照")
    colors = {"oracle": "#d62728", "half_oracle": "#1f77b4", "zero": "#2ca02c"}
    for name, solve in solves.items():
        cl = np.asarray(solve["out"]["centerline"], dtype=float)
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], "--", lw=2.0, color=colors.get(name), label=f"{name}初值")
        tip = np.asarray(solve["out"]["tip_p"], dtype=float)
        ax.scatter([tip[0]], [tip[1]], [tip[2]], s=35, color=colors.get(name))
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=45, label="基端")
    ax.set_title(f"{case_name}, alpha={alpha:.2f}: 连续基端载荷打靶")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_branches.png", dpi=180)
    plt.close(fig)


def _plot_metrics(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['case']}@{r['alpha']:.2f}\n{r['init_name']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(max(10, 0.9 * len(rows)), 11), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "terminal_position_error", "末端位置误差(m)"),
        (axes[1], "terminal_so3_error_deg", "末端姿态误差(deg)"),
        (axes[2], "boundary_norm", "边界残差"),
        (axes[3], "jacobian_condition", "局部条件数"),
    ]:
        ax.bar(x, [r[key] for r in rows])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[2].set_yscale("log")
    axes[3].set_yscale("log")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "base_wrench_metrics.png", dpi=180)
    plt.close(fig)

    labels = [f"{r['case']}@{r['alpha']:.2f}\n{r['branch_a']} vs {r['branch_b']}" for r in pair_rows]
    x = np.arange(len(pair_rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.9 * len(pair_rows)), 8), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "tip_distance", "末端分支距离(m)"),
        (axes[1], "centerline_rms", "中心线均方根差(m)"),
        (axes[2], "state_distance", "基端载荷差"),
    ]:
        ax.bar(x, [r[key] for r in pair_rows])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[2].set_yscale("log")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "base_wrench_pairwise.png", dpi=180)
    plt.close(fig)


def _run_case(
    *,
    case_name: str,
    alpha: float,
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build_case", "case": case_name, "alpha": alpha})
    params, I0, target_p, target_q, y_ref, oracle = _build_case(case_name, alpha)
    funcs = _make_functions(
        params_base=params,
        target_p=target_p,
        target_q=target_q,
        position_scale=args.position_scale,
        so3_scale=args.so3_scale,
        boundary_scale=args.boundary_scale,
        steps_per_flex=args.steps_per_flex,
    )
    init_map = {
        "oracle": y_ref[:6],
        "half_oracle": 0.5 * y_ref[:6],
        "zero": np.zeros(6, dtype=float),
    }
    _ = funcs["residual_jit"](jnp.asarray(init_map["oracle"]), jnp.asarray(I0)).block_until_ready()
    rows: list[dict[str, Any]] = []
    solves: dict[str, dict[str, Any]] = {}
    length = 0.11
    for init_name, y0 in init_map.items():
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "case": case_name, "alpha": alpha, "init": init_name})
        solve = _solve_base_wrench(y0=y0, I=I0, funcs=funcs, max_iter=args.max_iter)
        solves[init_name] = solve
        task = np.asarray(solve["task"], dtype=float)
        terminal_position_error = float(np.linalg.norm(task[:3] - target_p))
        terminal_so3_error_deg = float(np.linalg.norm(task[3:6]) * 180.0 / np.pi)
        trusted = bool(
            solve["success"]
            and terminal_position_error <= 0.05 * length
            and terminal_so3_error_deg <= 10.0
            and solve["boundary_norm"] <= 0.3
            and solve["stationarity_norm"] <= 1.0e-5
            and solve["jacobian_condition"] <= 1.0e10
        )
        row = {
            "case": case_name,
            "alpha": alpha,
            "init_name": init_name,
            "success": solve["success"],
            "trusted_gate": trusted,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "residual_norm": solve["residual_norm"],
            "stationarity_norm": solve["stationarity_norm"],
            "boundary_norm": solve["boundary_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "terminal_position_error": terminal_position_error,
            "terminal_so3_error_deg": terminal_so3_error_deg,
            "base_wrench": solve["y"].tolist(),
            "tip_p": np.asarray(solve["out"]["tip_p"], dtype=float).tolist(),
            "message": solve["message"],
        }
        rows.append(row)
        if not trusted:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "case": case_name,
                    "alpha": alpha,
                    "init_name": init_name,
                    "failure": "continuous_base_wrench_gate_failed",
                    "evidence": row,
                },
            )
    pair_rows: list[dict[str, Any]] = []
    for a, b in combinations(sorted(solves), 2):
        ca = np.asarray(solves[a]["out"]["centerline"], dtype=float)
        cb = np.asarray(solves[b]["out"]["centerline"], dtype=float)
        pair_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "branch_a": a,
                "branch_b": b,
                "tip_distance": float(np.linalg.norm(np.asarray(solves[a]["out"]["tip_p"]) - np.asarray(solves[b]["out"]["tip_p"]))),
                "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                "state_distance": float(np.linalg.norm(solves[a]["y"] - solves[b]["y"])),
                "both_success": bool(solves[a]["success"] and solves[b]["success"]),
            }
        )
    summary = {
        "case": case_name,
        "alpha": alpha,
        "trusted_count": sum(1 for row in rows if row["trusted_gate"]),
        "success_count": sum(1 for row in rows if row["success"]),
        "branch_count": len(rows),
        "max_tip_distance": max([row["tip_distance"] for row in pair_rows], default=0.0),
        "max_centerline_rms": max([row["centerline_rms"] for row in pair_rows], default=0.0),
        "max_state_distance": max([row["state_distance"] for row in pair_rows], default=0.0),
        "min_terminal_position_error": min(row["terminal_position_error"] for row in rows),
        "min_terminal_so3_error_deg": min(row["terminal_so3_error_deg"] for row in rows),
        "min_boundary_norm": min(row["boundary_norm"] for row in rows),
        "min_stationarity_norm": min(row["stationarity_norm"] for row in rows),
        "min_jacobian_condition": min(row["jacobian_condition"] for row in rows),
        "branch_consistent": bool(
            all(row["success"] for row in rows)
            and max([row["tip_distance"] for row in pair_rows], default=0.0) <= 1.0e-4
            and max([row["centerline_rms"] for row in pair_rows], default=0.0) <= 1.0e-3
        ),
    }
    np.savez_compressed(
        OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_base_wrench_branches.npz",
        fk_oracle_centerline=np.asarray(oracle["centerline"], dtype=float),
        target_p=target_p,
        target_q=target_q,
        **{f"{name}_y": solve["y"] for name, solve in solves.items()},
        **{f"{name}_centerline": np.asarray(solve["out"]["centerline"], dtype=float) for name, solve in solves.items()},
        **{f"{name}_tip_p": np.asarray(solve["out"]["tip_p"], dtype=float) for name, solve in solves.items()},
    )
    _plot_branches(case_name, alpha, oracle, solves)
    return rows, pair_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00")
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=120.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case_name, alpha in _parse_candidates(args.candidates):
        rows, pair_rows, summary = _run_case(
            case_name=case_name,
            alpha=alpha,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        all_rows.extend(rows)
        all_pair_rows.extend(pair_rows)
        summaries.append(summary)
    wall_s = time.perf_counter() - t_all
    _write_csv(OUT_DIR / "branch_rows.csv", all_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", all_pair_rows)
    _plot_metrics(all_rows, all_pair_rows)
    summary_payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(wall_s),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "case_summaries": summaries,
        "overall": {
            "trusted_count": sum(row["trusted_gate"] for row in all_rows),
            "success_count": sum(row["success"] for row in all_rows),
            "row_count": len(all_rows),
            "branch_consistent_count": sum(item["branch_consistent"] for item in summaries),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary_payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
