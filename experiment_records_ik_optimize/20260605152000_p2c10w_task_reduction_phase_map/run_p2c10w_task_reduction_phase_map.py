"""Map task-reduction gates for interface-slack shooting."""

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
P2C10T_DIR = ROOT / "experiment_records_ik_optimize" / "20260605102000_p2c10t_interface_slack_shooting"
for path in (ROOT, FK_OPT_DIR, P2C10T_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basics_nondim import quat_normalize, quat_to_rotmat
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import output_to_numpy

import run_p2c10t_interface_slack_shooting as p2c10t

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
AXIS_BODY = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64)


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


def _axis_from_q(q: jnp.ndarray) -> jnp.ndarray:
    return quat_to_rotmat(quat_normalize(q)) @ AXIS_BODY


def _axis_angle_error(current_q: np.ndarray, target_q: np.ndarray) -> float:
    cur = np.asarray(jax.device_get(_axis_from_q(jnp.asarray(current_q, dtype=jnp.float64))), dtype=float)
    tar = np.asarray(jax.device_get(_axis_from_q(jnp.asarray(target_q, dtype=jnp.float64))), dtype=float)
    c = float(np.clip(np.dot(cur, tar) / max(np.linalg.norm(cur) * np.linalg.norm(tar), 1e-30), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _build_case(case_name: str, alpha: float):
    return p2c10t._build_case(case_name, alpha)


def _initial_vector(y_ref: np.ndarray, init_name: str, params: Any) -> np.ndarray:
    n_interfaces = max(len(params.flex) - 1, 0)
    y = np.zeros(6 + 6 * n_interfaces, dtype=float)
    if init_name == "oracle":
        y[:6] = y_ref[:6]
    elif init_name == "half_oracle":
        y[:6] = 0.5 * y_ref[:6]
    elif init_name != "zero":
        raise ValueError(f"Unknown init: {init_name}")
    return y


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    task_kind: str,
    position_scale: float,
    axis_scale: float,
    so3_scale: float,
    boundary_scale: float,
    slack_scale: float,
    steps_per_flex: int,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    target_axis = _axis_from_q(target_q_j)

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=steps_per_flex)

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        q_res = quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        axis_res = jnp.cross(target_axis, _axis_from_q(out.tip_q))
        return jnp.concatenate([out.tip_p.reshape(-1), axis_res.reshape(-1), q_res.reshape(-1)], axis=0)

    def task_residual(out: Any) -> jnp.ndarray:
        parts = []
        if task_kind in {"position", "position_axis", "full_pose"}:
            parts.append(float(position_scale) * (out.tip_p - target_p_j))
        if task_kind in {"axis", "position_axis"}:
            parts.append(float(axis_scale) * jnp.cross(target_axis, _axis_from_q(out.tip_q)))
        if task_kind == "full_pose":
            parts.append(float(so3_scale) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0))
        if not parts:
            raise ValueError(f"Unknown task kind: {task_kind}")
        return jnp.concatenate([p.reshape(-1) for p in parts], axis=0)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        slack = jnp.asarray(y, dtype=jnp.float64).reshape(-1)[6:]
        r_task = task_residual(out)
        r_b = float(boundary_scale) * out.boundary
        r_slack = float(slack_scale) * slack
        return jnp.concatenate([r_task.reshape(-1), r_b.reshape(-1), r_slack.reshape(-1)], axis=0)

    return {
        "residual_jit": jax.jit(residual_yI),
        "task_jit": jax.jit(task_yI),
        "output_jit": jax.jit(output_yI),
        "Jy_jit": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], max_iter: int) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
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
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "task": np.asarray(jax.device_get(funcs["task_jit"](y_j, I_j)), dtype=float),
        "message": str(opt.message),
    }


def _task_gate(task_kind: str, pos_err: float, axis_err_deg: float, so3_err_deg: float) -> bool:
    length = 0.11
    if task_kind == "position":
        return pos_err <= 0.05 * length
    if task_kind == "axis":
        return axis_err_deg <= 10.0
    if task_kind == "position_axis":
        return pos_err <= 0.05 * length and axis_err_deg <= 10.0
    if task_kind == "full_pose":
        return pos_err <= 0.05 * length and so3_err_deg <= 10.0
    raise ValueError(task_kind)


def _run_one(case_name: str, alpha: float, task_kind: str, args: argparse.Namespace, heartbeat: Path, failures: Path):
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha, "task_kind": task_kind})
    params, I0, target_p, target_q, y_ref, oracle = _build_case(case_name, alpha)
    funcs = _make_functions(
        params_base=params,
        target_p=target_p,
        target_q=target_q,
        task_kind=task_kind,
        position_scale=args.position_scale,
        axis_scale=args.axis_scale,
        so3_scale=args.so3_scale,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
        steps_per_flex=args.steps_per_flex,
    )
    _ = funcs["residual_jit"](jnp.asarray(_initial_vector(y_ref, "oracle", params)), jnp.asarray(I0)).block_until_ready()
    rows = []
    solves = {}
    for init_name in ["oracle", "half_oracle", "zero"]:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "case": case_name, "alpha": alpha, "task_kind": task_kind, "init": init_name})
        solve = _solve(_initial_vector(y_ref, init_name, params), I0, funcs, args.max_iter)
        solves[init_name] = solve
        out = solve["out"]
        pos_err = float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float) - target_p))
        so3_err_deg = float(np.linalg.norm(np.asarray(solve["task"][6:9], dtype=float)) * 180.0 / np.pi)
        axis_err_deg = _axis_angle_error(out["tip_q"], target_q)
        task_ok = _task_gate(task_kind, pos_err, axis_err_deg, so3_err_deg)
        trusted = bool(
            solve["success"]
            and task_ok
            and solve["boundary_norm"] <= 0.3
            and solve["stationarity_norm"] <= 1.0e-5
            and solve["jacobian_condition"] <= 1.0e10
            and np.linalg.norm(solve["y"][6:]) <= 5.0
        )
        row = {
            "case": case_name,
            "alpha": alpha,
            "task_kind": task_kind,
            "init_name": init_name,
            "success": solve["success"],
            "task_gate": task_ok,
            "trusted_gate": trusted,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "residual_norm": solve["residual_norm"],
            "stationarity_norm": solve["stationarity_norm"],
            "boundary_norm": solve["boundary_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "terminal_position_error": pos_err,
            "terminal_axis_error_deg": axis_err_deg,
            "terminal_so3_error_deg": so3_err_deg,
            "slack_norm": float(np.linalg.norm(solve["y"][6:])),
            "tip_p": np.asarray(out["tip_p"], dtype=float).tolist(),
            "message": solve["message"],
        }
        rows.append(row)
        if not trusted:
            _append_jsonl(failures, {"ts": time.time(), "failure": "task_reduction_gate_failed", "evidence": row})
    pair_rows = []
    for a, b in combinations(sorted(solves), 2):
        ca = np.asarray(solves[a]["out"]["centerline"], dtype=float)
        cb = np.asarray(solves[b]["out"]["centerline"], dtype=float)
        pair_rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "task_kind": task_kind,
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
        "task_kind": task_kind,
        "trusted_count": sum(row["trusted_gate"] for row in rows),
        "task_gate_count": sum(row["task_gate"] for row in rows),
        "success_count": sum(row["success"] for row in rows),
        "branch_count": len(rows),
        "min_position_error": min(row["terminal_position_error"] for row in rows),
        "min_axis_error_deg": min(row["terminal_axis_error_deg"] for row in rows),
        "min_so3_error_deg": min(row["terminal_so3_error_deg"] for row in rows),
        "min_boundary_norm": min(row["boundary_norm"] for row in rows),
        "min_stationarity_norm": min(row["stationarity_norm"] for row in rows),
        "min_jacobian_condition": min(row["jacobian_condition"] for row in rows),
        "max_tip_distance": max([row["tip_distance"] for row in pair_rows], default=0.0),
        "max_centerline_rms": max([row["centerline_rms"] for row in pair_rows], default=0.0),
        "branch_consistent": bool(
            all(row["success"] and row["task_gate"] for row in rows)
            and max([row["tip_distance"] for row in pair_rows], default=0.0) <= 1.0e-4
            and max([row["centerline_rms"] for row in pair_rows], default=0.0) <= 1.0e-3
        ),
    }
    np.savez_compressed(
        OUT_DIR / f"{case_name}_alpha_{alpha:.2f}_{task_kind}_branches.npz",
        fk_oracle_centerline=np.asarray(oracle["centerline"], dtype=float),
        target_p=target_p,
        target_q=target_q,
        **{f"{name}_y": solve["y"] for name, solve in solves.items()},
        **{f"{name}_centerline": np.asarray(solve["out"]["centerline"], dtype=float) for name, solve in solves.items()},
        **{f"{name}_tip_p": np.asarray(solve["out"]["tip_p"], dtype=float) for name, solve in solves.items()},
    )
    return rows, pair_rows, summary


def _plot_phase(summaries: list[dict[str, Any]]) -> None:
    cases = sorted({f"{row['case']}@{row['alpha']:.2f}" for row in summaries})
    tasks = ["position", "axis", "position_axis", "full_pose"]
    metrics = [
        ("min_stationarity_norm", "stationarity", True),
        ("min_boundary_norm", "boundary", True),
        ("max_centerline_rms", "branch RMS", False),
        ("task_gate_count", "task gate count", False),
    ]
    data = {(f"{row['case']}@{row['alpha']:.2f}", row["task_kind"]): row for row in summaries}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, log_scale) in zip(axes.ravel(), metrics):
        mat = np.full((len(cases), len(tasks)), np.nan)
        for i, case in enumerate(cases):
            for j, task in enumerate(tasks):
                if (case, task) in data:
                    val = float(data[(case, task)][key])
                    mat[i, j] = np.log10(max(val, 1e-30)) if log_scale else val
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(title + (" (log10)" if log_scale else ""))
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels(tasks, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(cases)))
        ax.set_yticklabels(cases)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_reduction_phase_map.png", dpi=180)
    plt.close(fig)


def _plot_branch_metrics(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['case']}@{r['alpha']:.2f}\n{r['task_kind']}\n{r['init_name']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(max(14, 0.45 * len(rows)), 11), sharex=True)
    for ax, key, ylabel, log in [
        (axes[0], "terminal_position_error", "position error (m)", True),
        (axes[1], "terminal_axis_error_deg", "axis error (deg)", False),
        (axes[2], "boundary_norm", "boundary", True),
        (axes[3], "stationarity_norm", "stationarity", True),
    ]:
        ax.bar(x, [float(r[key]) for r in rows])
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=80, ha="right", fontsize=5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task_reduction_branch_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_rand20:0.40,n3_mixed_base_i4:1.00")
    parser.add_argument("--tasks", default="position,axis,position_axis,full_pose")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=120.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=0.05)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for case_name, alpha in _parse_candidates(args.candidates):
        for task_kind in tasks:
            rows, pairs, summary = _run_one(case_name, alpha, task_kind, args, heartbeat, failures)
            all_rows.extend(rows)
            pair_rows.extend(pairs)
            summaries.append(summary)
    _write_csv(OUT_DIR / "branch_rows.csv", all_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", pair_rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _plot_phase(summaries)
    _plot_branch_metrics(all_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "overall": {
            "combo_count": len(summaries),
            "trusted_combo_count": sum(row["trusted_count"] > 0 for row in summaries),
            "branch_consistent_count": sum(row["branch_consistent"] for row in summaries),
            "task_gate_combo_count": sum(row["task_gate_count"] > 0 for row in summaries),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
