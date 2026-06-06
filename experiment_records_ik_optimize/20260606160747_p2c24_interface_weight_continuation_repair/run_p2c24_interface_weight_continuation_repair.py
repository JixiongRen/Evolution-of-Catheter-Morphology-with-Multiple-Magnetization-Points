"""Probe interface-state and residual-weight continuation for shooting IK."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
P2C22_DIR = ROOT / "experiment_records_ik_optimize" / "20260606152302_p2c22_cross_condition_reliability_pilot"
for path in (ROOT, FK_OPT_DIR, P2C4_DIR, P2C10W_DIR, P2C13_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from forward_kinematics_optimized.equilibrium_solver_nondim import with_L1_dim, with_coil_currents

import run_clean_multi_case as p2c4
import run_p2c10w_task_reduction_phase_map as p2c10w
import run_p2c13_rigid_load_propagation_audit as p2c13


configure_matplotlib_cjk()
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
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        case, alpha = raw.split(":", 1)
        specs.append((case.strip(), float(alpha)))
    return specs


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _seed_from_reference(y_ref: np.ndarray, eta: float, base_scale: float = 1.0) -> np.ndarray:
    y = np.asarray(y_ref, dtype=float).reshape(-1).copy()
    y[:6] *= float(base_scale)
    y[6:] *= float(eta)
    return y


def _case_spec(case_name: str) -> Any:
    for case in p2c4._cases():
        if case.name == case_name:
            return case
    raise ValueError(f"Unknown case: {case_name}")


def _params_from_saved_case(case_name: str, alpha: float) -> tuple[Any, np.ndarray]:
    case = _case_spec(case_name)
    currents = np.asarray(case.currents, dtype=float) * float(alpha)
    engine = p2c4._build_engine(case, max_iter=1, tol=1.0e-5)
    coil = jnp.asarray(currents, dtype=jnp.float64).reshape(8)
    engine._ensure_static_initialized(L_protrude=0.11, coil_currents=coil)
    flex_lengths = engine._compute_flex_lengths(L_protrude=0.11)
    params = with_coil_currents(engine._params_static_base, coil)
    params = with_L1_dim(params, jnp.asarray(flex_lengths[0], dtype=jnp.float64))
    return params, currents


def _load_saved_target_and_seed(case_name: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    path = P2C22_DIR / f"{case_name}_alpha_{alpha:.2f}_position_linearity.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing P2c22 saved state: {path}")
    payload = np.load(path)
    return np.asarray(payload["target_p"], dtype=float), np.asarray(payload["best_y"], dtype=float)


def _make_functions(params: Any, target_p: np.ndarray, target_q: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    target_axis = p2c10w._axis_from_q(target_q_j)

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray):
        return p2c10w.p2c10t.forward_interface_slack(
            params_for_I(I),
            y,
            variant="wrench",
            steps_per_flex=args.steps_per_flex,
        )

    def task_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        q_res = p2c10w.quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        axis_res = jnp.cross(target_axis, p2c10w._axis_from_q(out.tip_q))
        return jnp.concatenate([out.tip_p.reshape(-1), axis_res.reshape(-1), q_res.reshape(-1)], axis=0)

    def task_residual(out: Any) -> jnp.ndarray:
        return float(args.position_scale) * (out.tip_p - target_p_j)

    def residual_yIbs(y: jnp.ndarray, I: jnp.ndarray, boundary_scale: jnp.ndarray, slack_scale: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        slack = jnp.asarray(y, dtype=jnp.float64).reshape(-1)[6:]
        r_task = task_residual(out)
        r_b = jnp.asarray(boundary_scale, dtype=jnp.float64).reshape(()) * out.boundary
        r_slack = jnp.asarray(slack_scale, dtype=jnp.float64).reshape(()) * slack
        return jnp.concatenate([r_task.reshape(-1), r_b.reshape(-1), r_slack.reshape(-1)], axis=0)

    return {
        "residual_jit": jax.jit(residual_yIbs),
        "task_jit": jax.jit(task_yI),
        "output_jit": jax.jit(output_yI),
        "Jy_jit": jax.jit(jax.jacfwd(residual_yIbs, argnums=0)),
    }


def _solve_scaled(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], max_iter: int, boundary_scale: float, slack_scale: float) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    t0 = time.perf_counter()
    opt = p2c10w.least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
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
    r = np.asarray(jax.device_get(funcs["residual_jit"](y_j, I_j, b_j, s_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["Jy_jit"](y_j, I_j, b_j, s_j)), dtype=float)
    out = p2c10w.output_to_numpy(funcs["output_jit"](y_j, I_j))
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


def _audit_load(params_base: Any, I: np.ndarray, y: np.ndarray) -> dict[str, float]:
    params = with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "max_slack_force_over_rigid_force": max((float(row["slack_force_over_rigid_force"]) for row in rows), default=float("nan")),
        "max_slack_torque_over_rigid_torque": max((float(row["slack_torque_over_rigid_torque"]) for row in rows), default=float("nan")),
        "trace_boundary_norm": float(np.linalg.norm(trace["boundary"])),
    }


def _solve_stage(
    *,
    case_name: str,
    alpha: float,
    method: str,
    stage: str,
    y0: np.ndarray,
    params: Any,
    I0: np.ndarray,
    target_p: np.ndarray,
    target_q: np.ndarray,
    funcs: dict[str, Any],
    boundary_scale: float,
    slack_scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    solve = _solve_scaled(y0, I0, funcs, args.max_iter, boundary_scale, slack_scale)
    load = _audit_load(params, I0, solve["y"])
    pos_err = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "case": case_name,
        "alpha": alpha,
        "method": method,
        "stage": stage,
        "boundary_scale": boundary_scale,
        "slack_scale": slack_scale,
        "success": solve["success"],
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "position_error_m": pos_err,
        "boundary_norm": solve["boundary_norm"],
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "slack_norm": float(np.linalg.norm(solve["y"][6:])),
        "residual_norm": solve["residual_norm"],
        **load,
    }
    row["trusted_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
    )
    return row, solve


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes = []
    if not row["success"]:
        modes.append("求解器达到迭代上限")
    if row["stationarity_norm"] > 1.0e-5:
        modes.append("一阶最优性退化")
    if row["max_slack_force_over_rigid_force"] > 2.0:
        modes.append("刚性磁体段载荷传播失配")
    if row["jacobian_condition"] > 1.0e10:
        modes.append("数值病态或局部不可辨识")
    if row["position_error_m"] > 0.05 * 0.11:
        modes.append("可达域或末端约束失配")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    for item in ["一阶最优性退化", "刚性磁体段载荷传播失配", "数值病态或局部不可辨识", "可达域或末端约束失配"]:
        if item in modes:
            return item
    return modes[0]


def _run_case(case_name: str, alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha})
    params, I0 = _params_from_saved_case(case_name, alpha)
    target_p, y_ref = _load_saved_target_and_seed(case_name, alpha)
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    target_p = np.asarray(target_p, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    funcs = _make_functions(params, target_p, target_q, args)
    rows: list[dict[str, Any]] = []
    best_y = None

    for eta in _parse_floats(args.eta_values):
        y0 = _seed_from_reference(y_ref, eta=eta, base_scale=1.0)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "stage_start", "case": case_name, "alpha": alpha, "method": "direct_eta_scan", "stage": f"eta_{eta:.2f}"})
        row, solve = _solve_stage(
            case_name=case_name,
            alpha=alpha,
            method="direct_eta_scan",
            stage=f"eta_{eta:.2f}",
            y0=y0,
            params=params,
            I0=I0,
            target_p=target_p,
            target_q=target_q,
            funcs=funcs,
            boundary_scale=args.target_boundary_scale,
            slack_scale=args.target_slack_scale,
            args=args,
        )
        row["eta"] = eta
        rows.append(row)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "stage_done", "case": case_name, "alpha": alpha, "method": "direct_eta_scan", "stage": f"eta_{eta:.2f}", "stationarity_norm": row["stationarity_norm"], "slack_force_ratio": row["max_slack_force_over_rigid_force"], "steady_s": row["steady_s"]})
        if best_y is None or row["stationarity_norm"] < min(r["stationarity_norm"] for r in rows[:-1]):
            best_y = solve["y"]

    y = _seed_from_reference(y_ref, eta=args.start_eta, base_scale=args.start_base_scale)
    for i, raw_stage in enumerate(args.weight_schedule.split(";")):
        b_text, s_text = raw_stage.split(":", 1)
        boundary_scale = float(b_text)
        slack_scale = float(s_text)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "stage_start", "case": case_name, "alpha": alpha, "method": "weight_continuation", "stage": f"stage_{i}", "boundary_scale": boundary_scale, "slack_scale": slack_scale})
        row, solve = _solve_stage(
            case_name=case_name,
            alpha=alpha,
            method="weight_continuation",
            stage=f"stage_{i}",
            y0=y,
            params=params,
            I0=I0,
            target_p=target_p,
            target_q=target_q,
            funcs=funcs,
            boundary_scale=boundary_scale,
            slack_scale=slack_scale,
            args=args,
        )
        row["eta"] = float("nan")
        rows.append(row)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "stage_done", "case": case_name, "alpha": alpha, "method": "weight_continuation", "stage": f"stage_{i}", "stationarity_norm": row["stationarity_norm"], "slack_force_ratio": row["max_slack_force_over_rigid_force"], "steady_s": row["steady_s"]})
        y = solve["y"]
    best = min(rows, key=lambda r: (not r["trusted_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"], r["position_error_m"]))
    modes = [] if best["trusted_gate"] else _failure_modes(best)
    summary = {
        "case": case_name,
        "alpha": alpha,
        "best_method": best["method"],
        "best_stage": best["stage"],
        "trusted_gate": best["trusted_gate"],
        "best_stationarity_norm": best["stationarity_norm"],
        "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
        "best_position_error_m": best["position_error_m"],
        "best_jacobian_condition": best["jacobian_condition"],
        "best_boundary_norm": best["boundary_norm"],
        "direct_best_stationarity": min(r["stationarity_norm"] for r in rows if r["method"] == "direct_eta_scan"),
        "continuation_final_stationarity": [r["stationarity_norm"] for r in rows if r["method"] == "weight_continuation"][-1],
        "primary_failure_mode": "" if best["trusted_gate"] else _primary_mode(modes),
        "failure_modes": "" if best["trusted_gate"] else ";".join(modes),
        "total_steady_s": sum(r["steady_s"] for r in rows),
    }
    if not best["trusted_gate"]:
        _append_jsonl(failures, {"ts": time.time(), "failure": "continuation_repair_failed", "case": case_name, "alpha": alpha, "evidence": summary})
    return summary, rows, y


def _plot_rows(rows: list[dict[str, Any]]) -> None:
    for case in sorted({r["case"] for r in rows}):
        subset = [r for r in rows if r["case"] == case]
        labels = [f"{r['method']}\n{r['stage']}" for r in subset]
        x = np.arange(len(labels))
        colors = ["green" if r["trusted_gate"] else "orange" for r in subset]
        fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.75 * len(labels)), 9), sharex=True)
        specs = [
            ("stationarity_norm", "一阶最优性", 1.0e-5),
            ("max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
            ("position_error_m", "末端位置误差 m", 0.05 * 0.11),
        ]
        for ax, (key, ylabel, gate) in zip(axes, specs):
            ax.bar(x, [r[key] for r in subset], color=colors)
            ax.axhline(gate, color="black", ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        fig.suptitle(f"{case} 延续修复指标")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case}_continuation_metrics.png", dpi=180)
        plt.close(fig)


def _plot_summary(summaries: list[dict[str, Any]]) -> None:
    labels = [s["case"] for s in summaries]
    x = np.arange(len(labels))
    colors = ["green" if s["trusted_gate"] else "orange" for s in summaries]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, key, ylabel, gate in [
        (axes[0], "best_stationarity_norm", "最佳一阶最优性", 1.0e-5),
        (axes[1], "best_slack_force_ratio", "最佳界面力松弛/刚段外力", 2.0),
        (axes[2], "best_position_error_m", "最佳末端误差 m", 0.05 * 0.11),
    ]:
        ax.bar(x, [s[key] for s in summaries], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "continuation_repair_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n1_tip_z_rand15:0.10,n3_mixed_base_i4:1.00")
    parser.add_argument("--eta_values", default="0.00,0.25,0.50,0.75,1.00")
    parser.add_argument("--start_eta", type=float, default=0.50)
    parser.add_argument("--start_base_scale", type=float, default=0.50)
    parser.add_argument("--weight_schedule", default="0.10:2.0;0.20:5.0;0.35:10.0;0.50:20.0")
    parser.add_argument("--target_boundary_scale", type=float, default=0.50)
    parser.add_argument("--target_slack_scale", type=float, default=20.0)
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries = []
    rows = []
    for case_name, alpha in _parse_candidates(args.candidates):
        summary, case_rows, _final_y = _run_case(case_name, alpha, args, heartbeat, failures)
        summaries.append(summary)
        rows.extend(case_rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "stage_rows.csv", rows)
    _plot_rows(rows)
    _plot_summary(summaries)
    counter = Counter(s["primary_failure_mode"] for s in summaries if s["primary_failure_mode"])
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
            "case_count": len(summaries),
            "trusted_count": sum(s["trusted_gate"] for s in summaries),
            "diagnostic_count": sum(not s["trusted_gate"] for s in summaries),
            "failure_mode_count": dict(counter),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
