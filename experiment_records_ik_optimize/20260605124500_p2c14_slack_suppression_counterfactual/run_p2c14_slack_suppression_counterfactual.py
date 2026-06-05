"""Run counterfactual suppression of nonphysical interface slack."""

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
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C10X_DIR, P2C13_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

import run_p2c10w_task_reduction_phase_map as p2c10w
import run_p2c13_rigid_load_propagation_audit as p2c13

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


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_seed_y(alpha: float) -> np.ndarray:
    path = P2C10X_DIR / f"n3_mixed_rand20_alpha_{alpha:.2f}_position_linearity.npz"
    return np.asarray(np.load(path)["best_y"], dtype=float)


def _make_functions(params_base: Any, target_p: np.ndarray, position_scale: float, boundary_scale: float, steps_per_flex: int):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c10w.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray, slack_scale: jnp.ndarray) -> jnp.ndarray:
        out = output_yI(y, I)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return jnp.concatenate(
            [
                float(position_scale) * (out.tip_p - target_p_j),
                float(boundary_scale) * out.boundary,
                slack_scale * y[6:],
            ],
            axis=0,
        )

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], slack_scale: float, max_iter: int) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j, s_j).block_until_ready()
    t0 = time.perf_counter()
    opt = scipy.optimize.least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j, s_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy), I_j, s_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=max_iter,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j, s_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j, s_j)), dtype=float)
    out = p2c10w.output_to_numpy(funcs["output"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "y": y,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "message": str(opt.message),
        "steady_s": float(steady_s),
        "position_error_m": float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float).reshape(3) - target_p_global)),
        "boundary_norm": float(np.linalg.norm(np.asarray(out["boundary"], dtype=float))),
        "slack_norm": float(np.linalg.norm(y[6:])),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "residual_norm": float(np.linalg.norm(r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
    }


target_p_global: np.ndarray


def _audit_load(params: Any, y: np.ndarray) -> dict[str, float]:
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "max_slack_force_over_rigid_force": max((float(row["slack_force_over_rigid_force"]) for row in rows), default=float("nan")),
        "max_slack_torque_over_rigid_torque": max((float(row["slack_torque_over_rigid_torque"]) for row in rows), default=float("nan")),
        "min_slack_force_cosine_rigid_force": min((float(row["slack_force_cosine_rigid_force"]) for row in rows), default=float("nan")),
        "min_slack_torque_cosine_rigid_torque": min((float(row["slack_torque_cosine_rigid_torque"]) for row in rows), default=float("nan")),
        "trace_boundary_norm": float(np.linalg.norm(trace["boundary"])),
    }


def _run_alpha(alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> list[dict[str, Any]]:
    global target_p_global
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    target_p_global = np.asarray(target_p, dtype=float).reshape(3)
    funcs = _make_functions(params, target_p_global, args.position_scale, args.boundary_scale, args.steps_per_flex)
    rows: list[dict[str, Any]] = []
    y_current = _load_seed_y(alpha)
    for slack_scale in _parse_floats(args.slack_scales):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "slack_scale": slack_scale})
        solve = _solve(y_current, I0, funcs, slack_scale, args.max_iter)
        y_current = solve["y"]
        audit = _audit_load(params, y_current)
        row = {
            "case": "n3_mixed_rand20",
            "alpha": alpha,
            "position_scale": args.position_scale,
            "boundary_scale": args.boundary_scale,
            "slack_scale": slack_scale,
            "max_iter": args.max_iter,
            **{k: v for k, v in solve.items() if k != "y"},
            **audit,
        }
        row["task_gate"] = bool(row["position_error_m"] <= 0.05 * 0.11)
        row["load_gate"] = bool(row["max_slack_force_over_rigid_force"] <= 2.0)
        row["single_point_candidate"] = bool(
            row["success"]
            and row["task_gate"]
            and row["load_gate"]
            and row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] <= 1.0e10
        )
        if not row["single_point_candidate"]:
            reasons = []
            if not row["success"]:
                reasons.append("求解器未报告成功")
            if not row["task_gate"]:
                reasons.append("末端位置误差未通过")
            if not row["load_gate"]:
                reasons.append("界面力松弛仍超过物理载荷门限")
            if row["stationarity_norm"] > 1.0e-5:
                reasons.append("一阶平稳性未通过")
            _append_jsonl(failures, {"ts": time.time(), "failure": "slack_suppression_gate_failed", "reasons": reasons, "evidence": row})
        rows.append(row)
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    alphas = sorted({row["alpha"] for row in rows})
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    keys = [
        ("position_error_m", "position error (m)"),
        ("boundary_norm", "boundary norm"),
        ("max_slack_force_over_rigid_force", "max force slack/load"),
        ("stationarity_norm", "stationarity"),
    ]
    for alpha in alphas:
        sub = [row for row in rows if row["alpha"] == alpha]
        x = [row["slack_scale"] for row in sub]
        for ax, (key, _label) in zip(axes, keys):
            ax.plot(x, [row[key] for row in sub], marker="o", label=f"alpha={alpha:.2f}")
    for ax, (_key, label) in zip(axes, keys):
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("interface slack penalty")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slack_suppression_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.40,0.45")
    parser.add_argument("--slack_scales", default="0.05,0.2,1,5,20")
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=180)
    args = parser.parse_args()
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for alpha in _parse_floats(args.alphas):
        rows.extend(_run_alpha(alpha, args, heartbeat, failures))
    _write_csv(OUT_DIR / "summary.csv", rows)
    _plot(rows)
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
            "case_count": len(rows),
            "single_point_candidate_count": sum(row["single_point_candidate"] for row in rows),
            "task_gate_count": sum(row["task_gate"] for row in rows),
            "load_gate_count": sum(row["load_gate"] for row in rows),
        },
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
