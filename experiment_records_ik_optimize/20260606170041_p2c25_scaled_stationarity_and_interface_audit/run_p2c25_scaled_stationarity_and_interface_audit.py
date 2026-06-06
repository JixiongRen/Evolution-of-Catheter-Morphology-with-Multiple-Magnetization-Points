"""Audit numerical scaling effects on shooting stationarity."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
for path in (ROOT, P2C24_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c24_interface_weight_continuation_repair as p2c24


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


def _parse_specs(text: str) -> list[tuple[str, float, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        case, alpha, eta = raw.split(":", 2)
        specs.append((case.strip(), float(alpha), float(eta)))
    return specs


def _physical_metrics(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray, boundary_scale: float, slack_scale: float) -> dict[str, Any]:
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual_jit"](y_j, I_j, b_j, s_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["Jy_jit"](y_j, I_j, b_j, s_j)), dtype=float)
    out = p2c24.p2c10w.output_to_numpy(funcs["output_jit"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "r": r,
        "J": J,
        "out": out,
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
    }


def _column_scale(funcs: dict[str, Any], y0: np.ndarray, I: np.ndarray, boundary_scale: float, slack_scale: float) -> np.ndarray:
    metrics = _physical_metrics(funcs, y0, I, boundary_scale, slack_scale)
    col_norm = np.linalg.norm(metrics["J"], axis=0)
    scale = 1.0 / np.maximum(col_norm, 1.0e-8)
    return np.clip(scale, 1.0e-5, 1.0e-1)


def _seed_abs_scale(y0: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y0, dtype=float)
    return np.maximum(np.abs(y0), 1.0e-3)


def _solve_strategy(
    *,
    strategy: str,
    funcs: dict[str, Any],
    y0: np.ndarray,
    I: np.ndarray,
    target_p: np.ndarray,
    params: Any,
    boundary_scale: float,
    slack_scale: float,
    max_iter: int,
) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    t0 = time.perf_counter()

    if strategy in {"baseline_y", "scipy_jac_xscale"}:
        opt = p2c24.p2c10w.least_squares(
            lambda yy: np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
            y0,
            jac=lambda yy: np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
            method="trf",
            x_scale=("jac" if strategy == "scipy_jac_xscale" else np.ones_like(y0)),
            loss="linear",
            max_nfev=int(max_iter),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        y = np.asarray(opt.x, dtype=float)
        transform_scale = np.ones_like(y0)
    elif strategy in {"seed_abs_variable", "jac_column_variable"}:
        transform_scale = _seed_abs_scale(y0) if strategy == "seed_abs_variable" else _column_scale(funcs, y0, I, boundary_scale, slack_scale)
        u0 = np.zeros_like(y0)

        def residual_u(u: np.ndarray) -> np.ndarray:
            y = y0 + transform_scale * np.asarray(u, dtype=float)
            return np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)

        def jac_u(u: np.ndarray) -> np.ndarray:
            y = y0 + transform_scale * np.asarray(u, dtype=float)
            J = np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)
            return J * transform_scale.reshape(1, -1)

        opt = p2c24.p2c10w.least_squares(
            residual_u,
            u0,
            jac=jac_u,
            method="trf",
            x_scale=np.ones_like(u0),
            loss="linear",
            max_nfev=int(max_iter),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        y = y0 + transform_scale * np.asarray(opt.x, dtype=float)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    steady_s = time.perf_counter() - t0
    metrics = _physical_metrics(funcs, y, I, boundary_scale, slack_scale)
    J_scaled = metrics["J"] * transform_scale.reshape(1, -1)
    svals_scaled = np.linalg.svd(J_scaled, compute_uv=False)
    load = p2c24._audit_load(params, I, y)
    pos_err = float(np.linalg.norm(np.asarray(metrics["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "strategy": strategy,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "message": str(opt.message),
        "position_error_m": pos_err,
        "residual_norm": metrics["residual_norm"],
        "stationarity_norm": metrics["stationarity_norm"],
        "scaled_stationarity_norm": float(np.linalg.norm(J_scaled.T @ metrics["r"])),
        "boundary_norm": metrics["boundary_norm"],
        "jacobian_condition": metrics["jacobian_condition"],
        "scaled_jacobian_condition": float(svals_scaled[0] / max(float(svals_scaled[-1]), np.finfo(float).tiny)) if svals_scaled.size else float("inf"),
        "jacobian_rank": metrics["jacobian_rank"],
        "transform_scale_min": float(np.min(transform_scale)),
        "transform_scale_max": float(np.max(transform_scale)),
        "slack_norm": float(np.linalg.norm(y[6:])),
        **load,
    }
    row["trusted_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
    )
    return row


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes = []
    if not row["success"]:
        modes.append("求解器达到迭代上限")
    if row["stationarity_norm"] > 1.0e-5:
        modes.append("一阶最优性退化")
    if row["jacobian_condition"] > 1.0e10:
        modes.append("数值病态或局部不可辨识")
    if row["max_slack_force_over_rigid_force"] > 2.0:
        modes.append("刚性磁体段载荷传播失配")
    if row["position_error_m"] > 0.05 * 0.11:
        modes.append("可达域或末端约束失配")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    for item in ["一阶最优性退化", "刚性磁体段载荷传播失配", "数值病态或局部不可辨识", "可达域或末端约束失配"]:
        if item in modes:
            return item
    return modes[0]


def _plot_case(rows: list[dict[str, Any]], case_name: str) -> None:
    subset = [r for r in rows if r["case"] == case_name]
    labels = [r["strategy"] for r in subset]
    x = np.arange(len(labels))
    colors = ["green" if r["trusted_gate"] else "orange" for r in subset]
    fig, axes = plt.subplots(4, 1, figsize=(max(9, 1.1 * len(labels)), 11), sharex=True)
    specs = [
        ("stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("jacobian_condition", "物理雅可比条件数", 1.0e10),
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
    axes[-1].set_xticklabels(labels, rotation=25, ha="right")
    fig.suptitle(f"{case_name} 尺度修复诊断")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{case_name}_scaling_audit.png", dpi=180)
    plt.close(fig)


def _plot_summary(summaries: list[dict[str, Any]]) -> None:
    labels = [s["case"] for s in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, key, ylabel, gate in [
        (axes[0], "best_stationarity_norm", "最佳一阶最优性残差", 1.0e-5),
        (axes[1], "best_jacobian_condition", "最佳物理条件数", 1.0e10),
        (axes[2], "best_slack_force_ratio", "最佳界面力松弛/刚段外力", 2.0),
    ]:
        ax.bar(x, [s[key] for s in summaries], color=["green" if s["trusted_gate"] else "orange" for s in summaries])
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaled_stationarity_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10:1.00,n3_mixed_base_i4:1.00:0.75")
    parser.add_argument("--strategies", default="baseline_y,scipy_jac_xscale,seed_abs_variable,jac_column_variable")
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.50)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    for case_name, alpha, eta in _parse_specs(args.specs):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha, "eta": eta})
        params, I0 = p2c24._params_from_saved_case(case_name, alpha)
        target_p, y_saved = p2c24._load_saved_target_and_seed(case_name, alpha)
        target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
        funcs = p2c24._make_functions(params, target_p, target_q, args)
        y0 = p2c24._seed_from_reference(y_saved, eta=eta, base_scale=1.0)
        for strategy in strategies:
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_start", "case": case_name, "alpha": alpha, "eta": eta, "strategy": strategy})
            row = _solve_strategy(
                strategy=strategy,
                funcs=funcs,
                y0=y0,
                I=I0,
                target_p=target_p,
                params=params,
                boundary_scale=args.boundary_scale,
                slack_scale=args.slack_scale,
                max_iter=args.max_iter,
            )
            row.update({"case": case_name, "alpha": alpha, "eta": eta})
            rows.append(row)
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_done", "case": case_name, "strategy": strategy, "stationarity_norm": row["stationarity_norm"], "slack_force_ratio": row["max_slack_force_over_rigid_force"], "steady_s": row["steady_s"]})
        subset = [r for r in rows if r["case"] == case_name]
        best = min(subset, key=lambda r: (not r["trusted_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"], r["position_error_m"]))
        modes = [] if best["trusted_gate"] else _failure_modes(best)
        summary = {
            "case": case_name,
            "alpha": alpha,
            "eta": eta,
            "best_strategy": best["strategy"],
            "trusted_gate": best["trusted_gate"],
            "best_stationarity_norm": best["stationarity_norm"],
            "best_scaled_stationarity_norm": best["scaled_stationarity_norm"],
            "best_jacobian_condition": best["jacobian_condition"],
            "best_scaled_jacobian_condition": best["scaled_jacobian_condition"],
            "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
            "best_position_error_m": best["position_error_m"],
            "primary_failure_mode": "" if best["trusted_gate"] else _primary_mode(modes),
            "failure_modes": "" if best["trusted_gate"] else ";".join(modes),
            "total_steady_s": sum(r["steady_s"] for r in subset),
        }
        summaries.append(summary)
        if not best["trusted_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "scaled_stationarity_repair_failed", "case": case_name, "alpha": alpha, "evidence": summary})
        _plot_case(rows, case_name)

    _write_csv(OUT_DIR / "strategy_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
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
