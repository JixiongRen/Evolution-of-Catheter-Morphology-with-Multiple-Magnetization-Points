"""Scan position weights and audit physical compatibility."""

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
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C10X_DIR):
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


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    boundary_scale: float,
    slack_scale: float,
    steps_per_flex: int,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yIw(y: jnp.ndarray, I: jnp.ndarray, _position_scale: jnp.ndarray) -> Any:
        return p2c10w.p2c10t.forward_interface_slack(
            params_for_I(I),
            y,
            variant="wrench",
            steps_per_flex=steps_per_flex,
        )

    def residual_yIw(y: jnp.ndarray, I: jnp.ndarray, position_scale: jnp.ndarray) -> jnp.ndarray:
        out = output_yIw(y, I, position_scale)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        r_pos = position_scale * (out.tip_p - target_p_j)
        r_boundary = float(boundary_scale) * out.boundary
        r_slack = float(slack_scale) * y[6:]
        return jnp.concatenate([r_pos.reshape(-1), r_boundary.reshape(-1), r_slack.reshape(-1)], axis=0)

    def unweighted_blocks_yIw(y: jnp.ndarray, I: jnp.ndarray, position_scale: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        out = output_yIw(y, I, position_scale)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        return out.tip_p - target_p_j, out.boundary, y[6:]

    return {
        "residual": jax.jit(residual_yIw),
        "J_residual": jax.jit(jax.jacfwd(residual_yIw, argnums=0)),
        "blocks": jax.jit(unweighted_blocks_yIw),
        "J_position_unweighted": jax.jit(jax.jacfwd(lambda y, I, w: unweighted_blocks_yIw(y, I, w)[0], argnums=0)),
        "J_boundary_unweighted": jax.jit(jax.jacfwd(lambda y, I, w: unweighted_blocks_yIw(y, I, w)[1], argnums=0)),
        "J_slack_unweighted": jax.jit(jax.jacfwd(lambda y, I, w: unweighted_blocks_yIw(y, I, w)[2], argnums=0)),
        "output": jax.jit(output_yIw),
    }


def _solve(
    *,
    y0: np.ndarray,
    I: np.ndarray,
    position_scale: float,
    boundary_scale: float,
    slack_scale: float,
    funcs: dict[str, Any],
    max_iter: int,
) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    w_j = jnp.asarray(position_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j, w_j).block_until_ready()
    t0 = time.perf_counter()
    opt = scipy.optimize.least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j, w_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["J_residual"](jnp.asarray(yy), I_j, w_j)), dtype=float),
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
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j, w_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["J_residual"](y_j, I_j, w_j)), dtype=float)
    pos, boundary, slack = [np.asarray(x, dtype=float).reshape(-1) for x in jax.device_get(funcs["blocks"](y_j, I_j, w_j))]
    Jp = np.asarray(jax.device_get(funcs["J_position_unweighted"](y_j, I_j, w_j)), dtype=float)
    Jb = np.asarray(jax.device_get(funcs["J_boundary_unweighted"](y_j, I_j, w_j)), dtype=float)
    Js = np.asarray(jax.device_get(funcs["J_slack_unweighted"](y_j, I_j, w_j)), dtype=float)
    gp = Jp.T @ pos
    gb = Jb.T @ boundary
    gs = Js.T @ slack
    g_weighted = J.T @ r
    svals = np.linalg.svd(J, compute_uv=False)
    out = p2c10w.output_to_numpy(funcs["output"](y_j, I_j, w_j))
    return {
        "y": y,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "message": str(opt.message),
        "steady_s": float(steady_s),
        "weighted_residual_norm": float(np.linalg.norm(r)),
        "weighted_stationarity_norm": float(np.linalg.norm(g_weighted)),
        "weighted_jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
        "weighted_jacobian_rank": int(np.linalg.matrix_rank(J)),
        "position_error_m": float(np.linalg.norm(pos)),
        "boundary_norm": float(np.linalg.norm(boundary)),
        "slack_norm": float(np.linalg.norm(slack)),
        "unweighted_position_gradient_norm": float(np.linalg.norm(gp)),
        "unweighted_boundary_gradient_norm": float(np.linalg.norm(gb)),
        "unweighted_slack_gradient_norm": float(np.linalg.norm(gs)),
        "unweighted_total_gradient_norm": float(np.linalg.norm(gp + gb + gs)),
        "weighted_position_gradient_norm": float(np.linalg.norm((position_scale**2) * gp)),
        "weighted_boundary_gradient_norm": float(np.linalg.norm((boundary_scale**2) * gb)),
        "weighted_slack_gradient_norm": float(np.linalg.norm((slack_scale**2) * gs)),
        "tip_p": np.asarray(out["tip_p"], dtype=float).reshape(3),
        "boundary": np.asarray(out["boundary"], dtype=float).reshape(-1),
    }


def _run_alpha(alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> list[dict[str, Any]]:
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    funcs = _make_functions(
        params_base=params,
        target_p=target_p,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
        steps_per_flex=args.steps_per_flex,
    )
    rows: list[dict[str, Any]] = []
    y_current = _load_seed_y(alpha)
    for position_scale in _parse_floats(args.position_scales):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha, "position_scale": position_scale})
        solve = _solve(
            y0=y_current,
            I=I0,
            position_scale=position_scale,
            boundary_scale=args.boundary_scale,
            slack_scale=args.slack_scale,
            funcs=funcs,
            max_iter=args.max_iter,
        )
        y_current = solve["y"]
        row = {
            "case": "n3_mixed_rand20",
            "alpha": alpha,
            "position_scale": position_scale,
            "boundary_scale": args.boundary_scale,
            "slack_scale": args.slack_scale,
            "max_iter": args.max_iter,
            **{k: v for k, v in solve.items() if k not in {"y", "tip_p", "boundary"}},
        }
        row["task_gate"] = bool(row["position_error_m"] <= 0.05 * 0.11)
        row["strict_publishable"] = bool(
            row["success"]
            and row["task_gate"]
            and row["boundary_norm"] <= 0.3
            and row["weighted_stationarity_norm"] <= 1.0e-5
            and row["weighted_jacobian_condition"] <= 1.0e10
        )
        if not row["strict_publishable"]:
            reasons = []
            if not row["success"]:
                reasons.append("求解器未报告成功")
            if row["weighted_stationarity_norm"] > 1.0e-5:
                reasons.append("加权一阶平稳性未通过")
            if row["position_error_m"] > 0.05 * 0.11:
                reasons.append("末端位置误差未通过")
            _append_jsonl(failures, {"ts": time.time(), "failure": "weight_scan_gate_failed", "reasons": reasons, "evidence": row})
        rows.append(row)
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    alphas = sorted({row["alpha"] for row in rows})
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for alpha in alphas:
        sub = [row for row in rows if row["alpha"] == alpha]
        x = [row["position_scale"] for row in sub]
        axes[0].plot(x, [row["position_error_m"] for row in sub], marker="o", label=f"alpha={alpha:.2f}")
        axes[1].plot(x, [row["boundary_norm"] for row in sub], marker="o", label=f"alpha={alpha:.2f}")
        axes[2].plot(x, [row["weighted_stationarity_norm"] for row in sub], marker="o", label=f"alpha={alpha:.2f}")
    axes[0].set_ylabel("position error (m)")
    axes[1].set_ylabel("boundary norm")
    axes[2].set_ylabel("weighted stationarity")
    axes[2].set_xlabel("position residual weight")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "weight_tradeoff_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for alpha in alphas:
        sub = [row for row in rows if row["alpha"] == alpha]
        x = [row["position_scale"] for row in sub]
        axes[0].plot(x, [row["unweighted_position_gradient_norm"] for row in sub], marker="o", label=f"position alpha={alpha:.2f}")
        axes[0].plot(x, [row["unweighted_boundary_gradient_norm"] for row in sub], marker="s", ls="--", label=f"boundary alpha={alpha:.2f}")
        axes[1].plot(x, [row["weighted_position_gradient_norm"] for row in sub], marker="o", label=f"position alpha={alpha:.2f}")
        axes[1].plot(x, [row["weighted_boundary_gradient_norm"] for row in sub], marker="s", ls="--", label=f"boundary alpha={alpha:.2f}")
    axes[0].set_ylabel("unweighted gradient norm")
    axes[1].set_ylabel("weighted component gradient")
    axes[1].set_xlabel("position residual weight")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gradient_component_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.40,0.45")
    parser.add_argument("--position_scales", default="15,30,60,120,240")
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=0.05)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=180)
    args = parser.parse_args()

    t0 = time.perf_counter()
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
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
            "publishable_count": sum(row["strict_publishable"] for row in rows),
            "task_gate_count": sum(row["task_gate"] for row in rows),
        },
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
