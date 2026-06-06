"""Run re-optimized finite-difference sensitivity gates for repaired shooting."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C25_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170041_p2c25_scaled_stationarity_and_interface_audit"
for path in (ROOT, P2C24_DIR, P2C25_DIR):
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
import run_p2c25_scaled_stationarity_and_interface_audit as p2c25


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


def _parse_eps(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _solve_jac_column(
    *,
    funcs: dict[str, Any],
    y0: np.ndarray,
    I: np.ndarray,
    target_p: np.ndarray,
    params: Any,
    boundary_scale: float,
    slack_scale: float,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    scale = p2c25._column_scale(funcs, y0, I, boundary_scale, slack_scale)
    u0 = np.zeros_like(y0)
    t0 = time.perf_counter()

    def residual_u(u: np.ndarray) -> np.ndarray:
        y = y0 + scale * np.asarray(u, dtype=float)
        return np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)

    def jac_u(u: np.ndarray) -> np.ndarray:
        y = y0 + scale * np.asarray(u, dtype=float)
        J = np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)
        return J * scale.reshape(1, -1)

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
    steady_s = time.perf_counter() - t0
    y = y0 + scale * np.asarray(opt.x, dtype=float)
    metrics = p2c25._physical_metrics(funcs, y, I, boundary_scale, slack_scale)
    J_scaled = metrics["J"] * scale.reshape(1, -1)
    svals_scaled = np.linalg.svd(J_scaled, compute_uv=False)
    load = p2c24._audit_load(params, I, y)
    pos_err = float(np.linalg.norm(np.asarray(metrics["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "message": str(opt.message),
        "position_error_m": pos_err,
        "residual_norm": metrics["residual_norm"],
        "stationarity_norm": metrics["stationarity_norm"],
        "boundary_norm": metrics["boundary_norm"],
        "jacobian_condition": metrics["jacobian_condition"],
        "scaled_jacobian_condition": float(svals_scaled[0] / max(float(svals_scaled[-1]), np.finfo(float).tiny)) if svals_scaled.size else float("inf"),
        "scaled_stationarity_norm": float(np.linalg.norm(J_scaled.T @ metrics["r"])),
        "jacobian_rank": metrics["jacobian_rank"],
        "tip_p": np.asarray(metrics["out"]["tip_p"], dtype=float).tolist(),
        **load,
    }
    row["static_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
    )
    return row, y


def _plot_eps(rows: list[dict[str, Any]]) -> None:
    eps = [r["eps_alpha"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(eps, [r["linearization_error_max"] for r in rows], "o-", label="最大线性化误差")
    axes[0].axhline(0.05, color="black", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("线性化误差")
    axes[1].plot(eps, [r["max_perturb_stationarity"] for r in rows], "o-", label="扰动后一阶最优性")
    axes[1].axhline(1.0e-5, color="black", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("一阶最优性")
    axes[2].plot(eps, [r["branch_rms_max"] for r in rows], "o-", label="最大分支漂移")
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("轴线均方根漂移 m")
    axes[2].set_xlabel("alpha 扰动幅值")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_gate_by_eps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_base_i4")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.75)
    parser.add_argument("--eps_values", default="0.005,0.01,0.02")
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--axis_scale", type=float, default=2.0)
    parser.add_argument("--so3_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()

    params, I0 = p2c24._params_from_saved_case(args.case, args.alpha)
    target_p, y_saved = p2c24._load_saved_target_and_seed(args.case, args.alpha)
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    funcs = p2c24._make_functions(params, target_p, target_q, args)
    y0 = p2c24._seed_from_reference(y_saved, eta=args.eta, base_scale=1.0)
    base_currents = np.asarray(p2c24._case_spec(args.case).currents, dtype=float)

    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "baseline_start", "case": args.case, "alpha": args.alpha})
    baseline, y_base = _solve_jac_column(
        funcs=funcs,
        y0=y0,
        I=I0,
        target_p=target_p,
        params=params,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
        max_iter=args.max_iter,
    )
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "baseline_done", "stationarity_norm": baseline["stationarity_norm"], "static_gate": baseline["static_gate"]})
    phi0 = np.asarray(baseline["tip_p"], dtype=float)

    eps_rows: list[dict[str, Any]] = []
    solve_rows: list[dict[str, Any]] = []
    for eps in _parse_eps(args.eps_values):
        phi: dict[str, np.ndarray] = {}
        ys: dict[str, np.ndarray] = {"base": y_base}
        centers: dict[str, np.ndarray] = {}
        for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
            I = I0 + sign * eps * base_currents
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": f"{sign_name}_start", "eps": eps})
            row, y = _solve_jac_column(
                funcs=funcs,
                y0=y_base,
                I=I,
                target_p=target_p,
                params=params,
                boundary_scale=args.boundary_scale,
                slack_scale=args.slack_scale,
                max_iter=args.max_iter,
            )
            row.update({"case": args.case, "alpha": args.alpha, "eps_alpha": eps, "perturbation": sign_name, "alpha_perturbed": args.alpha + sign * eps})
            solve_rows.append(row)
            phi[sign_name] = np.asarray(row["tip_p"], dtype=float)
            ys[sign_name] = y
            out = p2c25._physical_metrics(funcs, y, I, args.boundary_scale, args.slack_scale)["out"]
            centers[sign_name] = np.asarray(out["centerline"], dtype=float)
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": f"{sign_name}_done", "eps": eps, "stationarity_norm": row["stationarity_norm"], "static_gate": row["static_gate"]})
        out_base = p2c25._physical_metrics(funcs, y_base, I0, args.boundary_scale, args.slack_scale)["out"]
        centers["base"] = np.asarray(out_base["centerline"], dtype=float)
        sensitivity = (phi["plus"] - phi["minus"]) / (2.0 * eps)
        pred_plus = phi0 + sensitivity * eps
        pred_minus = phi0 - sensitivity * eps
        scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
        lin_plus = float(np.linalg.norm(phi["plus"] - pred_plus) / scale)
        lin_minus = float(np.linalg.norm(phi["minus"] - pred_minus) / scale)
        lin_sym = float(np.linalg.norm((phi["plus"] + phi["minus"]) * 0.5 - phi0) / scale)
        branch_rms = []
        branch_tip = []
        for a, b in [("base", "minus"), ("base", "plus"), ("minus", "plus")]:
            branch_rms.append(float(np.sqrt(np.mean(np.sum((centers[a] - centers[b]) ** 2, axis=1)))))
            branch_tip.append(float(np.linalg.norm(centers[a][-1] - centers[b][-1])))
        row = {
            "case": args.case,
            "alpha": args.alpha,
            "eps_alpha": eps,
            "publishable_gate": bool(
                baseline["static_gate"]
                and all(r["static_gate"] for r in solve_rows if r["eps_alpha"] == eps)
                and max(lin_plus, lin_minus, lin_sym) <= 0.05
                and max(branch_rms) <= 1.0e-3
            ),
            "baseline_stationarity_norm": baseline["stationarity_norm"],
            "baseline_jacobian_condition": baseline["jacobian_condition"],
            "baseline_slack_force_ratio": baseline["max_slack_force_over_rigid_force"],
            "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sensitivity)),
            "sensitivity_vector": sensitivity.tolist(),
            "linearization_error_plus": lin_plus,
            "linearization_error_minus": lin_minus,
            "linearization_error_symmetry": lin_sym,
            "linearization_error_max": max(lin_plus, lin_minus, lin_sym),
            "max_perturb_stationarity": max(r["stationarity_norm"] for r in solve_rows if r["eps_alpha"] == eps),
            "max_perturb_slack_force_ratio": max(r["max_slack_force_over_rigid_force"] for r in solve_rows if r["eps_alpha"] == eps),
            "max_perturb_jacobian_condition": max(r["jacobian_condition"] for r in solve_rows if r["eps_alpha"] == eps),
            "branch_rms_max": max(branch_rms),
            "branch_tip_distance_max": max(branch_tip),
        }
        eps_rows.append(row)
        np.savez_compressed(
            OUT_DIR / f"{args.case}_eps_{eps:.4f}_reoptimized_sensitivity.npz",
            phi0=phi0,
            phi_plus=phi["plus"],
            phi_minus=phi["minus"],
            sensitivity=sensitivity,
            centerline_base=centers["base"],
            centerline_plus=centers["plus"],
            centerline_minus=centers["minus"],
            y_base=y_base,
            y_plus=ys["plus"],
            y_minus=ys["minus"],
        )
        if not row["publishable_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "reoptimized_sensitivity_gate_failed", "case": args.case, "alpha": args.alpha, "eps_alpha": eps, "evidence": row})

    _write_csv(OUT_DIR / "eps_summary.csv", eps_rows)
    _write_csv(OUT_DIR / "solve_rows.csv", solve_rows)
    _plot_eps(eps_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "baseline": baseline,
        "overall": {
            "eps_count": len(eps_rows),
            "publishable_count": sum(r["publishable_gate"] for r in eps_rows),
            "diagnostic_count": sum(not r["publishable_gate"] for r in eps_rows),
            "max_linearization_error": max(r["linearization_error_max"] for r in eps_rows),
            "max_branch_rms": max(r["branch_rms_max"] for r in eps_rows),
            "max_perturb_stationarity": max(r["max_perturb_stationarity"] for r in eps_rows),
        },
        "eps_summaries": eps_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
