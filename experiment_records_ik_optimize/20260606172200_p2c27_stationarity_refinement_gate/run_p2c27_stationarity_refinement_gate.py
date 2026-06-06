"""Refine perturbation solves until physical stationarity gates are met."""

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
P2C26_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170930_p2c26_reoptimized_sensitivity_gate"
for path in (ROOT, P2C24_DIR, P2C25_DIR, P2C26_DIR):
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
import run_p2c26_reoptimized_sensitivity_gate as p2c26


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


def _physical_row(
    *,
    funcs: dict[str, Any],
    y: np.ndarray,
    I: np.ndarray,
    target_p: np.ndarray,
    params: Any,
    boundary_scale: float,
    slack_scale: float,
    method: str,
    message: str,
    nfev: int,
    steady_s: float,
    success: bool,
) -> dict[str, Any]:
    metrics = p2c25._physical_metrics(funcs, y, I, boundary_scale, slack_scale)
    load = p2c24._audit_load(params, I, y)
    pos_err = float(np.linalg.norm(np.asarray(metrics["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "method": method,
        "success": bool(success),
        "nfev": int(nfev),
        "steady_s": float(steady_s),
        "message": message,
        "position_error_m": pos_err,
        "residual_norm": metrics["residual_norm"],
        "stationarity_norm": metrics["stationarity_norm"],
        "boundary_norm": metrics["boundary_norm"],
        "jacobian_condition": metrics["jacobian_condition"],
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
    return row


def _solve_scaled_once(
    *,
    funcs: dict[str, Any],
    y_anchor: np.ndarray,
    I: np.ndarray,
    target_p: np.ndarray,
    params: Any,
    boundary_scale: float,
    slack_scale: float,
    max_iter: int,
    method: str,
    tol: float,
    label: str,
) -> tuple[dict[str, Any], np.ndarray]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y_anchor = np.asarray(y_anchor, dtype=float).reshape(-1)
    scale = p2c25._column_scale(funcs, y_anchor, I, boundary_scale, slack_scale)
    u0 = np.zeros_like(y_anchor)

    def residual_u(u: np.ndarray) -> np.ndarray:
        y = y_anchor + scale * np.asarray(u, dtype=float)
        return np.asarray(jax.device_get(funcs["residual_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)

    def jac_u(u: np.ndarray) -> np.ndarray:
        y = y_anchor + scale * np.asarray(u, dtype=float)
        J = np.asarray(jax.device_get(funcs["Jy_jit"](jnp.asarray(y), I_j, b_j, s_j)), dtype=float)
        return J * scale.reshape(1, -1)

    kwargs: dict[str, Any] = {
        "method": method,
        "x_scale": np.ones_like(u0),
        "max_nfev": int(max_iter),
        "ftol": float(tol),
        "xtol": float(tol),
        "gtol": float(tol),
    }
    if method != "lm":
        kwargs["loss"] = "linear"
    t0 = time.perf_counter()
    opt = p2c24.p2c10w.least_squares(residual_u, u0, jac=jac_u, **kwargs)
    steady_s = time.perf_counter() - t0
    y = y_anchor + scale * np.asarray(opt.x, dtype=float)
    row = _physical_row(
        funcs=funcs,
        y=y,
        I=I,
        target_p=target_p,
        params=params,
        boundary_scale=boundary_scale,
        slack_scale=slack_scale,
        method=label,
        message=str(opt.message),
        nfev=int(opt.nfev),
        steady_s=float(steady_s),
        success=bool(opt.success),
    )
    return row, y


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
    restarts: int,
    tol: float,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    y = np.asarray(y0, dtype=float).reshape(-1)
    if strategy == "trf_strict_restart":
        for k in range(restarts):
            row, y = _solve_scaled_once(
                funcs=funcs,
                y_anchor=y,
                I=I,
                target_p=target_p,
                params=params,
                boundary_scale=boundary_scale,
                slack_scale=slack_scale,
                max_iter=max_iter,
                method="trf",
                tol=tol,
                label=f"{strategy}_r{k}",
            )
            rows.append(row)
            if row["static_gate"]:
                break
    elif strategy == "lm_refine":
        row, y = _solve_scaled_once(
            funcs=funcs,
            y_anchor=y,
            I=I,
            target_p=target_p,
            params=params,
            boundary_scale=boundary_scale,
            slack_scale=slack_scale,
            max_iter=max_iter * max(1, restarts),
            method="lm",
            tol=tol,
            label=strategy,
        )
        rows.append(row)
    elif strategy == "trf_then_lm":
        row, y = _solve_scaled_once(
            funcs=funcs,
            y_anchor=y,
            I=I,
            target_p=target_p,
            params=params,
            boundary_scale=boundary_scale,
            slack_scale=slack_scale,
            max_iter=max_iter,
            method="trf",
            tol=tol,
            label=f"{strategy}_trf",
        )
        rows.append(row)
        row, y = _solve_scaled_once(
            funcs=funcs,
            y_anchor=y,
            I=I,
            target_p=target_p,
            params=params,
            boundary_scale=boundary_scale,
            slack_scale=slack_scale,
            max_iter=max_iter * max(1, restarts),
            method="lm",
            tol=tol,
            label=f"{strategy}_lm",
        )
        rows.append(row)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    best = min(rows, key=lambda r: (not r["static_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"]))
    best = dict(best)
    best["strategy"] = strategy
    best["restart_count"] = len(rows)
    best["total_nfev"] = sum(r["nfev"] for r in rows)
    best["total_steady_s"] = sum(r["steady_s"] for r in rows)
    return best, y, rows


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['eps_alpha']}\n{r['perturbation']}" for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.9 * len(labels)), 9), sharex=True)
    specs = [
        ("stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
        ("jacobian_condition", "物理雅可比条件数", 1.0e10),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [r[key] for r in rows], color=["green" if r["static_gate"] else "orange" for r in rows])
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "perturbation_refinement_gate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_base_i4")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.75)
    parser.add_argument("--eps_values", default="0.005,0.01,0.02")
    parser.add_argument("--strategies", default="trf_strict_restart,lm_refine,trf_then_lm")
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1.0e-12)
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
    y_init = p2c24._seed_from_reference(y_saved, eta=args.eta, base_scale=1.0)
    base_currents = np.asarray(p2c24._case_spec(args.case).currents, dtype=float)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    baseline, y_base = p2c26._solve_jac_column(
        funcs=funcs,
        y0=y_init,
        I=I0,
        target_p=target_p,
        params=params,
        boundary_scale=args.boundary_scale,
        slack_scale=args.slack_scale,
        max_iter=args.max_iter,
    )
    phi0 = np.asarray(baseline["tip_p"], dtype=float)

    solve_rows: list[dict[str, Any]] = []
    eps_rows: list[dict[str, Any]] = []
    for eps in _parse_eps(args.eps_values):
        selected: dict[str, dict[str, Any]] = {}
        selected_y: dict[str, np.ndarray] = {"base": y_base}
        centers: dict[str, np.ndarray] = {}
        for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
            I = I0 + sign * eps * base_currents
            best_row = None
            best_y = None
            for strategy in strategies:
                _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_start", "eps": eps, "perturbation": sign_name, "strategy": strategy})
                row, y, inner = _solve_strategy(
                    strategy=strategy,
                    funcs=funcs,
                    y0=y_base,
                    I=I,
                    target_p=target_p,
                    params=params,
                    boundary_scale=args.boundary_scale,
                    slack_scale=args.slack_scale,
                    max_iter=args.max_iter,
                    restarts=args.restarts,
                    tol=args.tol,
                )
                row.update({"case": args.case, "alpha": args.alpha, "eps_alpha": eps, "perturbation": sign_name, "alpha_perturbed": args.alpha + sign * eps})
                solve_rows.append(row)
                if best_row is None or (not row["static_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]) < (not best_row["static_gate"], best_row["stationarity_norm"], best_row["max_slack_force_over_rigid_force"]):
                    best_row = row
                    best_y = y
                _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_done", "eps": eps, "perturbation": sign_name, "strategy": strategy, "stationarity_norm": row["stationarity_norm"], "static_gate": row["static_gate"]})
            selected[sign_name] = best_row
            selected_y[sign_name] = best_y
            metrics = p2c25._physical_metrics(funcs, best_y, I, args.boundary_scale, args.slack_scale)
            centers[sign_name] = np.asarray(metrics["out"]["centerline"], dtype=float)
        base_metrics = p2c25._physical_metrics(funcs, y_base, I0, args.boundary_scale, args.slack_scale)
        centers["base"] = np.asarray(base_metrics["out"]["centerline"], dtype=float)
        phi_minus = np.asarray(selected["minus"]["tip_p"], dtype=float)
        phi_plus = np.asarray(selected["plus"]["tip_p"], dtype=float)
        sensitivity = (phi_plus - phi_minus) / (2.0 * eps)
        scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
        lin_plus = float(np.linalg.norm(phi_plus - (phi0 + sensitivity * eps)) / scale)
        lin_minus = float(np.linalg.norm(phi_minus - (phi0 - sensitivity * eps)) / scale)
        lin_sym = float(np.linalg.norm((phi_plus + phi_minus) * 0.5 - phi0) / scale)
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
                and selected["minus"]["static_gate"]
                and selected["plus"]["static_gate"]
                and max(lin_plus, lin_minus, lin_sym) <= 0.05
                and max(branch_rms) <= 1.0e-3
            ),
            "minus_strategy": selected["minus"]["strategy"],
            "plus_strategy": selected["plus"]["strategy"],
            "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sensitivity)),
            "sensitivity_vector": sensitivity.tolist(),
            "linearization_error_plus": lin_plus,
            "linearization_error_minus": lin_minus,
            "linearization_error_symmetry": lin_sym,
            "linearization_error_max": max(lin_plus, lin_minus, lin_sym),
            "max_perturb_stationarity": max(selected["minus"]["stationarity_norm"], selected["plus"]["stationarity_norm"]),
            "max_perturb_slack_force_ratio": max(selected["minus"]["max_slack_force_over_rigid_force"], selected["plus"]["max_slack_force_over_rigid_force"]),
            "max_perturb_jacobian_condition": max(selected["minus"]["jacobian_condition"], selected["plus"]["jacobian_condition"]),
            "branch_rms_max": max(branch_rms),
            "branch_tip_distance_max": max(branch_tip),
        }
        eps_rows.append(row)
        np.savez_compressed(
            OUT_DIR / f"{args.case}_eps_{eps:.4f}_refined_sensitivity.npz",
            phi0=phi0,
            phi_plus=phi_plus,
            phi_minus=phi_minus,
            sensitivity=sensitivity,
            centerline_base=centers["base"],
            centerline_plus=centers["plus"],
            centerline_minus=centers["minus"],
            y_base=y_base,
            y_plus=selected_y["plus"],
            y_minus=selected_y["minus"],
        )
        if not row["publishable_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "stationarity_refinement_gate_failed", "case": args.case, "alpha": args.alpha, "eps_alpha": eps, "evidence": row})

    _write_csv(OUT_DIR / "solve_rows.csv", solve_rows)
    _write_csv(OUT_DIR / "eps_summary.csv", eps_rows)
    _plot([r for r in solve_rows if r["strategy"] in {e["minus_strategy"] for e in eps_rows} | {e["plus_strategy"] for e in eps_rows}])
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
