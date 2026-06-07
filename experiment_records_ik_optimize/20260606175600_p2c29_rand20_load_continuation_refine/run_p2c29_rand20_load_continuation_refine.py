"""Test load-continuation refinement for the n3 mixed random-current case."""

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
P2C27_DIR = ROOT / "experiment_records_ik_optimize" / "20260606172200_p2c27_stationarity_refinement_gate"
for path in (ROOT, P2C24_DIR, P2C25_DIR, P2C27_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c25_scaled_stationarity_and_interface_audit as p2c25
import run_p2c27_stationarity_refinement_gate as p2c27


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


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _zero_seed_like(y: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(y, dtype=float).reshape(-1))


def _compete(
    *,
    label: str,
    alpha: float,
    funcs: dict[str, Any],
    y0: np.ndarray,
    I: np.ndarray,
    target_p: np.ndarray,
    params: Any,
    args: argparse.Namespace,
    heartbeat: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    all_rows: list[dict[str, Any]] = []
    best_row = None
    best_y = None
    for strategy in strategies:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_start", "label": label, "alpha": alpha, "strategy": strategy})
        row, y, inner = p2c27._solve_strategy(
            strategy=strategy,
            funcs=funcs,
            y0=y0,
            I=I,
            target_p=target_p,
            params=params,
            boundary_scale=args.boundary_scale,
            slack_scale=args.slack_scale,
            max_iter=args.max_iter,
            restarts=args.restarts,
            tol=args.tol,
        )
        row.update({"label": label, "alpha": alpha, "strategy": strategy})
        all_rows.append(row)
        if best_row is None or (not row["static_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]) < (not best_row["static_gate"], best_row["stationarity_norm"], best_row["max_slack_force_over_rigid_force"]):
            best_row = row
            best_y = y
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_done", "label": label, "alpha": alpha, "strategy": strategy, "stationarity_norm": row["stationarity_norm"], "static_gate": row["static_gate"]})
    return best_row, best_y, all_rows


def _run_path(
    *,
    label: str,
    y_start: np.ndarray,
    alpha_path: list[float],
    base_currents: np.ndarray,
    funcs: dict[str, Any],
    target_p: np.ndarray,
    params: Any,
    args: argparse.Namespace,
    heartbeat: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray(y_start, dtype=float).reshape(-1)
    best_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for alpha in alpha_path:
        I = base_currents * float(alpha)
        best, y, rows = _compete(
            label=label,
            alpha=alpha,
            funcs=funcs,
            y0=y,
            I=I,
            target_p=target_p,
            params=params,
            args=args,
            heartbeat=heartbeat,
        )
        best_rows.append(best)
        strategy_rows.extend(rows)
    return best_rows[-1], y, best_rows, strategy_rows


def _sensitivity_gate(
    *,
    y_base: np.ndarray,
    alpha: float,
    eps: float,
    base_currents: np.ndarray,
    funcs: dict[str, Any],
    target_p: np.ndarray,
    params: Any,
    args: argparse.Namespace,
    heartbeat: Path,
) -> dict[str, Any]:
    I0 = base_currents * float(alpha)
    base_metrics = p2c25._physical_metrics(funcs, y_base, I0, args.boundary_scale, args.slack_scale)
    phi0 = np.asarray(base_metrics["out"]["tip_p"], dtype=float)
    phis: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {"base": np.asarray(base_metrics["out"]["centerline"], dtype=float)}
    selected: dict[str, dict[str, Any]] = {}
    for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
        I = base_currents * float(alpha + sign * eps)
        row, y, _rows = _compete(
            label=f"sensitivity_{sign_name}",
            alpha=alpha + sign * eps,
            funcs=funcs,
            y0=y_base,
            I=I,
            target_p=target_p,
            params=params,
            args=args,
            heartbeat=heartbeat,
        )
        selected[sign_name] = row
        metrics = p2c25._physical_metrics(funcs, y, I, args.boundary_scale, args.slack_scale)
        phis[sign_name] = np.asarray(metrics["out"]["tip_p"], dtype=float)
        centers[sign_name] = np.asarray(metrics["out"]["centerline"], dtype=float)
    sensitivity = (phis["plus"] - phis["minus"]) / (2.0 * eps)
    scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
    lin_plus = float(np.linalg.norm(phis["plus"] - (phi0 + sensitivity * eps)) / scale)
    lin_minus = float(np.linalg.norm(phis["minus"] - (phi0 - sensitivity * eps)) / scale)
    lin_sym = float(np.linalg.norm((phis["plus"] + phis["minus"]) * 0.5 - phi0) / scale)
    branch_rms = []
    branch_tip = []
    for a, b in [("base", "minus"), ("base", "plus"), ("minus", "plus")]:
        branch_rms.append(float(np.sqrt(np.mean(np.sum((centers[a] - centers[b]) ** 2, axis=1)))))
        branch_tip.append(float(np.linalg.norm(centers[a][-1] - centers[b][-1])))
    return {
        "eps_alpha": eps,
        "publishable_gate": bool(
            selected["minus"]["static_gate"]
            and selected["plus"]["static_gate"]
            and max(lin_plus, lin_minus, lin_sym) <= 0.05
            and max(branch_rms) <= 1.0e-3
        ),
        "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sensitivity)),
        "sensitivity_vector": sensitivity.tolist(),
        "linearization_error_max": max(lin_plus, lin_minus, lin_sym),
        "max_perturb_stationarity": max(selected["minus"]["stationarity_norm"], selected["plus"]["stationarity_norm"]),
        "max_perturb_slack_force_ratio": max(selected["minus"]["max_slack_force_over_rigid_force"], selected["plus"]["max_slack_force_over_rigid_force"]),
        "branch_rms_max": max(branch_rms),
        "branch_tip_distance_max": max(branch_tip),
    }


def _plot_path(path_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = [f"{r['label']}\n{r['alpha']:.2f}" for r in path_rows]
    x = np.arange(len(labels))
    specs = [
        ("stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
        ("position_error_m", "末端位置误差 m", 0.05 * 0.11),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [r[key] for r in path_rows], color=["green" if r["static_gate"] else "orange" for r in path_rows])
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "load_continuation_path_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="n3_mixed_rand20")
    parser.add_argument("--target_alpha", type=float, default=0.4)
    parser.add_argument("--alpha_path", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--eps_alpha", type=float, default=0.01)
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

    params, _I_target = p2c24._params_from_saved_case(args.case, args.target_alpha)
    target_p, y_saved = p2c24._load_saved_target_and_seed(args.case, args.target_alpha)
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    funcs = p2c24._make_functions(params, target_p, target_q, args)
    base_currents = np.asarray(p2c24._case_spec(args.case).currents, dtype=float)
    alpha_path = _parse_floats(args.alpha_path)
    y_saved = np.asarray(y_saved, dtype=float)
    seeds = {
        "direct_saved": y_saved,
        "direct_zero": _zero_seed_like(y_saved),
        "homotopy_saved": y_saved,
        "homotopy_zero": _zero_seed_like(y_saved),
    }
    direct_alpha = [args.target_alpha]

    final_rows: list[dict[str, Any]] = []
    path_rows_all: list[dict[str, Any]] = []
    strategy_rows_all: list[dict[str, Any]] = []
    final_y: dict[str, np.ndarray] = {}
    for label, seed in seeds.items():
        path = alpha_path if label.startswith("homotopy") else direct_alpha
        best, y, path_rows, strategy_rows = _run_path(
            label=label,
            y_start=seed,
            alpha_path=path,
            base_currents=base_currents,
            funcs=funcs,
            target_p=target_p,
            params=params,
            args=args,
            heartbeat=heartbeat,
        )
        final = dict(best)
        final["candidate"] = label
        final_rows.append(final)
        final_y[label] = y
        path_rows_all.extend(path_rows)
        strategy_rows_all.extend(strategy_rows)
    best_final = min(final_rows, key=lambda r: (not r["static_gate"], r["stationarity_norm"], r["max_slack_force_over_rigid_force"]))
    sensitivity = _sensitivity_gate(
        y_base=final_y[best_final["candidate"]],
        alpha=args.target_alpha,
        eps=args.eps_alpha,
        base_currents=base_currents,
        funcs=funcs,
        target_p=target_p,
        params=params,
        args=args,
        heartbeat=heartbeat,
    )
    publishable = bool(best_final["static_gate"] and sensitivity["publishable_gate"])
    if not publishable:
        modes = []
        if best_final["stationarity_norm"] > 1.0e-5:
            modes.append("基线一阶最优性退化")
        if best_final["max_slack_force_over_rigid_force"] > 2.0:
            modes.append("刚性磁体段载荷传播失配")
        if sensitivity["max_perturb_stationarity"] > 1.0e-5:
            modes.append("扰动重求解一阶最优性退化")
        if sensitivity["linearization_error_max"] > 0.05:
            modes.append("局部不可线性化")
        _append_jsonl(failures, {"ts": time.time(), "failure": "load_continuation_not_publishable", "case": args.case, "alpha": args.target_alpha, "modes": modes or ["证据不足"], "best_final": best_final, "sensitivity": sensitivity})
    _write_csv(OUT_DIR / "final_candidates.csv", final_rows)
    _write_csv(OUT_DIR / "path_rows.csv", path_rows_all)
    _write_csv(OUT_DIR / "strategy_rows.csv", strategy_rows_all)
    _plot_path(path_rows_all)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "best_final": best_final,
        "sensitivity": sensitivity,
        "publishable_gate": publishable,
        "final_candidates": final_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable({"publishable_gate": publishable, "best_candidate": best_final["candidate"], "best_stationarity_norm": best_final["stationarity_norm"], "sensitivity_gate": sensitivity["publishable_gate"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
