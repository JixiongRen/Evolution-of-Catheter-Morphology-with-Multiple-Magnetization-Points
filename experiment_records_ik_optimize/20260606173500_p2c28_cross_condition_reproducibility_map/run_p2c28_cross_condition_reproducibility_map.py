"""Map cross-condition publishability with the refined shooting gate."""

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
P2C25_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170041_p2c25_scaled_stationarity_and_interface_audit"
P2C26_DIR = ROOT / "experiment_records_ik_optimize" / "20260606170930_p2c26_reoptimized_sensitivity_gate"
P2C27_DIR = ROOT / "experiment_records_ik_optimize" / "20260606172200_p2c27_stationarity_refinement_gate"
for path in (ROOT, P2C24_DIR, P2C25_DIR, P2C26_DIR, P2C27_DIR):
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
import run_p2c26_reoptimized_sensitivity_gate as p2c26
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


def _parse_cases(text: str) -> list[tuple[str, float, float, str]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        case, alpha, eta, repeat = raw.split(":", 3)
        specs.append((case.strip(), float(alpha), float(eta), repeat.strip()))
    return specs


def _parse_eps(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _failure_modes(summary: dict[str, Any]) -> list[str]:
    modes = []
    if not summary["baseline_static_gate"]:
        if summary["baseline_stationarity_norm"] > 1.0e-5:
            modes.append("一阶最优性退化")
        if summary["baseline_jacobian_condition"] > 1.0e10:
            modes.append("数值病态或局部不可辨识")
        if summary["baseline_slack_force_ratio"] > 2.0:
            modes.append("刚性磁体段载荷传播失配")
        if summary["baseline_position_error_m"] > 0.05 * 0.11:
            modes.append("可达域或末端约束失配")
    if summary["max_perturb_stationarity"] > 1.0e-5:
        modes.append("扰动重求解一阶最优性退化")
    if summary["max_linearization_error"] > 0.05:
        modes.append("局部不可线性化")
    if summary["branch_rms_max"] > 1.0e-3:
        modes.append("分支切换或多解竞争")
    if summary["max_perturb_slack_force_ratio"] > 2.0:
        modes.append("扰动后刚性磁体段载荷传播失配")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    priority = [
        "刚性磁体段载荷传播失配",
        "扰动后刚性磁体段载荷传播失配",
        "一阶最优性退化",
        "扰动重求解一阶最优性退化",
        "数值病态或局部不可辨识",
        "局部不可线性化",
        "分支切换或多解竞争",
        "可达域或末端约束失配",
    ]
    for item in priority:
        if item in modes:
            return item
    return modes[0]


def _magnet_count(case_name: str) -> int:
    case = p2c24._case_spec(case_name)
    return sum(float(np.linalg.norm(m)) > 0.0 for m in case.m_body_list)


def _run_case(
    *,
    case_name: str,
    alpha: float,
    eta: float,
    repeat: str,
    eps_values: list[float],
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha, "eta": eta, "repeat": repeat})
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_saved = p2c24._load_saved_target_and_seed(case_name, alpha)
    target_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    funcs = p2c24._make_functions(params, target_p, target_q, args)
    y_init = p2c24._seed_from_reference(y_saved, eta=eta, base_scale=1.0)
    base_currents = np.asarray(p2c24._case_spec(case_name).currents, dtype=float)
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

    for eps in eps_values:
        selected: dict[str, dict[str, Any]] = {}
        centers: dict[str, np.ndarray] = {}
        phis: dict[str, np.ndarray] = {}
        for sign_name, sign in [("minus", -1.0), ("plus", 1.0)]:
            I = I0 + sign * eps * base_currents
            best_row = None
            best_y = None
            for strategy in strategies:
                _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_start", "case": case_name, "repeat": repeat, "eps": eps, "perturbation": sign_name, "strategy": strategy})
                row, y, _inner = p2c27._solve_strategy(
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
                row.update({"case": case_name, "alpha": alpha, "eta": eta, "repeat": repeat, "eps_alpha": eps, "perturbation": sign_name, "alpha_perturbed": alpha + sign * eps})
                solve_rows.append(row)
                if best_row is None or (not row["static_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]) < (not best_row["static_gate"], best_row["stationarity_norm"], best_row["max_slack_force_over_rigid_force"]):
                    best_row = row
                    best_y = y
                _append_jsonl(heartbeat, {"ts": time.time(), "phase": "strategy_done", "case": case_name, "repeat": repeat, "eps": eps, "perturbation": sign_name, "strategy": strategy, "stationarity_norm": row["stationarity_norm"], "static_gate": row["static_gate"]})
            selected[sign_name] = best_row
            phis[sign_name] = np.asarray(best_row["tip_p"], dtype=float)
            metrics = p2c25._physical_metrics(funcs, best_y, I, args.boundary_scale, args.slack_scale)
            centers[sign_name] = np.asarray(metrics["out"]["centerline"], dtype=float)
        base_metrics = p2c25._physical_metrics(funcs, y_base, I0, args.boundary_scale, args.slack_scale)
        centers["base"] = np.asarray(base_metrics["out"]["centerline"], dtype=float)
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
        eps_row = {
            "case": case_name,
            "alpha": alpha,
            "eta": eta,
            "repeat": repeat,
            "magnet_count": _magnet_count(case_name),
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
            "linearization_error_max": max(lin_plus, lin_minus, lin_sym),
            "max_perturb_stationarity": max(selected["minus"]["stationarity_norm"], selected["plus"]["stationarity_norm"]),
            "max_perturb_slack_force_ratio": max(selected["minus"]["max_slack_force_over_rigid_force"], selected["plus"]["max_slack_force_over_rigid_force"]),
            "max_perturb_jacobian_condition": max(selected["minus"]["jacobian_condition"], selected["plus"]["jacobian_condition"]),
            "branch_rms_max": max(branch_rms),
            "branch_tip_distance_max": max(branch_tip),
        }
        eps_rows.append(eps_row)

    summary = {
        "case": case_name,
        "alpha": alpha,
        "eta": eta,
        "repeat": repeat,
        "magnet_count": _magnet_count(case_name),
        "publishable_gate": all(r["publishable_gate"] for r in eps_rows),
        "baseline_static_gate": baseline["static_gate"],
        "baseline_position_error_m": baseline["position_error_m"],
        "baseline_stationarity_norm": baseline["stationarity_norm"],
        "baseline_jacobian_condition": baseline["jacobian_condition"],
        "baseline_slack_force_ratio": baseline["max_slack_force_over_rigid_force"],
        "max_linearization_error": max(r["linearization_error_max"] for r in eps_rows),
        "branch_rms_max": max(r["branch_rms_max"] for r in eps_rows),
        "max_perturb_stationarity": max(r["max_perturb_stationarity"] for r in eps_rows),
        "max_perturb_slack_force_ratio": max(r["max_perturb_slack_force_ratio"] for r in eps_rows),
        "sensitivity_norm_mean": float(np.mean([r["sensitivity_norm_m_per_alpha"] for r in eps_rows])),
        "sensitivity_norm_std": float(np.std([r["sensitivity_norm_m_per_alpha"] for r in eps_rows])),
        "total_solve_s": sum(r["total_steady_s"] for r in solve_rows),
    }
    modes = [] if summary["publishable_gate"] else _failure_modes(summary)
    summary["failure_modes"] = "" if summary["publishable_gate"] else ";".join(modes)
    summary["primary_failure_mode"] = "" if summary["publishable_gate"] else _primary_mode(modes)
    if not summary["publishable_gate"]:
        _append_jsonl(failures, {"ts": time.time(), "failure": "cross_condition_gate_failed", "case": case_name, "alpha": alpha, "repeat": repeat, "evidence": summary})
    return summary, eps_rows, solve_rows


def _plot_phase(summaries: list[dict[str, Any]]) -> None:
    labels = [f"{s['case']}\n{s['repeat']}" for s in summaries]
    x = np.arange(len(labels))
    colors = ["green" if s["publishable_gate"] else "orange" for s in summaries]
    fig, axes = plt.subplots(4, 1, figsize=(max(10, 1.2 * len(labels)), 11), sharex=True)
    specs = [
        ("baseline_stationarity_norm", "基线一阶最优性", 1.0e-5),
        ("max_perturb_stationarity", "扰动后一阶最优性", 1.0e-5),
        ("max_linearization_error", "最大线性化误差", 0.05),
        ("max_perturb_slack_force_ratio", "最大界面力松弛/刚段外力", 2.0),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [s[key] for s in summaries], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_gate_phase_map.png", dpi=180)
    plt.close(fig)


def _plot_sensitivity(eps_rows: list[dict[str, Any]]) -> None:
    publish = [r for r in eps_rows if r["publishable_gate"]]
    diag = [r for r in eps_rows if not r["publishable_gate"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["eps_alpha"] for r in publish], [r["sensitivity_norm_m_per_alpha"] for r in publish], c="green", label="可发布")
    ax.scatter([r["eps_alpha"] for r in diag], [r["sensitivity_norm_m_per_alpha"] for r in diag], c="orange", label="诊断")
    ax.set_xlabel("alpha 扰动幅值")
    ax.set_ylabel("敏感度范数 m/alpha")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=(
            "n3_mixed_base_i4:1.00:0.75:rep_a,"
            "n3_mixed_base_i4:1.00:0.75:rep_b,"
            "n1_tip_z_rand15:0.10:1.00:diag_n1,"
            "n2_z_pair_rand12:0.10:1.00:diag_n2,"
            "n3_mixed_rand20:0.40:1.00:diag_rand20"
        ),
    )
    parser.add_argument("--eps_values", default="0.01")
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
    summaries: list[dict[str, Any]] = []
    eps_rows_all: list[dict[str, Any]] = []
    solve_rows_all: list[dict[str, Any]] = []
    eps_values = _parse_eps(args.eps_values)

    for case_name, alpha, eta, repeat in _parse_cases(args.cases):
        summary, eps_rows, solve_rows = _run_case(
            case_name=case_name,
            alpha=alpha,
            eta=eta,
            repeat=repeat,
            eps_values=eps_values,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        summaries.append(summary)
        eps_rows_all.extend(eps_rows)
        solve_rows_all.extend(solve_rows)

    _write_csv(OUT_DIR / "case_summary.csv", summaries)
    _write_csv(OUT_DIR / "eps_rows.csv", eps_rows_all)
    _write_csv(OUT_DIR / "solve_rows.csv", solve_rows_all)
    _plot_phase(summaries)
    _plot_sensitivity(eps_rows_all)
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
            "publishable_case_count": sum(s["publishable_gate"] for s in summaries),
            "diagnostic_case_count": sum(not s["publishable_gate"] for s in summaries),
            "failure_mode_count": dict(counter),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
