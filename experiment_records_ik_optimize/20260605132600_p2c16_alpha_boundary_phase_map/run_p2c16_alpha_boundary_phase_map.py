"""Map the local sensitivity boundary around a slack-suppressed candidate."""

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
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
P2C15_DIR = ROOT / "experiment_records_ik_optimize" / "20260605125315_p2c15_slack_suppressed_sensitivity_audit"
for path in (ROOT, P2C10W_DIR, P2C10X_DIR, P2C15_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
import run_p2c10w_task_reduction_phase_map as p2c10w
import run_p2c10x_position_alpha_linearity_audit as p2c10x
import run_p2c15_slack_suppressed_sensitivity_audit as p2c15

configure_matplotlib_cjk()
jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
SEED_ALPHAS = np.asarray([0.30, 0.35, 0.40, 0.45, 0.50], dtype=float)


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


def _nearest_seed_alpha(alpha: float) -> float:
    return float(SEED_ALPHAS[int(np.argmin(np.abs(SEED_ALPHAS - alpha)))])


def _load_seed_y(alpha: float) -> np.ndarray:
    return p2c15._load_seed_y(_nearest_seed_alpha(alpha))


def _load_score(row: dict[str, Any]) -> tuple[bool, bool, float, float, float]:
    task_gate = row["position_error_m"] <= 0.05 * 0.11
    load_gate = row["max_slack_force_over_rigid_force"] <= 2.0
    return (
        not bool(row["success"]),
        not bool(task_gate),
        not bool(load_gate),
        float(row["stationarity_norm"]),
        float(row["boundary_norm"]),
    )


def _classify(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not summary["baseline_success"]:
        reasons.append("基线求解器未成功")
    if summary["max_position_error_m"] > 0.05 * 0.11:
        reasons.append("末端位置误差超过门限")
    if summary["max_stage_stationarity"] > 1.0e-5:
        reasons.append("一阶平稳性残差超过门限")
    if summary["max_stage_condition"] > 1.0e10:
        reasons.append("局部条件数超过门限")
    if summary["max_linearization_error"] > 0.05:
        reasons.append("局部线性化误差超过门限")
    if summary["branch_rms_max"] > 1.0e-3:
        reasons.append("中心线分支漂移超过门限")
    if summary["max_slack_force_over_rigid_force"] > 2.0:
        reasons.append("界面力松弛与刚段外力比例超过门限")
    if not summary["independent_seed_gate"]:
        reasons.append("最近历史种子不能独立复现可信基线，结果存在延拓路径依赖")
    if not reasons:
        return "可信工况", []
    severe = any(
        text in reasons
        for text in [
            "一阶平稳性残差超过门限",
            "局部线性化误差超过门限",
            "中心线分支漂移超过门限",
            "界面力松弛与刚段外力比例超过门限",
        ]
    )
    return ("不可用工况" if severe else "临界工况"), reasons


def _select_baseline(
    *,
    alpha: float,
    params: Any,
    I0: np.ndarray,
    funcs: dict[str, Any],
    target_p: np.ndarray,
    args: argparse.Namespace,
    previous_y: np.ndarray | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[tuple[str, np.ndarray]] = [(f"nearest_seed_{_nearest_seed_alpha(alpha):.2f}", _load_seed_y(alpha))]
    if previous_y is not None:
        candidates.append(("continuation_previous_alpha", np.asarray(previous_y, dtype=float)))
    solved: list[tuple[dict[str, Any], dict[str, float]]] = []
    rows: list[dict[str, Any]] = []
    for name, y0 in candidates:
        solve = p2c15._solve(
            stage=f"base_{name}",
            y0=y0,
            I=I0,
            funcs=funcs,
            target_p=target_p,
            slack_scale=args.slack_scale,
            max_iter=args.max_iter,
        )
        load = p2c15._audit_load(params, I0, solve["y"])
        row = {
            "alpha": alpha,
            "init_source": name,
            "stage": "baseline_candidate",
            **{key: value for key, value in solve.items() if key not in {"y", "out", "phi"}},
            **load,
        }
        rows.append(row)
        solved.append((solve, load))
    scored = sorted(zip(rows, solved, strict=True), key=lambda item: _load_score(item[0]))
    best_row, (best_solve, best_load) = scored[0]
    best_solve["selected_init_source"] = best_row["init_source"]
    best_solve["selected_load"] = best_load
    return best_solve, rows


def _run_alpha(
    *,
    alpha: float,
    previous_y: np.ndarray | None,
    args: argparse.Namespace,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "alpha_start", "alpha": alpha})
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    I_base = p2c10x._base_currents("n3_mixed_rand20")
    funcs = p2c15._make_functions(params, target_p, args.position_scale, args.boundary_scale, args.steps_per_flex)
    base, baseline_rows = _select_baseline(
        alpha=alpha,
        params=params,
        I0=I0,
        funcs=funcs,
        target_p=target_p,
        args=args,
        previous_y=previous_y,
    )
    independent_seed_gate = any(
        row["init_source"].startswith("nearest_seed")
        and row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
        for row in baseline_rows
    )
    solves = {"base": base}
    load_by_stage = {"base": base["selected_load"]}
    stage_rows = baseline_rows + [
        p2c15._stage_row(alpha, args.eps_alpha, base, base["selected_load"]) | {"selected_baseline": True, "init_source": base["selected_init_source"]}
    ]
    for stage, sign in [("minus", -1.0), ("plus", 1.0)]:
        I = I0 + sign * args.eps_alpha * I_base
        solves[stage] = p2c15._solve(
            stage=stage,
            y0=base["y"],
            I=I,
            funcs=funcs,
            target_p=target_p,
            slack_scale=args.slack_scale,
            max_iter=args.max_iter,
        )
        load_by_stage[stage] = p2c15._audit_load(params, I, solves[stage]["y"])
        stage_rows.append(p2c15._stage_row(alpha, args.eps_alpha, solves[stage], load_by_stage[stage]) | {"selected_baseline": False})
    phi0 = solves["base"]["phi"]
    phi_plus = solves["plus"]["phi"]
    phi_minus = solves["minus"]["phi"]
    sensitivity = (phi_plus - phi_minus) / (2.0 * args.eps_alpha)
    scale = max(float(np.linalg.norm(phi0)), 1.0e-12)
    lin_plus = float(np.linalg.norm(phi_plus - (phi0 + sensitivity * args.eps_alpha)) / scale)
    lin_minus = float(np.linalg.norm(phi_minus - (phi0 - sensitivity * args.eps_alpha)) / scale)
    lin_sym = float(np.linalg.norm((phi_plus + phi_minus) * 0.5 - phi0) / scale)
    centerlines = {stage: np.asarray(solve["out"]["centerline"], dtype=float) for stage, solve in solves.items()}
    branch_rms_max, branch_rows = p2c15._branch_rms(centerlines)
    for row in branch_rows:
        row.update({"alpha": alpha, "eps_alpha": args.eps_alpha})
    max_stage_stationarity = max(solve["stationarity_norm"] for solve in solves.values())
    max_stage_condition = max(solve["jacobian_condition"] for solve in solves.values())
    max_linearization_error = max(lin_plus, lin_minus, lin_sym)
    max_slack_force_ratio = max(load["max_slack_force_over_rigid_force"] for load in load_by_stage.values())
    max_position_error = max(solve["position_error_m"] for solve in solves.values())
    summary = {
        "case": "n3_mixed_rand20",
        "alpha": alpha,
        "eps_alpha": args.eps_alpha,
        "nearest_seed_alpha": _nearest_seed_alpha(alpha),
        "selected_init_source": base["selected_init_source"],
        "independent_seed_gate": bool(independent_seed_gate),
        "baseline_success": base["success"],
        "publishable_gate": bool(
            all(solve["success"] for solve in solves.values())
            and independent_seed_gate
            and max_position_error <= 0.05 * 0.11
            and max_stage_stationarity <= 1.0e-5
            and max_stage_condition <= 1.0e10
            and max_linearization_error <= 0.05
            and branch_rms_max <= 1.0e-3
            and max_slack_force_ratio <= 2.0
        ),
        "sensitivity_norm_m_per_alpha": float(np.linalg.norm(sensitivity)),
        "sensitivity_vector": sensitivity.tolist(),
        "linearization_error_plus": lin_plus,
        "linearization_error_minus": lin_minus,
        "linearization_error_symmetry": lin_sym,
        "max_linearization_error": max_linearization_error,
        "branch_rms_max": branch_rms_max,
        "max_stage_stationarity": max_stage_stationarity,
        "max_stage_condition": max_stage_condition,
        "max_slack_force_over_rigid_force": max_slack_force_ratio,
        "max_position_error_m": max_position_error,
        "base_boundary_norm": base["boundary_norm"],
        "total_steady_s": sum(solve["steady_s"] for solve in solves.values()),
    }
    label, reasons = _classify(summary)
    summary["classification"] = label
    summary["failure_reasons"] = reasons
    if label != "可信工况":
        _append_jsonl(failures, {"ts": time.time(), "failure": "alpha_boundary_gate_failed", "reasons": reasons, "evidence": summary})
    np.savez_compressed(
        OUT_DIR / f"alpha_{alpha:.3f}_eps_{args.eps_alpha:.3f}_phase_map.npz",
        phi0=phi0,
        phi_plus=phi_plus,
        phi_minus=phi_minus,
        sensitivity=sensitivity,
        base_centerline=centerlines["base"],
        plus_centerline=centerlines["plus"],
        minus_centerline=centerlines["minus"],
        base_y=solves["base"]["y"],
        plus_y=solves["plus"]["y"],
        minus_y=solves["minus"]["y"],
    )
    return summary, stage_rows, branch_rows, base["y"]


def _plot(summaries: list[dict[str, Any]]) -> None:
    alphas = np.asarray([row["alpha"] for row in summaries], dtype=float)
    classes = [row["classification"] for row in summaries]
    colors = {"可信工况": "green", "临界工况": "orange", "不可用工况": "red"}
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    specs = [
        ("sensitivity_norm_m_per_alpha", "敏感度范数 m/alpha", None),
        ("max_stage_stationarity", "最大一阶平稳性", 1.0e-5),
        ("max_linearization_error", "最大线性化误差", 0.05),
        ("branch_rms_max", "最大中心线分支差异 m", 1.0e-3),
        ("max_slack_force_over_rigid_force", "最大界面力松弛/刚段外力", 2.0),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.scatter(alphas, [row[key] for row in summaries], c=[colors[c] for c in classes], s=55)
        ax.plot(alphas, [row[key] for row in summaries], color="black", alpha=0.35)
        if gate is not None:
            ax.axhline(gate, color="r", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("alpha")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_boundary_phase_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.zeros_like(alphas)
    ax.scatter(alphas, y, c=[colors[c] for c in classes], s=140)
    for alpha, cls in zip(alphas, classes, strict=True):
        ax.text(alpha, 0.05, cls, rotation=35, ha="center", va="bottom", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("alpha")
    ax.set_title("敏感度可信区、临界区与失效区分类")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_classification_strip.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for row in summaries:
        data = np.load(OUT_DIR / f"alpha_{row['alpha']:.3f}_eps_{row['eps_alpha']:.3f}_phase_map.npz")
        pts = np.asarray(data["base_centerline"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=f"a={row['alpha']:.3f} {row['classification']}")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("不同 alpha 基线中心线")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_centerline_by_alpha.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", default="0.35,0.375,0.40,0.425,0.45")
    parser.add_argument("--eps_alpha", type=float, default=0.01)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--boundary_scale", type=float, default=0.5)
    parser.add_argument("--slack_scale", type=float, default=20.0)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=240)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    previous_y: np.ndarray | None = None
    for alpha in _parse_floats(args.alphas):
        summary, stages, branches, previous_y = _run_alpha(
            alpha=alpha,
            previous_y=previous_y,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        summaries.append(summary)
        stage_rows.extend(stages)
        branch_rows.extend(branches)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "solve_rows.csv", stage_rows)
    _write_csv(OUT_DIR / "branch_rows.csv", branch_rows)
    _plot(summaries)
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
            "alpha_count": len(summaries),
            "publishable_count": sum(row["publishable_gate"] for row in summaries),
            "trusted_count": sum(row["classification"] == "可信工况" for row in summaries),
            "critical_count": sum(row["classification"] == "临界工况" for row in summaries),
            "failed_count": sum(row["classification"] == "不可用工况" for row in summaries),
        },
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
