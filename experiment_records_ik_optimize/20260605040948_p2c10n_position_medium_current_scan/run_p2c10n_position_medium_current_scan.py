"""Scan position-task sensitivity gates over medium current scales."""

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
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
P2C10M_DIR = ROOT / "experiment_records_ik_optimize" / "20260605032052_p2c10m_task_reduction_gate"
for path in (ROOT, P2C10I_DIR, P2C10M_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10i_boundary_candidate_strict_gate as p2c10i
import run_p2c10m_task_reduction_gate as p2c10m

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260605040948)


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


def _parse_cases(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_alphas(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _directions(I0: np.ndarray, random_count: int) -> list[tuple[str, np.ndarray]]:
    active = int(np.argmax(np.abs(I0))) if np.max(np.abs(I0)) > 0.0 else 5
    e = np.zeros(8)
    e[active] = 1.0
    directions = [(f"active_{active}", e)]
    for i in range(int(random_count)):
        d = RNG.normal(size=8)
        d = d / max(float(np.linalg.norm(d)), 1.0e-12)
        directions.append((f"random_{i}", d))
    return directions


def _classify_base(row: dict[str, Any]) -> tuple[str, str]:
    if not row["base_success"]:
        return "诊断结果", "基线求解未通过"
    if row["stationarity_norm"] > 1.0e-3:
        return "诊断结果", "一阶平稳性不足"
    if row["hessian_condition"] > 1.0e10:
        return "诊断结果", "局部条件数过高"
    if not row["branch_consistent"]:
        return "诊断结果", "多初值分支不一致"
    if not row["rank_ok"]:
        return "诊断结果", "末端位置有效秩不足"
    return "候选", "基线门禁通过"


def _classify_direction(base_status: str, row: dict[str, Any]) -> tuple[str, str]:
    if base_status != "候选":
        return "诊断结果", row["base_failure_class"]
    if not row["plus_success"] or not row["minus_success"]:
        return "诊断结果", "扰动重求解失败"
    if row["branch_jump"]:
        return "诊断结果", "扰动后分支跳变"
    if row["rel_err_central"] > 0.3 or row["cosine_central"] <= 0.0:
        return "诊断结果", "扰动有限差分不一致"
    if row["linearization_error"] > 0.1:
        return "诊断结果", "局部线性化误差过大"
    return "可发布候选", "通过严格门禁"


def _audit_sample(
    *,
    case_name: str,
    alpha: float,
    continuity_mu: float,
    reg_mu: float,
    max_iter: int,
    step: float,
    random_count: int,
    heartbeat: Path,
    failures: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_case = p2c10i._case_by_name(case_name)
    case = p2c10i._scaled_case(target_case, alpha)
    _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
    bundle = p2c10i._build_case_functions(case, continuity_mu, reg_mu, max_iter)
    base = bundle["base"]
    branch = p2c10i._branch_spread(bundle, max_iter)
    task_bundle = p2c10m._make_task_bundle(bundle, "position3")
    _ = task_bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"])).block_until_ready()
    sensitivity, diag = p2c10m._sensitivity_matrix(bundle, task_bundle)
    rank = p2c10m._rank_metrics(sensitivity, task_bundle["required_rank"])
    task0 = np.asarray(
        jax.device_get(task_bundle["task_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"]))),
        dtype=float,
    )
    base_row = {
        "case": case_name,
        "scaled_case": case.name,
        "alpha": alpha,
        "active_magnet_count": p2c10i._active_magnet_count(target_case),
        "current_norm": float(np.linalg.norm(bundle["I0"])),
        "current_max_abs": float(np.max(np.abs(bundle["I0"]))),
        "base_success": base.success,
        "base_residual_norm": base.residual_norm,
        "base_stationarity_norm": base.stationarity_norm,
        "base_boundary_norm": base.boundary_norm,
        "base_steady_s": base.steady_s,
        "stationarity_norm": diag["stationarity_norm"],
        "hessian_condition": diag["hessian_condition"],
        "hessian_min_singular": diag["hessian_min_singular"],
        "hessian_max_singular": diag["hessian_max_singular"],
        "rank": rank["rank"],
        "required_rank": rank["required_rank"],
        "rank_ok": rank["rank_ok"],
        "sensitivity_condition": rank["condition"],
        "sensitivity_sigma_max": rank["sigma_max"],
        "sensitivity_sigma_min": rank["sigma_min"],
        "sensitivity_fro_norm": rank["fro_norm"],
        "singular_values": rank["singular_values"],
        "task_norm": float(np.linalg.norm(task0)),
        **branch,
    }
    base_status, base_failure = _classify_base(base_row)
    base_row["base_status"] = base_status
    base_row["base_failure_class"] = base_failure

    rows: list[dict[str, Any]] = []
    for direction_name, direction in _directions(bundle["I0"], random_count):
        delta_I = float(step) * direction
        plus, minus = p2c10m._solve_perturbed(bundle, delta_I, max_iter)
        predicted = sensitivity @ delta_I
        task_plus = np.asarray(
            jax.device_get(task_bundle["task_jit"](jnp.asarray(plus.y), jnp.asarray(bundle["I0"] + delta_I))),
            dtype=float,
        )
        task_minus = np.asarray(
            jax.device_get(task_bundle["task_jit"](jnp.asarray(minus.y), jnp.asarray(bundle["I0"] - delta_I))),
            dtype=float,
        )
        actual_plus = task_plus - task0
        central_delta = 0.5 * (task_plus - task_minus)
        rel_plus, cosine_plus, actual_norm = p2c10m._safe_rel(predicted, actual_plus)
        rel_central, cosine_central, central_norm = p2c10m._safe_rel(predicted, central_delta)
        row = {
            **base_row,
            "direction": direction_name,
            "step_A": step,
            "rel_err_plus": rel_plus,
            "cosine_plus": cosine_plus,
            "actual_plus_norm": actual_norm,
            "rel_err_central": rel_central,
            "cosine_central": cosine_central,
            "central_delta_norm": central_norm,
            "linearization_error": float(np.linalg.norm(task_plus - task0 - predicted) / max(float(np.linalg.norm(task0)), 1.0e-12)),
            "predicted_norm": float(np.linalg.norm(predicted)),
            "plus_success": plus.success,
            "minus_success": minus.success,
            "branch_delta_plus": float(np.linalg.norm(plus.y - base.y)),
            "branch_delta_minus": float(np.linalg.norm(minus.y - base.y)),
            "branch_jump": bool(max(np.linalg.norm(plus.y - base.y), np.linalg.norm(minus.y - base.y)) > 5.0),
        }
        status, failure_class = _classify_direction(base_status, row)
        row["status"] = status
        row["failure_class"] = failure_class
        rows.append(row)
        if status != "可发布候选":
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "direction": direction_name, "failure_class": failure_class, "evidence": row})
    return base_row, rows


def _plot_alpha_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for case in sorted({row["case"] for row in rows}):
        subset = sorted([row for row in rows if row["case"] == case], key=lambda r: r["alpha"])
        ax.plot([row["alpha"] for row in subset], [row[metric] for row in subset], marker="o", label=case)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("current scale alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " versus current scale")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def _plot_publish_map(base_rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]]) -> None:
    cases = sorted({row["case"] for row in base_rows})
    alphas = sorted({row["alpha"] for row in base_rows})
    matrix = np.zeros((len(cases), len(alphas)))
    for i, case in enumerate(cases):
        for j, alpha in enumerate(alphas):
            rows = [row for row in direction_rows if row["case"] == case and abs(row["alpha"] - alpha) < 1.0e-12]
            matrix[i, j] = sum(1 for row in rows if row["status"] == "可发布候选") / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
    ax.set_yticks(range(len(cases)), cases)
    ax.set_xlabel("current scale alpha")
    ax.set_title("publishable direction ratio for terminal position task")
    fig.colorbar(im, ax=ax, label="publishable ratio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "publishable_position_map.png", dpi=180)
    plt.close(fig)


def _plot_failure_counts(rows: list[dict[str, Any]]) -> None:
    label_map = {
        "基线求解未通过": "base failed",
        "一阶平稳性不足": "stationarity failed",
        "局部条件数过高": "ill conditioned",
        "多初值分支不一致": "branch inconsistent",
        "末端位置有效秩不足": "rank deficient",
        "扰动重求解失败": "perturb solve failed",
        "扰动后分支跳变": "branch jump",
        "扰动有限差分不一致": "finite difference mismatch",
        "局部线性化误差过大": "linearization failed",
        "通过严格门禁": "strict gate passed",
    }
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["failure_class"]] = counts.get(row["failure_class"], 0) + 1
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(np.arange(len(labels)), values, color="#c5221f")
    ax.set_xticks(np.arange(len(labels)), [label_map.get(label, label) for label in labels], rotation=30, ha="right")
    ax.set_ylabel("direction samples")
    ax.set_title("failure class counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "failure_class_counts.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n1_tip_z_rand15,n2_z_pair_rand12,n3_mixed_rand20,n3_mixed_base_i4")
    parser.add_argument("--alphas", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=32)
    parser.add_argument("--random_count", type=int, default=1)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()

    base_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for case_name in _parse_cases(args.cases):
        for alpha in _parse_alphas(args.alphas):
            base_row, rows = _audit_sample(
                case_name=case_name,
                alpha=alpha,
                continuity_mu=args.continuity_mu,
                reg_mu=args.reg_mu,
                max_iter=args.max_iter,
                step=args.step,
                random_count=args.random_count,
                heartbeat=heartbeat,
                failures=failures,
            )
            base_rows.append(base_row)
            direction_rows.extend(rows)

    total_wall_s = float(time.perf_counter() - t_all)
    _plot_alpha_metric(base_rows, "stationarity_norm", "stationarity_vs_alpha.png", "stationarity residual", logy=True)
    _plot_alpha_metric(base_rows, "hessian_condition", "hessian_condition_vs_alpha.png", "second-order condition number", logy=True)
    _plot_alpha_metric(base_rows, "sensitivity_fro_norm", "position_sensitivity_norm_vs_alpha.png", "position sensitivity norm", logy=True)
    _plot_alpha_metric(base_rows, "base_steady_s", "time_cost_by_case.png", "base solve time (s)")
    _plot_publish_map(base_rows, direction_rows)
    _plot_failure_counts(direction_rows)

    class_counts: dict[str, int] = {}
    for row in direction_rows:
        class_counts[row["failure_class"]] = class_counts.get(row["failure_class"], 0) + 1
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": total_wall_s,
            "step_A": args.step,
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
        },
        "classification": "末端位置三维中等电流扫描",
        "publishable_count": sum(1 for row in direction_rows if row["status"] == "可发布候选"),
        "class_counts": class_counts,
        "base_rows": base_rows,
        "rows": direction_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "base_summary.csv", base_rows)
    _write_csv(OUT_DIR / "summary.csv", direction_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
