"""Run a small cross-condition sensitivity reliability pilot."""

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
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
for path in (ROOT, P2C10X_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c10x_position_alpha_linearity_audit as p2c10x


configure_matplotlib_cjk()
OUT_DIR = Path(__file__).resolve().parent
p2c10x.OUT_DIR = OUT_DIR


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
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        case, alpha = item.split(":", 1)
        specs.append((case.strip(), float(alpha)))
    return specs


def _magnet_count(case_name: str) -> int:
    if case_name.startswith("n1_"):
        return 1
    if case_name.startswith("n2_"):
        return 2
    if case_name.startswith("n3_"):
        return 3
    return -1


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if not row.get("baseline_success", False):
        modes.append("基线求解未成功")
    if not row.get("baseline_task_gate", False):
        modes.append("可达域或末端约束失配")
    if float(row.get("baseline_stationarity_norm", 0.0)) > 1.0e-5:
        modes.append("一阶最优性退化")
    if float(row.get("baseline_jacobian_condition", 0.0)) > 1.0e10:
        modes.append("数值病态或局部不可辨识")
    max_lin = max(
        float(row.get("linearization_error_plus", 0.0)),
        float(row.get("linearization_error_minus", 0.0)),
        float(row.get("linearization_error_symmetry", 0.0)),
    )
    if max_lin > 0.05:
        modes.append("局部线性近似失效")
    if float(row.get("branch_rms_max", 0.0)) > 1.0e-3:
        modes.append("分支不一致或分支切换")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    priority = [
        "分支不一致或分支切换",
        "一阶最优性退化",
        "局部线性近似失效",
        "数值病态或局部不可辨识",
        "可达域或末端约束失配",
        "基线求解未成功",
    ]
    for item in priority:
        if item in modes:
            return item
    return modes[0]


def _annotate_summary(row: dict[str, Any]) -> dict[str, Any]:
    modes = [] if row["publishable_gate"] else _failure_modes(row)
    annotated = dict(row)
    annotated["magnet_count"] = _magnet_count(row["case"])
    annotated["evidence_type"] = "高置信敏感度" if row["publishable_gate"] else "诊断结果"
    annotated["failure_modes"] = ";".join(modes)
    annotated["primary_failure_mode"] = "" if row["publishable_gate"] else _primary_mode(modes)
    annotated["max_linearization_error"] = max(
        float(row["linearization_error_plus"]),
        float(row["linearization_error_minus"]),
        float(row["linearization_error_symmetry"]),
    )
    return annotated


def _plot_reliability_map(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        color = "green" if row["publishable_gate"] else "orange"
        ax.scatter(row["magnet_count"], row["alpha"], s=130, color=color)
        ax.text(row["magnet_count"] + 0.03, row["alpha"], row["case"], fontsize=7, va="center")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("磁化段数量")
    ax.set_ylabel("电流缩放系数 alpha")
    ax.set_title("跨工况敏感度可靠性覆盖图")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_reliability_map.png", dpi=180)
    plt.close(fig)


def _plot_metric_bars(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}\nalpha={row['alpha']:.2f}" for row in rows]
    x = np.arange(len(labels))
    colors = ["green" if row["publishable_gate"] else "orange" for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    specs = [
        ("sensitivity_norm_m_per_alpha", "敏感度范数 m/alpha", None),
        ("max_linearization_error", "最大线性化误差", 0.05),
        ("baseline_stationarity_norm", "基线一阶最优性", 1.0e-5),
        ("branch_rms_max", "分支中心线差异 m", 1.0e-3),
    ]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [float(row[key]) for row in rows], color=colors)
        if gate is not None:
            ax.axhline(gate, color="black", ls="--", lw=1.0)
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_gate_metrics.png", dpi=180)
    plt.close(fig)


def _plot_failure_counts(rows: list[dict[str, Any]]) -> None:
    counter = Counter(row["primary_failure_mode"] for row in rows if row["primary_failure_mode"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(counter)
    ax.bar(range(len(labels)), [counter[label] for label in labels], color="#9c4a1a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("样本数量")
    ax.set_title("跨工况失败主类型")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_failure_counts.png", dpi=180)
    plt.close(fig)


def _plot_centerlines(rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for row in rows:
        path = OUT_DIR / f"{row['case']}_alpha_{row['alpha']:.2f}_position_linearity.npz"
        if not path.exists():
            continue
        pts = np.asarray(np.load(path)["base_centerline"], dtype=float)
        label = f"{row['case']} a={row['alpha']:.2f}"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=30)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("代表性跨工况中心线")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_centerline_representatives.png", dpi=180)
    plt.close(fig)


def _plot_time_cost(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{row['case']}\n{row['alpha']:.2f}" for row in rows]
    values = [float(row.get("total_steady_s", 0.0)) for row in rows]
    ax.bar(range(len(labels)), values, color="#3f6c8f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("净求解时间 s")
    ax.set_title("跨工况敏感度审计时间成本")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_condition_time_cost.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_base_i4:1.00,n3_mixed_rand20:0.40",
    )
    parser.add_argument("--eps_alpha", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=160)
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
    baseline_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for case_name, alpha in _parse_candidates(args.candidates):
        summary, base_rows, pert_rows, pairs = p2c10x._run_alpha(
            case_name=case_name,
            alpha=alpha,
            eps_alpha=args.eps_alpha,
            args=args,
            heartbeat=heartbeat,
            failures=failures,
        )
        annotated = _annotate_summary(summary)
        summaries.append(annotated)
        baseline_rows.extend(base_rows)
        perturb_rows.extend(pert_rows)
        pair_rows.extend(pairs)
        if not annotated["publishable_gate"]:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "failure": "cross_condition_reliability_gate_failed",
                    "case": case_name,
                    "alpha": alpha,
                    "failure_modes": annotated["failure_modes"],
                    "primary_failure_mode": annotated["primary_failure_mode"],
                    "evidence": annotated,
                },
            )

    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "baseline_rows.csv", baseline_rows)
    _write_csv(OUT_DIR / "perturb_rows.csv", perturb_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", pair_rows)
    _plot_reliability_map(summaries)
    _plot_metric_bars(summaries)
    _plot_failure_counts(summaries)
    _plot_centerlines(summaries)
    _plot_time_cost(summaries)

    failure_counter = Counter(row["primary_failure_mode"] for row in summaries if row["primary_failure_mode"])
    recommendations = {
        "verified_conclusions": [
            "本轮把 P2c21 门禁迁移到单磁化段、双磁化段和三磁化段代表工况。",
            "发布敏感度必须同时满足基线、扰动重求解、线性化误差和分支一致性门禁。",
        ],
        "engineering_guidance": [
            "若单磁化段或双磁化段样本出现低敏感度但门禁通过，可作为低风险工作区候选。",
            "若三磁化段样本失败，应优先检查多解竞争和段间载荷传播，而不是只调末端误差权重。",
            "跨工况扩展时应保留每个工况自身电流缩放方向，避免把不同线圈组合混为同一扰动方向。",
        ],
    }
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
            "publishable_count": sum(row["publishable_gate"] for row in summaries),
            "diagnostic_count": sum(not row["publishable_gate"] for row in summaries),
            "failure_mode_count": dict(failure_counter),
            "magnet_counts": sorted({row["magnet_count"] for row in summaries}),
        },
        "recommendations": recommendations,
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
