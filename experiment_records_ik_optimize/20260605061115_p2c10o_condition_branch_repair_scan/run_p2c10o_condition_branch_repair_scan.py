"""Scan regularization settings for position-task sensitivity gates."""

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
P2C10N_DIR = ROOT / "experiment_records_ik_optimize" / "20260605040948_p2c10n_position_medium_current_scan"
for path in (ROOT, P2C10N_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10n_position_medium_current_scan as p2c10n

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


def _parse_candidates(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _candidate_label(row: dict[str, Any]) -> str:
    return f"{row['case']}@{row['alpha']:.2f}"


def _plot_heatmap(
    rows: list[dict[str, Any]],
    *,
    case_name: str,
    alpha: float,
    metric: str,
    filename: str,
    title: str,
    log_value: bool = False,
) -> None:
    subset = [row for row in rows if row["case"] == case_name and abs(row["alpha"] - alpha) < 1.0e-12]
    if not subset:
        return
    continuity_values = sorted({row["continuity_mu"] for row in subset})
    reg_values = sorted({row["reg_mu"] for row in subset})
    matrix = np.full((len(continuity_values), len(reg_values)), np.nan)
    for row in subset:
        i = continuity_values.index(row["continuity_mu"])
        j = reg_values.index(row["reg_mu"])
        value = row[metric]
        matrix[i, j] = np.log10(max(float(value), 1.0e-300)) if log_value else float(value)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(reg_values)), [f"{x:.0e}" for x in reg_values])
    ax.set_yticks(range(len(continuity_values)), [f"{x:g}" for x in continuity_values])
    ax.set_xlabel("state regularization weight")
    ax.set_ylabel("continuity weight")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"log10({metric})" if log_value else metric)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def _plot_gate_summary(rows: list[dict[str, Any]]) -> None:
    labels = [_candidate_label(row) + f"\ncmu={row['continuity_mu']:g}, rmu={row['reg_mu']:.0e}" for row in rows]
    x = np.arange(len(rows))
    colors = ["#188038" if row["base_status"] == "候选" else "#c5221f" for row in rows]
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.65), 5.5))
    ax.bar(x, [row["hessian_condition"] for row in rows], color=colors)
    ax.axhline(1.0e10, color="k", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("second-order condition number")
    ax.set_title("condition gate under branch-regularization scan")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "condition_gate_summary.png", dpi=180)
    plt.close(fig)


def _plot_publishable(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, tuple[int, int]] = {}
    for row in rows:
        key = _candidate_label(row)
        pub, total = counts.get(key, (0, 0))
        counts[key] = (pub + int(row["status"] == "可发布候选"), total + 1)
    labels = list(counts)
    values = [counts[label][0] / max(counts[label][1], 1) for label in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(np.arange(len(labels)), values, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("publishable direction ratio")
    ax.set_title("publishable ratio after repair scan")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "publishable_ratio.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00")
    parser.add_argument("--continuity_mus", default="1.0,3.0")
    parser.add_argument("--reg_mus", default="1e-5,1e-4,1e-3")
    parser.add_argument("--step", type=float, default=0.05)
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
    for case_name, alpha in _parse_candidates(args.candidates):
        for continuity_mu in _parse_floats(args.continuity_mus):
            for reg_mu in _parse_floats(args.reg_mus):
                base_row, rows = p2c10n._audit_sample(
                    case_name=case_name,
                    alpha=alpha,
                    continuity_mu=continuity_mu,
                    reg_mu=reg_mu,
                    max_iter=args.max_iter,
                    step=args.step,
                    random_count=args.random_count,
                    heartbeat=heartbeat,
                    failures=failures,
                )
                base_row["continuity_mu"] = continuity_mu
                base_row["reg_mu"] = reg_mu
                base_rows.append(base_row)
                for row in rows:
                    row["continuity_mu"] = continuity_mu
                    row["reg_mu"] = reg_mu
                    direction_rows.append(row)

    total_wall_s = float(time.perf_counter() - t_all)
    _plot_gate_summary(base_rows)
    _plot_publishable(direction_rows)
    for case_name, alpha in _parse_candidates(args.candidates):
        safe_case = case_name.replace("/", "_")
        _plot_heatmap(
            base_rows,
            case_name=case_name,
            alpha=alpha,
            metric="hessian_condition",
            filename=f"{safe_case}_condition_heatmap.png",
            title=f"{case_name}@{alpha:.2f} condition number",
            log_value=True,
        )
        _plot_heatmap(
            base_rows,
            case_name=case_name,
            alpha=alpha,
            metric="branch_task_spread",
            filename=f"{safe_case}_branch_task_spread_heatmap.png",
            title=f"{case_name}@{alpha:.2f} branch task spread",
            log_value=True,
        )
        _plot_heatmap(
            base_rows,
            case_name=case_name,
            alpha=alpha,
            metric="stationarity_norm",
            filename=f"{safe_case}_stationarity_heatmap.png",
            title=f"{case_name}@{alpha:.2f} stationarity residual",
            log_value=True,
        )

    class_counts: dict[str, int] = {}
    for row in direction_rows:
        class_counts[row["failure_class"]] = class_counts.get(row["failure_class"], 0) + 1
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": total_wall_s,
            "step_A": args.step,
            "max_iter": args.max_iter,
        },
        "classification": "末端位置病态与分支修复扫描",
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
