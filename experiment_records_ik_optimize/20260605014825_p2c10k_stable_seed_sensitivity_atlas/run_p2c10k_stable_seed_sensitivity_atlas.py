"""Build a seed atlas of strict-gated shooting task sensitivity cases."""

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
for path in (ROOT, P2C10I_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_p2c10i_boundary_candidate_strict_gate as p2c10i

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


def _candidate_specs(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha = raw.split(":", 1)
        specs.append((name.strip(), float(alpha)))
    return specs


def _zone(case_rows: list[dict[str, Any]], row_rows: list[dict[str, Any]], case_name: str, alpha: float) -> str:
    case = next(row for row in case_rows if row["case"] == case_name and row["alpha"] == alpha)
    sub = [row for row in row_rows if row["case"] == case_name and row["alpha"] == alpha]
    if sub and all(row["status"] == "可发布候选" for row in sub):
        return "可信区"
    if case["base_success"] and case["base_stationarity_norm"] <= 1.0e-3 and case["hessian_condition"] <= 1.0e10:
        return "临界区"
    return "诊断区"


def _zone_label(zone: str) -> str:
    labels = {"可信区": "trusted", "临界区": "borderline", "诊断区": "diagnostic"}
    return labels.get(zone, zone)


def _direction_label(direction: str) -> str:
    if direction.startswith("主激活线圈_"):
        return "active_" + direction.rsplit("_", 1)[-1]
    if direction.startswith("随机方向_"):
        return "random_" + direction.rsplit("_", 1)[-1]
    return direction


def _plot_gate_scatter(case_rows: list[dict[str, Any]], row_rows: list[dict[str, Any]]) -> None:
    if not case_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"可信区": "#188038", "临界区": "#f9ab00", "诊断区": "#c5221f"}
    markers = {1: "o", 2: "s", 3: "^"}
    for row in case_rows:
        zone = _zone(case_rows, row_rows, row["case"], row["alpha"])
        ax.scatter(
            row["hessian_condition"],
            row["base_stationarity_norm"],
            s=90,
            marker=markers.get(row["active_magnet_count"], "o"),
            color=colors[zone],
            label=f"{_zone_label(zone)}-{row['active_magnet_count']}",
        )
        ax.text(row["hessian_condition"], row["base_stationarity_norm"], f"{row['case']}@{row['alpha']:.2f}", fontsize=7)
    ax.axhline(1.0e-3, color="k", linestyle="--", linewidth=1.0)
    ax.axvline(1.0e10, color="k", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("local condition number")
    ax.set_ylabel("stationarity residual")
    ax.set_title("strict gate seed atlas")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_scatter.png", dpi=180)
    plt.close(fig)


def _plot_direction_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [f"{row['case']}@{row['alpha']:.2f}\\n{_direction_label(row['direction'])}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.3), 8))
    axes[0].bar(x, [row["linearization_error"] for row in rows], color="#1f77b4")
    axes[0].axhline(0.1, color="r", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("linearization error")
    axes[0].set_title("local linearization gate")
    axes[1].bar(x, [row["rel_err_central"] for row in rows], color="#ff7f0e")
    axes[1].axhline(0.3, color="r", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("central FD relative error")
    axes[1].set_title("re-optimized finite-difference gate")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "direction_gate_metrics.png", dpi=180)
    plt.close(fig)


def _plot_status_counts(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(counts)), list(counts.values()), color=["#188038" if key == "可发布候选" else "#c5221f" for key in counts])
    ax.set_xticks(range(len(counts)), ["publishable" if key == "可发布候选" else "diagnostic" for key in counts])
    ax.set_ylabel("count")
    ax.set_title("status counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "status_counts.png", dpi=180)
    plt.close(fig)


def _plot_time_cost(case_rows: list[dict[str, Any]], total_wall_s: float) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}" for row in case_rows] + ["total"]
    values = [row["base_steady_s"] for row in case_rows] + [total_wall_s]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(np.arange(len(labels)), values, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("time (s)")
    ax.set_title("time cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_cost.png", dpi=180)
    plt.close(fig)


def _plot_representative_morphology(morphology_rows: list[dict[str, Any]]) -> None:
    if not morphology_rows:
        return
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for row in morphology_rows:
        centerline = np.asarray(row["centerline"], dtype=float)
        target_p = np.asarray(row["target_p"], dtype=float)
        tip_p = np.asarray(row["tip_p"], dtype=float)
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], linewidth=2.0, label=row["label"])
        ax.scatter([target_p[0]], [target_p[1]], [target_p[2]], marker="^", s=50)
        ax.scatter([tip_p[0]], [tip_p[1]], [tip_p[2]], marker="o", s=35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("representative morphology")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_morphology.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_rand20:0.10,n3_mixed_base_i4:1.00",
    )
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
    case_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []

    for case_name, alpha in _candidate_specs(args.candidates):
        target_case = p2c10i._case_by_name(case_name)
        case = p2c10i._scaled_case(target_case, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
        bundle = p2c10i._build_case_functions(case, args.continuity_mu, args.reg_mu, args.max_iter)
        base = bundle["base"]
        continuity_norm = float(
            np.linalg.norm(
                np.asarray(jax.device_get(bundle["continuity_jit"](jnp.asarray(base.y), jnp.asarray(bundle["I0"]))), dtype=float)
            )
        )
        branch = p2c10i._branch_spread(bundle, args.max_iter)
        sensitivity, diag = p2c10i._exact_sensitivity_matrix(
            y=base.y,
            I=bundle["I0"],
            grad_jit=bundle["grad_jit"],
            H_jit=bundle["H_jit"],
            GI_jit=bundle["GI_jit"],
            Ty_jit=bundle["Ty_jit"],
            TI_jit=bundle["TI_jit"],
        )
        case_row = {
            "case": case_name,
            "scaled_case": case.name,
            "alpha": alpha,
            "active_magnet_count": p2c10i._active_magnet_count(target_case),
            "base_success": base.success,
            "base_residual_norm": base.residual_norm,
            "base_stationarity_norm": base.stationarity_norm,
            "base_boundary_norm": base.boundary_norm,
            "base_continuity_norm": continuity_norm,
            "base_steady_s": base.steady_s,
            "sensitivity_fro_norm": float(np.linalg.norm(sensitivity)),
            "currents": bundle["I0"].tolist(),
            **branch,
            **diag,
        }
        case_rows.append(case_row)
        if len(morphology_rows) < 3:
            morphology_rows.append(
                {
                    "label": f"{case_name}@{alpha:.2f}",
                    "centerline": np.asarray(base.out["centerline"], dtype=float).tolist(),
                    "tip_p": np.asarray(base.out["tip_p"], dtype=float).reshape(3).tolist(),
                    "target_p": np.asarray(bundle["target_p"], dtype=float).reshape(3).tolist(),
                }
            )

        for direction_name, direction in p2c10i._directions(bundle["I0"], args.random_count):
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "audit", "case": case.name, "direction": direction_name})
            direction_row = p2c10i._audit_direction(
                bundle=bundle,
                sensitivity=sensitivity,
                direction_name=direction_name,
                direction=direction,
                step=args.step,
                max_iter=args.max_iter,
            )
            row = {**case_row, **direction_row}
            status, failure_class = p2c10i._classify(row)
            row["status"] = status
            row["failure_class"] = failure_class
            row_rows.append(row)
            if status != "可发布候选":
                _append_jsonl(
                    failures,
                    {
                        "ts": time.time(),
                        "case": case.name,
                        "direction": direction_name,
                        "failure_class": failure_class,
                        "evidence": row,
                    },
                )

    total_wall_s = float(time.perf_counter() - t_all)
    for case_row in case_rows:
        case_row["zone"] = _zone(case_rows, row_rows, case_row["case"], case_row["alpha"])

    _plot_gate_scatter(case_rows, row_rows)
    _plot_direction_metrics(row_rows)
    _plot_status_counts(row_rows)
    _plot_time_cost(case_rows, total_wall_s)
    _plot_representative_morphology(morphology_rows)
    publishable = [row for row in row_rows if row["status"] == "可发布候选"]
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
        "classification": "可信样本库种子扫描",
        "case_count": len(case_rows),
        "direction_count": len(row_rows),
        "publishable_direction_count": len(publishable),
        "publishable_case_count": sum(1 for row in case_rows if row["zone"] == "可信区"),
        "case_rows": case_rows,
        "rows": row_rows,
        "morphology_rows": morphology_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "case_summary.csv", case_rows)
    _write_csv(OUT_DIR / "summary.csv", row_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
