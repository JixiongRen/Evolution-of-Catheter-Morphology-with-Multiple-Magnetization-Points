"""Audit task-current drive rank for low-current shooting failures."""

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
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def _rank_from_singular_values(svals: np.ndarray, *, rel_tol: float, abs_tol: float) -> int:
    if svals.size == 0:
        return 0
    threshold = max(abs_tol, rel_tol * float(svals[0]))
    return int(np.sum(svals >= threshold))


def _spectrum_metrics(matrix: np.ndarray, *, rel_tol: float, abs_tol: float) -> dict[str, Any]:
    svals = np.linalg.svd(matrix, compute_uv=False)
    rank = _rank_from_singular_values(svals, rel_tol=rel_tol, abs_tol=abs_tol)
    if svals.size:
        cond = float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny))
        sigma_max = float(svals[0])
        sigma_min = float(svals[-1])
    else:
        cond = float("inf")
        sigma_max = 0.0
        sigma_min = 0.0
    return {
        "rank": rank,
        "condition": cond,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "fro_norm": float(np.linalg.norm(matrix)),
        "singular_values": svals.tolist(),
    }


def _classify(row: dict[str, Any], *, min_rank: int, min_sigma: float) -> str:
    if row["total_rank"] < min_rank:
        return "末端位姿有效驱动秩不足"
    if row["total_sigma_max"] < min_sigma:
        return "有效磁驱动强度不足"
    if not row["base_success"]:
        return "驱动存在但基线物理解不成立"
    if not row["branch_consistent"]:
        return "多解分支不一致"
    if row["hessian_condition"] > 1.0e10:
        return "局部可辨识性病态"
    return "驱动秩可接受"


def _plot_spectrum(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in rows:
        label = f"{row['case']}@{row['alpha']:.2f}"
        direct = np.maximum(np.asarray(row["direct_singular_values"], dtype=float), 1.0e-16)
        total = np.maximum(np.asarray(row["total_singular_values"], dtype=float), 1.0e-16)
        axes[0].plot(direct, marker="o", label=label)
        axes[1].plot(total, marker="o", label=label)
    axes[0].set_title("direct current-task spectrum")
    axes[1].set_title("implicit total sensitivity spectrum")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("singular value")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drive_singular_spectrum.png", dpi=180)
    plt.close(fig)


def _plot_coil_gain(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    gains = np.asarray([row["total_coil_gains"] for row in rows], dtype=float)
    labels = [f"{row['case']}@{row['alpha']:.2f}" for row in rows]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(rows) + 2.0)))
    im = ax.imshow(gains, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(8), [f"I{i}" for i in range(8)])
    ax.set_xlabel("coil current")
    ax.set_title("effective task gain per coil")
    fig.colorbar(im, ax=ax, label="effective task gain per ampere")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coil_gain_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_gate_vs_drive(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    marker_map = {1: "o", 2: "s", 3: "^"}
    for row in rows:
        ax.scatter(
            row["total_fro_norm"],
            row["base_stationarity_norm"],
            marker=marker_map.get(row["active_magnet_count"], "o"),
            s=90,
            color="#c5221f" if row["failure_class"] != "驱动秩可接受" else "#188038",
        )
        ax.text(row["total_fro_norm"], row["base_stationarity_norm"], f"{row['case']}@{row['alpha']:.2f}", fontsize=7)
    ax.axhline(1.0e-3, color="k", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective drive Frobenius norm")
    ax.set_ylabel("stationarity residual")
    ax.set_title("drive strength versus stationarity")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_vs_drive.png", dpi=180)
    plt.close(fig)


def _plot_time_cost(rows: list[dict[str, Any]], total_wall_s: float) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}" for row in rows] + ["total"]
    values = [row["base_steady_s"] for row in rows] + [total_wall_s]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.8))
    ax.bar(np.arange(len(labels)), values, color="#1f77b4")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("time (s)")
    ax.set_title("time cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_cost.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_rand20:0.10,n3_mixed_base_i4:1.00")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=32)
    parser.add_argument("--rank_rel_tol", type=float, default=1.0e-3)
    parser.add_argument("--rank_abs_tol", type=float, default=1.0e-8)
    parser.add_argument("--min_rank", type=int, default=6)
    parser.add_argument("--min_sigma", type=float, default=1.0e-8)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for case_name, alpha in _candidate_specs(args.candidates):
        target_case = p2c10i._case_by_name(case_name)
        case = p2c10i._scaled_case(target_case, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case.name})
        bundle = p2c10i._build_case_functions(case, args.continuity_mu, args.reg_mu, args.max_iter)
        base = bundle["base"]
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
        y_j = jnp.asarray(base.y, dtype=jnp.float64)
        I_j = jnp.asarray(bundle["I0"], dtype=jnp.float64)
        direct = np.asarray(jax.device_get(bundle["TI_jit"](y_j, I_j)), dtype=float)
        direct_metrics = _spectrum_metrics(direct, rel_tol=args.rank_rel_tol, abs_tol=args.rank_abs_tol)
        total_metrics = _spectrum_metrics(sensitivity, rel_tol=args.rank_rel_tol, abs_tol=args.rank_abs_tol)
        direct_coil_gains = np.linalg.norm(direct, axis=0)
        total_coil_gains = np.linalg.norm(sensitivity, axis=0)
        row = {
            "case": case_name,
            "scaled_case": case.name,
            "alpha": alpha,
            "active_magnet_count": p2c10i._active_magnet_count(target_case),
            "base_success": base.success,
            "base_residual_norm": base.residual_norm,
            "base_stationarity_norm": base.stationarity_norm,
            "base_boundary_norm": base.boundary_norm,
            "base_steady_s": base.steady_s,
            "hessian_condition": diag["hessian_condition"],
            "hessian_rank": diag["hessian_rank"],
            "direct_rank": direct_metrics["rank"],
            "direct_condition": direct_metrics["condition"],
            "direct_sigma_max": direct_metrics["sigma_max"],
            "direct_sigma_min": direct_metrics["sigma_min"],
            "direct_fro_norm": direct_metrics["fro_norm"],
            "direct_singular_values": direct_metrics["singular_values"],
            "total_rank": total_metrics["rank"],
            "total_condition": total_metrics["condition"],
            "total_sigma_max": total_metrics["sigma_max"],
            "total_sigma_min": total_metrics["sigma_min"],
            "total_fro_norm": total_metrics["fro_norm"],
            "total_singular_values": total_metrics["singular_values"],
            "direct_coil_gains": direct_coil_gains.tolist(),
            "total_coil_gains": total_coil_gains.tolist(),
            "max_direct_coil_gain": float(np.max(direct_coil_gains)),
            "max_total_coil_gain": float(np.max(total_coil_gains)),
            "currents": bundle["I0"].tolist(),
            **branch,
        }
        row["failure_class"] = _classify(row, min_rank=args.min_rank, min_sigma=args.min_sigma)
        rows.append(row)
        if row["failure_class"] != "驱动秩可接受":
            _append_jsonl(failures, {"ts": time.time(), "case": case.name, "failure_class": row["failure_class"], "evidence": row})

    total_wall_s = float(time.perf_counter() - t_all)
    _plot_spectrum(rows)
    _plot_coil_gain(rows)
    _plot_gate_vs_drive(rows)
    _plot_time_cost(rows, total_wall_s)
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["failure_class"]] = class_counts.get(row["failure_class"], 0) + 1
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": total_wall_s,
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
            "rank_rel_tol": args.rank_rel_tol,
            "rank_abs_tol": args.rank_abs_tol,
        },
        "classification": "驱动秩审计",
        "class_counts": class_counts,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
