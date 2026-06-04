"""Representative feasible-branch distance phase map for P2c."""

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
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C10G_DIR = ROOT / "experiment_records_ik_optimize" / "20260605004600_p2c10g_feasible_manifold_distance_map"
for path in (ROOT, P2C4_DIR, P2C10G_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_clean_multi_case as p2c4
import run_p2c10g_feasible_manifold_distance_map as p2c10g

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


def _case_by_name(name: str) -> Any:
    for case in p2c4._cases():
        if case.name == name:
            return case
    raise ValueError(f"Unknown case: {name}")


def _scaled_case(case: Any, alpha: float) -> Any:
    currents = alpha * np.asarray(case.currents, dtype=float)
    return p2c4.CaseSpec(
        name=f"{case.name}_alpha_{alpha:.2f}",
        m_body_list=case.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _active_magnet_count(case: Any) -> int:
    return int(sum(np.linalg.norm(np.asarray(m, dtype=float)) > 1.0e-12 for m in case.m_body_list))


def _zone(row: dict[str, Any]) -> str:
    pos = row["terminal_position_error_m"]
    ori = row["terminal_orientation_error_deg"]
    stat = row["stationarity_norm"]
    if stat <= 1.0e-3 and pos <= 0.005 and ori <= 15.0:
        return "可信候选区"
    if pos <= 0.008 and ori <= 25.0:
        return "边界候选区"
    return "高风险区"


def _plot_phase(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    markers = {1: "o", 2: "s", 3: "^"}
    for case_name in sorted({row["case"] for row in rows}):
        sub = sorted([row for row in rows if row["case"] == case_name], key=lambda item: item["alpha"])
        count = sub[0]["active_magnet_count"]
        axes[0].plot(
            [row["alpha"] for row in sub],
            [row["terminal_position_error_m"] for row in sub],
            marker=markers.get(count, "o"),
            label=f"{case_name} ({count} magnets)",
        )
        axes[1].plot(
            [row["alpha"] for row in sub],
            [row["terminal_orientation_error_deg"] for row in sub],
            marker=markers.get(count, "o"),
            label=f"{case_name} ({count} magnets)",
        )
    axes[0].axhline(0.005, color="#188038", linestyle="--", linewidth=1.0, label="5 mm")
    axes[0].axhline(0.008, color="#c5221f", linestyle="--", linewidth=1.0, label="8 mm")
    axes[1].axhline(15.0, color="#188038", linestyle="--", linewidth=1.0, label="15 deg")
    axes[1].axhline(25.0, color="#c5221f", linestyle="--", linewidth=1.0, label="25 deg")
    axes[0].set_xlabel("current scaling alpha")
    axes[1].set_xlabel("current scaling alpha")
    axes[0].set_ylabel("terminal position error (m)")
    axes[1].set_ylabel("terminal orientation error (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Representative distance from target to feasible branch")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_distance_phase_map.png", dpi=180)
    plt.close(fig)


def _plot_zone(rows: list[dict[str, Any]]) -> None:
    zones = ["可信候选区", "边界候选区", "高风险区"]
    colors = {"可信候选区": "#188038", "边界候选区": "#f9ab00", "高风险区": "#c5221f"}
    labels = {"可信候选区": "safe", "边界候选区": "border", "高风险区": "risk"}
    case_names = sorted({row["case"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    zone_matrix = np.zeros((len(case_names), len(alphas)), dtype=int)
    zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
    for i, case_name in enumerate(case_names):
        for j, alpha in enumerate(alphas):
            item = next(row for row in rows if row["case"] == case_name and row["alpha"] == alpha)
            zone_matrix[i, j] = zone_to_idx[item["zone"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = matplotlib.colors.ListedColormap([colors[zone] for zone in zones])
    ax.imshow(zone_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=len(zones) - 1)
    ax.set_xticks(range(len(alphas)), [f"{alpha:.2f}" for alpha in alphas])
    ax.set_yticks(range(len(case_names)), case_names)
    ax.set_xlabel("current scaling alpha")
    ax.set_title("Feasible-branch distance zone map")
    for i, case_name in enumerate(case_names):
        for j, alpha in enumerate(alphas):
            item = next(row for row in rows if row["case"] == case_name and row["alpha"] == alpha)
            ax.text(j, i, labels[item["zone"]], ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_zone_map.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="n1_tip_z_rand15,n2_z_pair_rand12,n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--continuity_scale", type=float, default=1000.0)
    parser.add_argument("--boundary_scale", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-6)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    case_names = [name.strip() for name in args.cases.split(",") if name.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for case_name in case_names:
        target_case = _case_by_name(case_name)
        active_count = _active_magnet_count(target_case)
        y0 = None
        for alpha in alphas:
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": case_name, "alpha": alpha})
            funcs = p2c10g._build_functions(
                _scaled_case(target_case, alpha),
                continuity_scale=args.continuity_scale,
                boundary_scale=args.boundary_scale,
                reg_mu=args.reg_mu,
                steps_per_flex=args.steps_per_flex,
            )
            if y0 is None:
                y0 = funcs["y_ref"]
            _ = funcs["feasibility_residual_jit"](jnp.asarray(y0)).block_until_ready()
            _ = funcs["feasibility_jac_jit"](jnp.asarray(y0)).block_until_ready()
            _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "case": case_name, "alpha": alpha})
            result = p2c10g._solve_feasible(funcs, y0, max_iter=args.max_iter)
            y0 = result["y"]
            row = {
                "case": case_name,
                "active_magnet_count": active_count,
                "alpha": alpha,
                "success_flag": result["success_flag"],
                "nfev": result["nfev"],
                "steady_s": result["steady_s"],
                "boundary_norm": result["boundary_norm"],
                "raw_continuity_norm": result["raw_continuity_norm"],
                "stationarity_norm": result["stationarity_norm"],
                "terminal_position_error_m": result["terminal_position_error_m"],
                "terminal_orientation_error_deg": result["terminal_orientation_error_deg"],
                "tip_p": result["tip_p"],
                "target_p": result["target_p"],
                "diagnostic_class": "representative-feasible-distance",
            }
            row["zone"] = _zone(row)
            rows.append(row)
            if row["zone"] == "高风险区":
                _append_jsonl(failures, {"ts": time.time(), "failure_class": "代表性工况高风险区", "evidence": row})

    _plot_phase(rows)
    _plot_zone(rows)
    per_case = []
    for case_name in case_names:
        sub = [row for row in rows if row["case"] == case_name]
        safe = [row for row in sub if row["zone"] == "可信候选区"]
        boundary = [row for row in sub if row["zone"] == "边界候选区"]
        per_case.append(
            {
                "case": case_name,
                "active_magnet_count": sub[0]["active_magnet_count"] if sub else None,
                "safe_alpha_max": max([row["alpha"] for row in safe], default=None),
                "boundary_alpha_max": max([row["alpha"] for row in boundary], default=None),
                "high_risk_count": sum(row["zone"] == "高风险区" for row in sub),
                "max_position_error_m": max([row["terminal_position_error_m"] for row in sub], default=float("nan")),
                "max_orientation_error_deg": max([row["terminal_orientation_error_deg"] for row in sub], default=float("nan")),
            }
        )
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "classification_thresholds": {
                "safe_candidate": "stationarity <= 1e-3, position <= 0.005 m, orientation <= 15 deg",
                "boundary_candidate": "position <= 0.008 m and orientation <= 25 deg",
                "high_risk": "otherwise",
            },
        },
        "classification": "representative-feasible-distance-phase-map",
        "per_case": per_case,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    _write_csv(OUT_DIR / "per_case_summary.csv", per_case)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
