"""Cluster alpha=0.375 shooting branches and audit round-trip stability."""

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
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
P2C15_DIR = ROOT / "experiment_records_ik_optimize" / "20260605125315_p2c15_slack_suppressed_sensitivity_audit"
P2C16_DIR = ROOT / "experiment_records_ik_optimize" / "20260605132600_p2c16_alpha_boundary_phase_map"
for path in (ROOT, P2C10W_DIR, P2C10X_DIR, P2C13_DIR, P2C15_DIR, P2C16_DIR):
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
import run_p2c13_rigid_load_propagation_audit as p2c13
import run_p2c15_slack_suppressed_sensitivity_audit as p2c15
import run_p2c16_alpha_boundary_phase_map as p2c16

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


def _load_npz_y(path: str, key: str = "base_y") -> np.ndarray:
    return np.asarray(np.load(ROOT / path)[key], dtype=float)


def _case(alpha: float, args: argparse.Namespace) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    target_p = np.asarray(target_p, dtype=float).reshape(3)
    funcs = p2c15._make_functions(params, target_p, args.position_scale, args.boundary_scale, args.steps_per_flex)
    return params, I0, target_p, funcs


def _baseline_gate(row: dict[str, Any]) -> bool:
    return bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] <= 1.0e10
        and row["max_slack_force_over_rigid_force"] <= 2.0
    )


def _candidate_seeds(alpha: float, args: argparse.Namespace) -> list[tuple[str, str, np.ndarray]]:
    params, _I0, _target_p, _target_q, y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", alpha)
    nearest_035 = p2c15._load_seed_y(0.35)
    nearest_040 = p2c15._load_seed_y(0.40)
    cont_0376 = _load_npz_y("experiment_records_ik_optimize/20260605153000_p2c18_lower_alpha_breakpoint_load_audit/alpha_0.376_lower_breakpoint.npz")
    cont_0375 = _load_npz_y("experiment_records_ik_optimize/20260606103000_p2c19_ultrafine_lower_alpha_breakpoint/alpha_0.3750_ultrafine_lower_breakpoint.npz")
    zero = p2c10w._initial_vector(y_ref, "zero", params)
    half_oracle = p2c10w._initial_vector(y_ref, "half_oracle", params)
    oracle = p2c10w._initial_vector(y_ref, "oracle", params)
    seeds: list[tuple[str, str, np.ndarray]] = [
        ("nearest_seed_0.35", "strict_independent", nearest_035),
        ("nearest_seed_0.40", "strict_independent", nearest_040),
        ("zero", "strict_independent", zero),
        ("half_oracle", "assisted_reference", half_oracle),
        ("oracle", "assisted_reference", oracle),
        ("continuation_0.376", "path_continuation", cont_0376),
        ("continuation_0.375", "path_continuation", cont_0375),
    ]
    for w in (0.25, 0.50, 0.75):
        seeds.append((f"blend_035_040_{w:.2f}", "mixed_seed", (1.0 - w) * nearest_035 + w * nearest_040))
        seeds.append((f"blend_040_cont_{w:.2f}", "mixed_seed", (1.0 - w) * nearest_040 + w * cont_0376))
        seeds.append((f"blend_035_cont_{w:.2f}", "mixed_seed", (1.0 - w) * nearest_035 + w * cont_0376))
    perturb = np.zeros_like(cont_0376)
    perturb[:6] = np.array([1, -1, 0.5, -0.5, 0.25, -0.25])
    for scale in (1.0e-4, 5.0e-4, 1.0e-3):
        seeds.append((f"cont_perturb_{scale:g}", "path_continuation", cont_0376 + scale * perturb))
    return seeds


def _solve_seed(alpha: float, seed_name: str, seed_class: str, y0: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    params, I0, target_p, funcs = _case(alpha, args)
    solve = p2c15._solve(
        stage=seed_name,
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
        "seed_name": seed_name,
        "seed_class": seed_class,
        **{k: v for k, v in solve.items() if k not in {"y", "out", "phi"}},
        **load,
    }
    row["baseline_gate"] = _baseline_gate(row)
    row["branch_label"] = "physical_branch" if row["baseline_gate"] else "nonphysical_or_unresolved"
    row["tip_x"] = float(solve["phi"][0])
    row["tip_y"] = float(solve["phi"][1])
    row["tip_z"] = float(solve["phi"][2])
    return row, solve


def _pairwise_rows(solves: dict[str, dict[str, Any]], branch_by_seed: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(solves)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ca = np.asarray(solves[a]["out"]["centerline"], dtype=float)
            cb = np.asarray(solves[b]["out"]["centerline"], dtype=float)
            rows.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "branch_a": branch_by_seed[a],
                    "branch_b": branch_by_seed[b],
                    "same_branch_label": branch_by_seed[a] == branch_by_seed[b],
                    "centerline_rms": float(np.sqrt(np.mean(np.sum((ca - cb) ** 2, axis=1)))),
                    "tip_distance": float(np.linalg.norm(ca[-1] - cb[-1])),
                    "state_distance": float(np.linalg.norm(solves[a]["y"] - solves[b]["y"])),
                }
            )
    return rows


def _roundtrip(args: argparse.Namespace, failures: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alpha0 = float(args.alpha)
    y0 = _load_npz_y("experiment_records_ik_optimize/20260606103000_p2c19_ultrafine_lower_alpha_breakpoint/alpha_0.3750_ultrafine_lower_breakpoint.npz")
    rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for delta in (0.0002, -0.0002):
        sequence = [alpha0, alpha0 + delta, alpha0]
        y = y0.copy()
        centers: dict[str, np.ndarray] = {}
        for i, alpha in enumerate(sequence):
            params, I0, target_p, funcs = _case(alpha, args)
            solve = p2c15._solve(
                stage=f"roundtrip_{delta:+.4f}_{i}",
                y0=y,
                I=I0,
                funcs=funcs,
                target_p=target_p,
                slack_scale=args.slack_scale,
                max_iter=args.max_iter,
            )
            load = p2c15._audit_load(params, I0, solve["y"])
            y = solve["y"]
            centers[f"step_{i}_{alpha:.4f}"] = np.asarray(solve["out"]["centerline"], dtype=float)
            row = {
                "delta": delta,
                "step": i,
                "alpha": alpha,
                **{k: v for k, v in solve.items() if k not in {"y", "out", "phi"}},
                **load,
            }
            row["baseline_gate"] = _baseline_gate(row)
            rows.append(row)
        start = centers[f"step_0_{alpha0:.4f}"]
        end = centers[f"step_2_{alpha0:.4f}"]
        rms = float(np.sqrt(np.mean(np.sum((start - end) ** 2, axis=1))))
        tip = float(np.linalg.norm(start[-1] - end[-1]))
        br = {"delta": delta, "roundtrip_centerline_rms": rms, "roundtrip_tip_distance": tip}
        branch_rows.append(br)
        if rms > 1.0e-3:
            _append_jsonl(failures, {"ts": time.time(), "failure": "roundtrip_branch_drift", "evidence": br})
    return rows, branch_rows


def _plot(seed_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], roundtrip_rows: list[dict[str, Any]]) -> None:
    colors = {"physical_branch": "green", "nonphysical_or_unresolved": "red"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    labels = [row["seed_name"] for row in seed_rows]
    x = np.arange(len(labels))
    for ax, key, ylabel, gate in [
        (axes[0], "stationarity_norm", "一阶平稳性", 1.0e-5),
        (axes[1], "max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
        (axes[2], "position_error_m", "末端位置误差 m", 0.05 * 0.11),
    ]:
        ax.bar(x, [row[key] for row in seed_rows], color=[colors[row["branch_label"]] for row in seed_rows])
        ax.axhline(gate, color="black", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_branch_gate_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in seed_rows:
        ax.scatter(row["tip_x"], row["tip_y"], color=colors[row["branch_label"]], s=55)
        ax.text(row["tip_x"], row["tip_y"], row["seed_name"], fontsize=6, rotation=25)
    ax.set_xlabel("tip x (m)")
    ax.set_ylabel("tip y (m)")
    ax.set_title("alpha=0.3750 多初值末端聚类")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "seed_tip_cluster_xy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    same = [row for row in pair_rows if row["same_branch_label"]]
    diff = [row for row in pair_rows if not row["same_branch_label"]]
    ax.hist([row["centerline_rms"] for row in same], bins=20, alpha=0.7, label="同类分支")
    ax.hist([row["centerline_rms"] for row in diff], bins=20, alpha=0.7, label="异类分支")
    ax.set_xlabel("中心线均方根差异 m")
    ax.set_ylabel("样本对数量")
    ax.set_title("多初值分支间中心线差异")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_centerline_rms_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(row["delta"]) for row in roundtrip_rows], [row["roundtrip_centerline_rms"] for row in roundtrip_rows])
    ax.axhline(1.0e-3, color="r", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("回环中心线差异 m")
    ax.set_xlabel("delta alpha")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roundtrip_centerline_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3750)
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

    seed_rows: list[dict[str, Any]] = []
    solves: dict[str, dict[str, Any]] = {}
    for seed_name, seed_class, y0 in _candidate_seeds(args.alpha, args):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "seed_solve", "seed": seed_name})
        row, solve = _solve_seed(args.alpha, seed_name, seed_class, y0, args)
        seed_rows.append(row)
        solves[seed_name] = solve
        if not row["baseline_gate"]:
            _append_jsonl(failures, {"ts": time.time(), "failure": "seed_failed_baseline_gate", "evidence": row})
    branch_by_seed = {row["seed_name"]: row["branch_label"] for row in seed_rows}
    pair_rows = _pairwise_rows(solves, branch_by_seed)
    roundtrip_stage_rows, roundtrip_rows = _roundtrip(args, failures)
    _plot(seed_rows, pair_rows, roundtrip_rows)

    _write_csv(OUT_DIR / "seed_rows.csv", seed_rows)
    _write_csv(OUT_DIR / "pairwise_rows.csv", pair_rows)
    _write_csv(OUT_DIR / "roundtrip_stage_rows.csv", roundtrip_stage_rows)
    _write_csv(OUT_DIR / "roundtrip_rows.csv", roundtrip_rows)
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
            "seed_count": len(seed_rows),
            "physical_branch_count": sum(row["branch_label"] == "physical_branch" for row in seed_rows),
            "nonphysical_or_unresolved_count": sum(row["branch_label"] != "physical_branch" for row in seed_rows),
            "strict_independent_physical_count": sum(
                row["branch_label"] == "physical_branch" and row["seed_class"] == "strict_independent" for row in seed_rows
            ),
            "continuation_physical_count": sum(
                row["branch_label"] == "physical_branch" and row["seed_class"] == "path_continuation" for row in seed_rows
            ),
            "roundtrip_count": len(roundtrip_rows),
            "roundtrip_pass_count": sum(row["roundtrip_centerline_rms"] <= 1.0e-3 for row in roundtrip_rows),
        },
        "seed_rows": seed_rows,
        "roundtrip_rows": roundtrip_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
