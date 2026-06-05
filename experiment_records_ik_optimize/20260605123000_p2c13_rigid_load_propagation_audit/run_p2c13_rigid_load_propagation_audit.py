"""Audit rigid magnetic load propagation against interface slack."""

from __future__ import annotations

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
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C10T_DIR = ROOT / "experiment_records_ik_optimize" / "20260605102000_p2c10t_interface_slack_shooting"
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10Y_DIR = ROOT / "experiment_records_ik_optimize" / "20260605211000_p2c10y_position_refinement_step_audit"
for path in (ROOT, FK_OPT_DIR, P2C10T_DIR, P2C10W_DIR, P2C10Y_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.external_wrench_nondim import compute_external_wrench_total_rigid
from inverse_kinematics.ik_terminal_shooting_jax import _advance_rigid, _integrate_flex, initial_state, normalize_state

import run_p2c10t_interface_slack_shooting as p2c10t
import run_p2c10w_task_reduction_phase_map as p2c10w

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


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {"alpha": 0.40, "eps": 0.005, "max_iter": 100, "label": "a040_best_stationarity"},
        {"alpha": 0.40, "eps": 0.005, "max_iter": 250, "label": "a040_best_branch"},
        {"alpha": 0.45, "eps": 0.005, "max_iter": 500, "label": "a045_best_stationarity"},
        {"alpha": 0.45, "eps": 0.010, "max_iter": 250, "label": "a045_best_branch"},
    ]


def _load_y(spec: dict[str, Any]) -> np.ndarray:
    path = P2C10Y_DIR / f"alpha_{spec['alpha']:.2f}_eps_{spec['eps']:.3f}_iter_{spec['max_iter']}.npz"
    return np.asarray(np.load(path)["baseline_y"], dtype=float)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-30 or nb <= 1.0e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _rigid_wrench(params: Any, x_prox: jnp.ndarray, i: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    f_total, tau_total = compute_external_wrench_total_rigid(
        x_proximal=x_prox,
        rigid_length=float(params.rigid[i].length),
        gravity=params.gravity_rigid_list[i],
        magnetic_model=params.magnetic_model,
        magnet_params=params.magnet_params_list[i],
        coil_currents=params.coil_currents,
    )
    f_total = f_total + jnp.asarray(params.f_ext_list[i], dtype=jnp.float64).reshape(3)
    tau_total = tau_total + jnp.asarray(params.tau_ext_list[i], dtype=jnp.float64).reshape(3)
    return f_total, tau_total


def _trace_state(params: Any, y: np.ndarray, steps_per_flex: int = 4) -> dict[str, Any]:
    n_interfaces = max(len(params.flex) - 1, 0)
    base, force_slack, torque_slack = p2c10t._split_state(jnp.asarray(y, dtype=jnp.float64), "wrench", n_interfaces)
    x = initial_state(params, base)
    rows = []
    for i in range(len(params.flex)):
        x, _flex_points = _integrate_flex(params, x, i, steps_per_flex)
        x_rigid_prox = normalize_state(x)
        f_total, tau_total = _rigid_wrench(params, x_rigid_prox, i)
        x_rigid_dist, _mid_p, _mid_q = _advance_rigid(params, x_rigid_prox, i)
        natural_delta_f = x_rigid_dist[7:10] - x_rigid_prox[7:10]
        natural_delta_tau = x_rigid_dist[10:13] - x_rigid_prox[10:13]
        row = {
            "segment": i,
            "rigid_force_norm_dim": float(jnp.linalg.norm(f_total)),
            "rigid_torque_norm_dim": float(jnp.linalg.norm(tau_total)),
            "natural_force_delta_norm_dim": float(jnp.linalg.norm(natural_delta_f)),
            "natural_torque_delta_norm_dim": float(jnp.linalg.norm(natural_delta_tau)),
            "prox_force_norm_dim": float(jnp.linalg.norm(x_rigid_prox[7:10])),
            "prox_torque_norm_dim": float(jnp.linalg.norm(x_rigid_prox[10:13])),
            "dist_force_norm_dim": float(jnp.linalg.norm(x_rigid_dist[7:10])),
            "dist_torque_norm_dim": float(jnp.linalg.norm(x_rigid_dist[10:13])),
            "rigid_force": np.asarray(jax.device_get(f_total), dtype=float),
            "rigid_torque": np.asarray(jax.device_get(tau_total), dtype=float),
            "natural_delta_force": np.asarray(jax.device_get(natural_delta_f), dtype=float),
            "natural_delta_torque": np.asarray(jax.device_get(natural_delta_tau), dtype=float),
        }
        if i < n_interfaces:
            slack_f_dim = force_slack[i] * float(params.scales.F_ref)
            slack_tau_dim = torque_slack[i] * float(params.scales.M_ref)
            row.update(
                {
                    "interface_index": i,
                    "slack_force_norm_dim": float(jnp.linalg.norm(slack_f_dim)),
                    "slack_torque_norm_dim": float(jnp.linalg.norm(slack_tau_dim)),
                    "slack_force_over_rigid_force": float(jnp.linalg.norm(slack_f_dim) / jnp.maximum(jnp.linalg.norm(f_total), 1.0e-30)),
                    "slack_torque_over_rigid_torque": float(
                        jnp.linalg.norm(slack_tau_dim) / jnp.maximum(jnp.linalg.norm(tau_total), 1.0e-30)
                    ),
                    "slack_force_cosine_rigid_force": _cosine(np.asarray(slack_f_dim, dtype=float), np.asarray(f_total, dtype=float)),
                    "slack_torque_cosine_rigid_torque": _cosine(np.asarray(slack_tau_dim, dtype=float), np.asarray(tau_total, dtype=float)),
                    "slack_force_cosine_natural_delta": _cosine(np.asarray(slack_f_dim, dtype=float), np.asarray(natural_delta_f, dtype=float)),
                    "slack_torque_cosine_natural_delta": _cosine(np.asarray(slack_tau_dim, dtype=float), np.asarray(natural_delta_tau, dtype=float)),
                    "slack_force": np.asarray(jax.device_get(slack_f_dim), dtype=float),
                    "slack_torque": np.asarray(jax.device_get(slack_tau_dim), dtype=float),
                }
            )
            x_rigid_dist = x_rigid_dist.at[7:10].add(slack_f_dim)
            x_rigid_dist = x_rigid_dist.at[10:13].add(slack_tau_dim)
        rows.append(row)
        x = normalize_state(x_rigid_dist)
    boundary = np.asarray(
        jax.device_get(
            jnp.concatenate(
                [
                    x[7:10] / jnp.maximum(float(params.scales.F_ref), 1.0e-12),
                    x[10:13] / jnp.maximum(float(params.scales.M_ref), 1.0e-12),
                ],
                axis=0,
            )
        ),
        dtype=float,
    )
    return {"rows": rows, "boundary": boundary}


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            for i, item in enumerate(value.reshape(-1)):
                out[f"{key}_{i}"] = float(item)
        else:
            out[key] = value
    return out


def _analyze_spec(spec: dict[str, Any], failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params, _I0, _target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", spec["alpha"])
    y = _load_y(spec)
    trace = _trace_state(params, y, steps_per_flex=4)
    rows = []
    for row in trace["rows"]:
        flat = {**spec, **_flatten_row(row)}
        if "interface_index" in flat:
            suspicious = bool(
                flat["slack_force_over_rigid_force"] > 5.0
                or flat["slack_torque_over_rigid_torque"] > 5.0
                or flat["slack_force_cosine_rigid_force"] < -0.5
                or flat["slack_torque_cosine_rigid_torque"] < -0.5
            )
            flat["load_mismatch_flag"] = suspicious
            if suspicious:
                _append_jsonl(
                    failures,
                    {
                        "ts": time.time(),
                        "failure": "interface_slack_load_mismatch",
                        "reasons": ["界面松弛量与刚段外载荷量级或方向不相容"],
                        "evidence": flat,
                    },
                )
        rows.append(flat)
    interface_rows = [row for row in rows if "interface_index" in row]
    summary = {
        **spec,
        "boundary_norm": float(np.linalg.norm(trace["boundary"])),
        "interface_count": len(interface_rows),
        "max_slack_force_over_rigid_force": max((row["slack_force_over_rigid_force"] for row in interface_rows), default=float("nan")),
        "max_slack_torque_over_rigid_torque": max((row["slack_torque_over_rigid_torque"] for row in interface_rows), default=float("nan")),
        "min_slack_force_cosine_rigid_force": min((row["slack_force_cosine_rigid_force"] for row in interface_rows), default=float("nan")),
        "min_slack_torque_cosine_rigid_torque": min((row["slack_torque_cosine_rigid_torque"] for row in interface_rows), default=float("nan")),
        "mismatch_count": sum(bool(row.get("load_mismatch_flag", False)) for row in interface_rows),
    }
    return summary, rows


def _plot(summary_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].bar(x, [row["max_slack_force_over_rigid_force"] for row in summary_rows])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max force slack/load")
    axes[1].bar(x, [row["max_slack_torque_over_rigid_torque"] for row in summary_rows])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max torque slack/load")
    axes[2].bar(x, [row["boundary_norm"] for row in summary_rows])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("boundary norm")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "load_slack_ratio_overview.png", dpi=180)
    plt.close(fig)

    interface_rows = [row for row in segment_rows if "interface_index" in row]
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in labels:
        sub = [row for row in interface_rows if row["label"] == label]
        ax.scatter(
            [row["slack_force_cosine_rigid_force"] for row in sub],
            [row["slack_torque_cosine_rigid_torque"] for row in sub],
            label=label,
            s=60,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("force cosine")
    ax.set_ylabel("torque cosine")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slack_load_direction_cosines.png", dpi=180)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "analyze", **spec})
        summary, rows = _analyze_spec(spec, failures)
        summary_rows.append(summary)
        segment_rows.extend(rows)
    _write_csv(OUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUT_DIR / "segment_rows.csv", segment_rows)
    _plot(summary_rows, segment_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "overall": {
            "case_count": len(summary_rows),
            "interface_count": sum(row["interface_count"] for row in summary_rows),
            "mismatch_count": sum(row["mismatch_count"] for row in summary_rows),
        },
        "case_summaries": summary_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
