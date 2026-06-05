"""Decompose rigid magnetic loads for saved branch shooting states."""

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
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
P2C10P_DIR = ROOT / "experiment_records_ik_optimize" / "20260605074411_p2c10p_branch_load_propagation_audit"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C10I_DIR, P2C10P_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from external_wrench_nondim import compute_external_wrench_total_rigid
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_terminal_shooting_jax import _advance_rigid, _integrate_flex, initial_state, normalize_state

import run_p2c10i_boundary_candidate_strict_gate as p2c10i

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
BRANCH_NAMES = ("oracle", "half_oracle", "zero")


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


def _parse_candidates(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))
    return specs


def _load_branch_states(case_name: str, alpha: float) -> dict[str, np.ndarray]:
    path = P2C10P_DIR / f"{case_name}_alpha_{alpha:.2f}_branches.npz"
    data = np.load(path)
    return {name: np.asarray(data[f"{name}_y"], dtype=float) for name in BRANCH_NAMES}


def _build_params(case_name: str, alpha: float) -> tuple[Any, np.ndarray]:
    target_case = p2c10i._case_by_name(case_name)
    case = p2c10i._scaled_case(target_case, alpha)
    I0 = np.asarray(case.currents, dtype=float)
    engine = p2c10i.p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    oracle = p2c10i.p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    params = with_coil_currents(oracle["params"], jnp.asarray(I0, dtype=jnp.float64).reshape(8))
    return params, I0


def _magnetic_load_rows(params: Any, y: np.ndarray, *, case_name: str, alpha: float, branch: str, steps_per_flex: int) -> list[dict[str, Any]]:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    rows: list[dict[str, Any]] = []
    f_ref = max(float(params.scales.F_ref), 1.0e-12)
    m_ref = max(float(params.scales.M_ref), 1.0e-12)
    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = normalize_state(x)
        x, _flex_points = _integrate_flex(params, x, i, steps_per_flex)
        x_prox = x
        rigid = params.rigid[i]
        f_mag, tau_mag = compute_external_wrench_total_rigid(
            x_proximal=x_prox,
            rigid_length=float(rigid.length),
            gravity=None,
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )
        f_grav_total, tau_grav_total = compute_external_wrench_total_rigid(
            x_proximal=x_prox,
            rigid_length=float(rigid.length),
            gravity=params.gravity_rigid_list[i],
            magnetic_model=None,
            magnet_params=None,
            coil_currents=None,
        )
        f_ext = jnp.asarray(params.f_ext_list[i], dtype=jnp.float64).reshape(3)
        tau_ext = jnp.asarray(params.tau_ext_list[i], dtype=jnp.float64).reshape(3)
        f_mag_np = np.asarray(jax.device_get(f_mag), dtype=float)
        tau_mag_np = np.asarray(jax.device_get(tau_mag), dtype=float)
        f_grav_np = np.asarray(jax.device_get(f_grav_total), dtype=float)
        tau_grav_np = np.asarray(jax.device_get(tau_grav_total), dtype=float)
        f_ext_np = np.asarray(jax.device_get(f_ext), dtype=float)
        tau_ext_np = np.asarray(jax.device_get(tau_ext), dtype=float)
        x_dist, _mid_p, _mid_q = _advance_rigid(params, x, i)
        dist_wrench = np.concatenate(
            [
                np.asarray(jax.device_get(x_dist[7:10]), dtype=float) / f_ref,
                np.asarray(jax.device_get(x_dist[10:13]), dtype=float) / m_ref,
            ]
        )
        rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "branch": branch,
                "rigid_index": i,
                "prox_p": np.asarray(jax.device_get(x_prox[0:3]), dtype=float).tolist(),
                "prox_q": np.asarray(jax.device_get(x_prox[3:7]), dtype=float).tolist(),
                "mag_force": f_mag_np.tolist(),
                "mag_torque": tau_mag_np.tolist(),
                "gravity_force": f_grav_np.tolist(),
                "gravity_torque": tau_grav_np.tolist(),
                "explicit_force": f_ext_np.tolist(),
                "explicit_torque": tau_ext_np.tolist(),
                "mag_force_norm": float(np.linalg.norm(f_mag_np)),
                "mag_torque_norm": float(np.linalg.norm(tau_mag_np)),
                "gravity_force_norm": float(np.linalg.norm(f_grav_np)),
                "gravity_torque_norm": float(np.linalg.norm(tau_grav_np)),
                "distal_wrench": dist_wrench.tolist(),
                "distal_force_norm_nd": float(np.linalg.norm(dist_wrench[0:3])),
                "distal_moment_norm_nd": float(np.linalg.norm(dist_wrench[3:6])),
            }
        )
        x = x_dist
    return rows


def _pairwise_load_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_key: dict[tuple[str, float, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["case"], row["alpha"], row["rigid_index"])
        by_key.setdefault(key, {})[row["branch"]] = row
    for (case, alpha, rigid_index), branch_map in by_key.items():
        for a, b in combinations(sorted(branch_map), 2):
            ra = branch_map[a]
            rb = branch_map[b]
            f_delta = np.asarray(ra["mag_force"]) - np.asarray(rb["mag_force"])
            tau_delta = np.asarray(ra["mag_torque"]) - np.asarray(rb["mag_torque"])
            prox_delta = np.asarray(ra["prox_p"]) - np.asarray(rb["prox_p"])
            dist_delta = np.asarray(ra["distal_wrench"]) - np.asarray(rb["distal_wrench"])
            out.append(
                {
                    "case": case,
                    "alpha": alpha,
                    "rigid_index": rigid_index,
                    "branch_a": a,
                    "branch_b": b,
                    "prox_position_distance": float(np.linalg.norm(prox_delta)),
                    "mag_force_distance": float(np.linalg.norm(f_delta)),
                    "mag_torque_distance": float(np.linalg.norm(tau_delta)),
                    "distal_wrench_distance": float(np.linalg.norm(dist_delta)),
                    "distal_force_distance_nd": float(np.linalg.norm(dist_delta[0:3])),
                    "distal_moment_distance_nd": float(np.linalg.norm(dist_delta[3:6])),
                }
            )
    return out


def _plot_load_norms(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f}\n{row['branch']} r{row['rigid_index']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.45 * len(rows)), 7.5), sharex=True)
    axes[0].bar(x, [row["mag_force_norm"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("magnetic force norm (N)")
    axes[1].bar(x, [row["mag_torque_norm"] for row in rows], color="#ff7f0e")
    axes[1].set_ylabel("magnetic torque norm (N*m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "magnetic_load_norms.png", dpi=180)
    plt.close(fig)


def _plot_pairwise(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['case']}@{row['alpha']:.2f} r{row['rigid_index']}\n{row['branch_a']} vs {row['branch_b']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.55 * len(rows)), 9.5), sharex=True)
    axes[0].bar(x, [row["mag_force_distance"] for row in rows], color="#1f77b4")
    axes[0].set_ylabel("magnetic force distance")
    axes[1].bar(x, [row["mag_torque_distance"] for row in rows], color="#ff7f0e")
    axes[1].set_ylabel("magnetic torque distance")
    axes[2].bar(x, [row["distal_wrench_distance"] for row in rows], color="#2ca02c")
    axes[2].set_ylabel("distal wrench distance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_magnetic_load_distances.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00")
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()

    load_rows: list[dict[str, Any]] = []
    for case_name, alpha in _parse_candidates(args.candidates):
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build_params", "case": case_name, "alpha": alpha})
        params, _I0 = _build_params(case_name, alpha)
        states = _load_branch_states(case_name, alpha)
        for branch, y in states.items():
            load_rows.extend(
                _magnetic_load_rows(
                    params,
                    y,
                    case_name=case_name,
                    alpha=alpha,
                    branch=branch,
                    steps_per_flex=args.steps_per_flex,
                )
            )
    pair_rows = _pairwise_load_rows(load_rows)
    _plot_load_norms(load_rows)
    _plot_pairwise(pair_rows)

    case_summaries = []
    for case_name, alpha in _parse_candidates(args.candidates):
        subset = [row for row in pair_rows if row["case"] == case_name and abs(row["alpha"] - alpha) < 1.0e-12]
        case_summaries.append(
            {
                "case": case_name,
                "alpha": alpha,
                "max_mag_force_distance": max([row["mag_force_distance"] for row in subset], default=0.0),
                "max_mag_torque_distance": max([row["mag_torque_distance"] for row in subset], default=0.0),
                "max_distal_wrench_distance": max([row["distal_wrench_distance"] for row in subset], default=0.0),
                "max_prox_position_distance": max([row["prox_position_distance"] for row in subset], default=0.0),
            }
        )
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "steps_per_flex": args.steps_per_flex,
        },
        "classification": "显式刚性磁体磁载荷分解审计",
        "case_summaries": case_summaries,
        "load_rows": load_rows,
        "pair_rows": pair_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "magnetic_load_summary.csv", load_rows)
    _write_csv(OUT_DIR / "pairwise_magnetic_load_summary.csv", pair_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
