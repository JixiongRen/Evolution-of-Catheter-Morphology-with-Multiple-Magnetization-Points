"""Run a lightweight fixed-state FD audit for the P2c.1 best candidate."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_kinematics.ik_observations import axis_from_quat

import run_p2c1_convergence as p2c1

jax.config.update("jax_enable_x64", True)


def _task_from_oracle(oracle: dict[str, Any], task: str) -> np.ndarray:
    sites = oracle["sites"]
    if task == "position":
        return np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float).reshape(3)
    if task == "axis":
        q = np.asarray(jax.device_get(sites["rigid_mid_Q_wxyz"]), dtype=float)[-1]
        return np.asarray(jax.device_get(axis_from_quat(jnp.asarray(q, dtype=jnp.float64))), dtype=float).reshape(3)
    raise ValueError(f"unknown task: {task}")


def _task_from_shooting(params: Any, y: np.ndarray, task: str) -> np.ndarray:
    out = p2c1._forward_modes(params, y, "segment_wrench", steps_per_flex=4)
    if task == "position":
        return np.asarray(out["tip_p"], dtype=float).reshape(3)
    if task == "axis":
        q = np.asarray(out["rigid_mid_q"], dtype=float)[-1]
        return np.asarray(jax.device_get(axis_from_quat(jnp.asarray(q, dtype=jnp.float64))), dtype=float).reshape(3)
    raise ValueError(f"unknown task: {task}")


def main() -> None:
    t0 = time.perf_counter()
    y = np.load(OUT_DIR / "M111" / "c3_segment_wrench_18d" / "shooting_state_y.npy")
    engine = p2c1.p2c._build_engine((1, 1, 1), max_iter=120, tol=1e-5)
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    base = p2c1.p2c._solve_oracle(engine, I0, z0=None)
    active_index = 5
    eps_values = [0.02, 0.05]
    tasks = ["position", "axis"]
    rows: list[dict[str, Any]] = []
    cache: dict[float, dict[str, Any]] = {}

    def oracle_at(delta_i: float) -> dict[str, Any]:
        if delta_i not in cache:
            d = np.zeros_like(I0)
            d[active_index] = delta_i
            cache[delta_i] = p2c1.p2c._solve_oracle(engine, I0 + d, z0=base["z"])
        return cache[delta_i]

    for eps in eps_values:
        plus = oracle_at(float(eps))
        minus = oracle_at(-float(eps))
        for task in tasks:
            fd_oracle = (_task_from_oracle(plus, task) - _task_from_oracle(minus, task)) / (2.0 * eps)
            fd_shooting = (
                _task_from_shooting(plus["params"], y, task)
                - _task_from_shooting(minus["params"], y, task)
            ) / (2.0 * eps)
            diff = fd_shooting - fd_oracle
            oracle_norm = max(float(np.linalg.norm(fd_oracle)), 1e-12)
            rows.append(
                {
                    "mode": "fixed_state",
                    "candidate": "c3_segment_wrench_18d",
                    "active_current_index": active_index,
                    "eps": float(eps),
                    "task": task,
                    "oracle_fd_norm": float(np.linalg.norm(fd_oracle)),
                    "shooting_fd_norm": float(np.linalg.norm(fd_shooting)),
                    "difference_norm": float(np.linalg.norm(diff)),
                    "relative_error": float(np.linalg.norm(diff) / oracle_norm),
                }
            )

    payload = {
        "metadata": {
            "mode": "fixed_state",
            "candidate": "c3_segment_wrench_18d",
            "active_current_index": active_index,
            "eps_values": eps_values,
            "tasks": tasks,
            "wall_s": float(time.perf_counter() - t0),
            "base_oracle_normE": float(base["stats"].normE),
            "base_oracle_ok": bool(base["ok"]),
            "note": "Fixed-state audit only. It checks local task response with the best shooting state held constant.",
        },
        "rows": rows,
    }
    (OUT_DIR / "fd_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "fd_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    labels = [f"{row['task']} eps={row['eps']}" for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(rows)), [row["relative_error"] for row in rows])
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Fixed-state FD audit")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fd_audit_errors.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
