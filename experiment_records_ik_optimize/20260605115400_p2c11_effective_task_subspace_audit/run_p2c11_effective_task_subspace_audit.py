"""Audit effective task subspaces for position shooting candidates."""

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
P2C10W_DIR = ROOT / "experiment_records_ik_optimize" / "20260605152000_p2c10w_task_reduction_phase_map"
P2C10Y_DIR = ROOT / "experiment_records_ik_optimize" / "20260605211000_p2c10y_position_refinement_step_audit"
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C10Y_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

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


def _load_linearity(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    path = P2C10Y_DIR / f"alpha_{spec['alpha']:.2f}_eps_{spec['eps']:.3f}_iter_{spec['max_iter']}.npz"
    data = np.load(path)
    return {
        "phi0": np.asarray(data["phi0"], dtype=float),
        "phi_plus": np.asarray(data["phi_plus"], dtype=float),
        "phi_minus": np.asarray(data["phi_minus"], dtype=float),
        "sensitivity": np.asarray(data["sensitivity"], dtype=float),
    }


def _make_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    position_scale: float,
    boundary_scale: float,
    slack_scale: float,
    steps_per_flex: int,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c10w.p2c10t.forward_interface_slack(
            params_for_I(I),
            y,
            variant="wrench",
            steps_per_flex=steps_per_flex,
        )

    def position_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return float(position_scale) * (output_yI(y, I).tip_p - target_p_j)

    def boundary_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return float(boundary_scale) * output_yI(y, I).boundary

    def slack_residual(y: jnp.ndarray, _I: jnp.ndarray) -> jnp.ndarray:
        return float(slack_scale) * jnp.asarray(y, dtype=jnp.float64).reshape(-1)[6:]

    def total_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [
                position_residual(y, I).reshape(-1),
                boundary_residual(y, I).reshape(-1),
                slack_residual(y, I).reshape(-1),
            ],
            axis=0,
        )

    return {
        "position": jax.jit(position_residual),
        "boundary": jax.jit(boundary_residual),
        "slack": jax.jit(slack_residual),
        "total": jax.jit(total_residual),
        "J_position": jax.jit(jax.jacfwd(position_residual, argnums=0)),
        "J_total": jax.jit(jax.jacfwd(total_residual, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _relative_linearity_for_projection(linearity: dict[str, np.ndarray], basis: np.ndarray) -> tuple[float, float]:
    if basis.size == 0:
        return float("nan"), float("nan")
    phi0 = linearity["phi0"]
    plus = linearity["phi_plus"]
    minus = linearity["phi_minus"]
    sensitivity = linearity["sensitivity"]
    scale = max(float(np.linalg.norm(basis.T @ phi0)), 1.0e-12)
    eps = 1.0
    pred_plus = basis.T @ (phi0 + sensitivity * eps)
    pred_minus = basis.T @ (phi0 - sensitivity * eps)
    err_plus = float(np.linalg.norm((basis.T @ plus) - pred_plus) / scale)
    err_minus = float(np.linalg.norm((basis.T @ minus) - pred_minus) / scale)
    return err_plus, err_minus


def _analyze_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", spec["alpha"])
    funcs = _make_functions(
        params_base=params,
        target_p=target_p,
        position_scale=120.0,
        boundary_scale=0.5,
        slack_scale=0.05,
        steps_per_flex=4,
    )
    y = _load_y(spec)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64)
    _ = funcs["total"](y_j, I_j).block_until_ready()
    r_pos = np.asarray(jax.device_get(funcs["position"](y_j, I_j)), dtype=float).reshape(-1)
    r_total = np.asarray(jax.device_get(funcs["total"](y_j, I_j)), dtype=float).reshape(-1)
    J_pos = np.asarray(jax.device_get(funcs["J_position"](y_j, I_j)), dtype=float)
    J_total = np.asarray(jax.device_get(funcs["J_total"](y_j, I_j)), dtype=float)
    g_pos = J_pos.T @ r_pos
    g_total = J_total.T @ r_total
    U, svals, Vt = np.linalg.svd(J_total, full_matrices=False)
    coeff_total = Vt @ g_total
    coeff_pos = Vt @ g_pos
    total_energy = float(np.sum(coeff_total**2))
    pos_energy = float(np.sum(coeff_pos**2))
    stable_threshold = 1.0e-4 * float(svals[0])
    stable_mask = svals >= stable_threshold
    weak_mask = ~stable_mask
    stable_count = int(np.sum(stable_mask))
    weak_count = int(np.sum(weak_mask))
    weak_total_fraction = float(np.sum(coeff_total[weak_mask] ** 2) / max(total_energy, 1.0e-30))
    weak_pos_fraction = float(np.sum(coeff_pos[weak_mask] ** 2) / max(pos_energy, 1.0e-30))

    response_U, response_s, response_Vt = np.linalg.svd(J_pos, full_matrices=False)
    linearity = _load_linearity(spec)
    rows: list[dict[str, Any]] = []
    for rank in (1, 2, 3):
        basis = response_U[:, :rank]
        err_plus, err_minus = _relative_linearity_for_projection(linearity, basis)
        projected_sens = basis @ (basis.T @ linearity["sensitivity"])
        rows.append(
            {
                **spec,
                "projection_rank": rank,
                "position_response_singular_value": float(response_s[rank - 1]) if rank <= response_s.size else float("nan"),
                "projected_sensitivity_norm": float(np.linalg.norm(projected_sens)),
                "projected_sensitivity_fraction": float(
                    np.linalg.norm(projected_sens) / max(float(np.linalg.norm(linearity["sensitivity"])), 1.0e-30)
                ),
                "projected_linearity_error_plus": err_plus,
                "projected_linearity_error_minus": err_minus,
            }
        )

    summary = {
        **spec,
        "total_stationarity_norm": float(np.linalg.norm(g_total)),
        "position_stationarity_norm": float(np.linalg.norm(g_pos)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
        "jacobian_rank": int(np.linalg.matrix_rank(J_total)),
        "singular_max": float(svals[0]),
        "singular_min": float(svals[-1]),
        "stable_threshold": stable_threshold,
        "stable_direction_count": stable_count,
        "weak_direction_count": weak_count,
        "weak_total_gradient_energy_fraction": weak_total_fraction,
        "weak_position_gradient_energy_fraction": weak_pos_fraction,
        "top1_total_gradient_energy_fraction": float(coeff_total[0] ** 2 / max(total_energy, 1.0e-30)),
        "top3_total_gradient_energy_fraction": float(np.sum(coeff_total[:3] ** 2) / max(total_energy, 1.0e-30)),
        "top1_position_gradient_energy_fraction": float(coeff_pos[0] ** 2 / max(pos_energy, 1.0e-30)),
        "top3_position_gradient_energy_fraction": float(np.sum(coeff_pos[:3] ** 2) / max(pos_energy, 1.0e-30)),
        "position_response_condition": float(response_s[0] / max(float(response_s[-1]), np.finfo(float).tiny)),
        "position_response_singular_values": response_s.tolist(),
        "total_singular_values": svals.tolist(),
    }
    return summary, rows


def _plot(summary_rows: list[dict[str, Any]], projection_rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in summary_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].bar(x, [row["jacobian_condition"] for row in summary_rows])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("condition")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["weak_total_gradient_energy_fraction"] for row in summary_rows])
    axes[1].set_ylabel("weak energy fraction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, [row["position_response_condition"] for row in summary_rows])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("position response condition")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "subspace_condition_overview.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    for j, rank in enumerate((1, 2, 3)):
        vals = [
            next(r for r in projection_rows if r["label"] == label and r["projection_rank"] == rank)[
                "projected_sensitivity_fraction"
            ]
            for label in labels
        ]
        ax.bar(x + (j - 1) * width, vals, width=width, label=f"rank {rank}")
    ax.set_ylabel("sensitivity fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "projected_sensitivity_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for row in summary_rows:
        vals = np.asarray(row["total_singular_values"], dtype=float)
        ax.plot(np.arange(1, len(vals) + 1), vals / max(vals[0], 1.0e-30), marker="o", label=row["label"])
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "normalized_singular_spectra.png", dpi=180)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    summary_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "analyze", **spec})
        summary, projection = _analyze_spec(spec)
        summary_rows.append(summary)
        projection_rows.extend(projection)
        if summary["jacobian_condition"] > 1.0e8 or summary["weak_total_gradient_energy_fraction"] > 0.1:
            _append_jsonl(
                failures,
                {
                    "ts": time.time(),
                    "failure": "subspace_pathology_detected",
                    "reasons": ["局部雅可比矩阵病态或弱方向梯度占比过高"],
                    "evidence": summary,
                },
            )

    _write_csv(OUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUT_DIR / "projection_rows.csv", projection_rows)
    _plot(summary_rows, projection_rows)
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
            "condition_over_1e8_count": sum(row["jacobian_condition"] > 1.0e8 for row in summary_rows),
            "weak_energy_over_0p1_count": sum(row["weak_total_gradient_energy_fraction"] > 0.1 for row in summary_rows),
        },
        "case_summaries": summary_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
