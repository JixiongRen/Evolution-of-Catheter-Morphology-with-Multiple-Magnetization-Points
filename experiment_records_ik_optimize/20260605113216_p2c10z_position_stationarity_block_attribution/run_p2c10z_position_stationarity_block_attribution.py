"""Attribute position-task stationarity failures to residual blocks."""

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
P2C10X_DIR = ROOT / "experiment_records_ik_optimize" / "20260605202000_p2c10x_position_alpha_linearity_audit"
P2C10Y_DIR = ROOT / "experiment_records_ik_optimize" / "20260605211000_p2c10y_position_refinement_step_audit"
for path in (ROOT, FK_OPT_DIR, P2C10W_DIR, P2C10X_DIR, P2C10Y_DIR):
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-30 or nb <= 1.0e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {"alpha": 0.40, "eps": 0.005, "max_iter": 100, "label": "a040_best_stationarity"},
        {"alpha": 0.40, "eps": 0.005, "max_iter": 250, "label": "a040_best_branch"},
        {"alpha": 0.45, "eps": 0.005, "max_iter": 500, "label": "a045_best_stationarity"},
        {"alpha": 0.45, "eps": 0.010, "max_iter": 250, "label": "a045_best_branch"},
    ]


def _load_y(spec: dict[str, Any]) -> np.ndarray:
    path = P2C10Y_DIR / f"alpha_{spec['alpha']:.2f}_eps_{spec['eps']:.3f}_iter_{spec['max_iter']}.npz"
    data = np.load(path)
    return np.asarray(data["baseline_y"], dtype=float)


def _make_block_functions(
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

    def residual_block(y: jnp.ndarray, I: jnp.ndarray, block: str) -> jnp.ndarray:
        out = output_yI(y, I)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        if block == "position":
            return float(position_scale) * (out.tip_p - target_p_j)
        if block == "boundary":
            return float(boundary_scale) * out.boundary
        if block == "slack":
            return float(slack_scale) * y[6:]
        raise ValueError(f"Unknown residual block: {block}")

    funcs: dict[str, Any] = {}
    for block_name in ("position", "boundary", "slack"):

        def block_func(y: jnp.ndarray, I: jnp.ndarray, *, _block: str = block_name) -> jnp.ndarray:
            return residual_block(y, I, _block)

        funcs[block_name] = {
            "residual": jax.jit(block_func),
            "jacobian": jax.jit(jax.jacfwd(block_func, argnums=0)),
        }

    def total_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [funcs[name]["residual"](y, I).reshape(-1) for name in ("position", "boundary", "slack")],
            axis=0,
        )

    funcs["total"] = {
        "residual": jax.jit(total_residual),
        "jacobian": jax.jit(jax.jacfwd(total_residual, argnums=0)),
    }
    funcs["output"] = jax.jit(output_yI)
    return funcs


def _analyze_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params, I0, target_p, _target_q, _y_ref, _oracle = p2c10w._build_case("n3_mixed_rand20", spec["alpha"])
    funcs = _make_block_functions(
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
    _ = funcs["total"]["residual"](y_j, I_j).block_until_ready()

    block_rows: list[dict[str, Any]] = []
    gradients: dict[str, np.ndarray] = {}
    residual_norms: dict[str, float] = {}
    for block in ("position", "boundary", "slack"):
        r = np.asarray(jax.device_get(funcs[block]["residual"](y_j, I_j)), dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs[block]["jacobian"](y_j, I_j)), dtype=float)
        g = J.T @ r
        residual_norms[block] = float(np.linalg.norm(r))
        gradients[block] = g
        block_rows.append(
            {
                **spec,
                "block": block,
                "residual_norm": float(np.linalg.norm(r)),
                "gradient_norm": float(np.linalg.norm(g)),
            }
        )

    total_r = np.asarray(jax.device_get(funcs["total"]["residual"](y_j, I_j)), dtype=float)
    total_J = np.asarray(jax.device_get(funcs["total"]["jacobian"](y_j, I_j)), dtype=float)
    total_g = total_J.T @ total_r
    total_norm = float(np.linalg.norm(total_g))
    svals = np.linalg.svd(total_J, compute_uv=False)
    sum_block_norms = sum(float(np.linalg.norm(gradients[name])) for name in gradients)
    cancellation_ratio = total_norm / max(sum_block_norms, 1.0e-30)

    for row in block_rows:
        g = gradients[row["block"]]
        row["cosine_to_total_gradient"] = _cosine(g, total_g)
        row["signed_projection_fraction"] = float(np.dot(g, total_g) / max(float(np.dot(total_g, total_g)), 1.0e-30))

    pair_rows = []
    names = list(gradients)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair_rows.append(
                {
                    **spec,
                    "block_a": a,
                    "block_b": b,
                    "gradient_cosine": _cosine(gradients[a], gradients[b]),
                }
            )

    out = p2c10w.output_to_numpy(funcs["output"](y_j, I_j))
    summary = {
        **spec,
        "position_residual_norm": residual_norms["position"],
        "boundary_residual_norm": residual_norms["boundary"],
        "slack_residual_norm": residual_norms["slack"],
        "total_residual_norm": float(np.linalg.norm(total_r)),
        "total_stationarity_norm": total_norm,
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(total_J)),
        "cancellation_ratio": cancellation_ratio,
        "dominant_gradient_block": max(block_rows, key=lambda row: row["gradient_norm"])["block"],
        "dominant_projection_block": max(block_rows, key=lambda row: abs(row["signed_projection_fraction"]))["block"],
        "terminal_position_error_m": float(np.linalg.norm(np.asarray(out["tip_p"], dtype=float).reshape(3) - target_p)),
        "physical_boundary_norm": float(np.linalg.norm(np.asarray(out["boundary"], dtype=float))),
        "antagonistic_pair_count": sum(1 for row in pair_rows if row["gradient_cosine"] < -0.2),
    }
    return summary, block_rows, pair_rows


def _plot(summary_rows: list[dict[str, Any]], block_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].bar(x, [row["total_stationarity_norm"] for row in summary_rows])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("stationarity")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["cancellation_ratio"] for row in summary_rows])
    axes[1].set_ylabel("gradient sum ratio")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(x, [row["jacobian_condition"] for row in summary_rows])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("condition")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stationarity_overview.png", dpi=180)
    plt.close(fig)

    blocks = ["position", "boundary", "slack"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    for j, block in enumerate(blocks):
        vals = []
        for spec in summary_rows:
            row = next(r for r in block_rows if r["label"] == spec["label"] and r["block"] == block)
            vals.append(row["gradient_norm"])
        ax.bar(x + (j - 1) * width, vals, width=width, label=block)
    ax.set_yscale("log")
    ax.set_ylabel("block gradient norm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_gradient_norms.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(summary_rows), figsize=(4 * len(summary_rows), 4), squeeze=False)
    for ax, spec in zip(axes[0], summary_rows):
        mat = np.eye(len(blocks))
        for row in pair_rows:
            if row["label"] != spec["label"]:
                continue
            i = blocks.index(row["block_a"])
            j = blocks.index(row["block_b"])
            mat[i, j] = mat[j, i] = row["gradient_cosine"]
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(spec["label"])
        ax.set_xticks(range(len(blocks)))
        ax.set_yticks(range(len(blocks)))
        ax.set_xticklabels(blocks, rotation=45, ha="right")
        ax.set_yticklabels(blocks)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72)
    fig.savefig(OUT_DIR / "block_gradient_cosines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        with heartbeat.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "phase": "analyze", **spec}, ensure_ascii=False) + "\n")
        summary, blocks, pairs = _analyze_spec(spec)
        summary_rows.append(summary)
        block_rows.extend(blocks)
        pair_rows.extend(pairs)
        if summary["total_stationarity_norm"] > 1.0e-5:
            with failures.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "failure": "stationarity_block_gate_failed",
                            "reasons": ["一阶平稳性残差未达到可发布门限"],
                            "evidence": summary,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    _write_csv(OUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUT_DIR / "block_rows.csv", block_rows)
    _write_csv(OUT_DIR / "pair_rows.csv", pair_rows)
    _plot(summary_rows, block_rows, pair_rows)
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
            "publishable_count": sum(row["total_stationarity_norm"] <= 1.0e-5 for row in summary_rows),
            "antagonistic_case_count": sum(row["antagonistic_pair_count"] > 0 for row in summary_rows),
        },
        "case_summaries": summary_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
