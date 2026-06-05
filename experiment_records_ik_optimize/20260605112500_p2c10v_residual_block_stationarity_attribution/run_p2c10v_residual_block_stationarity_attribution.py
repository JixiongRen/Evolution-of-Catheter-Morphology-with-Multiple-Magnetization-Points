"""Attribute stationarity failures to residual blocks."""

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
P2C10U_DIR = ROOT / "experiment_records_ik_optimize" / "20260605104500_p2c10u_physical_direction_slack"
for path in (ROOT, FK_OPT_DIR, P2C10T_DIR, P2C10U_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_observations import quaternion_so3_residual

import run_p2c10t_interface_slack_shooting as p2c10t
import run_p2c10u_physical_direction_slack as p2c10u

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


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _choose_cases() -> list[dict[str, Any]]:
    specs = []
    for source, base_dir, variants in [
        ("free", P2C10T_DIR, ["force", "torque", "wrench"]),
        ("physical_axis", P2C10U_DIR, ["force_axis", "torque_axis", "wrench_axis"]),
    ]:
        rows = _load_rows(base_dir / "branch_rows.csv")
        for case_name, alpha in [("n3_mixed_rand20", 0.25), ("n3_mixed_base_i4", 1.0)]:
            for variant in variants:
                candidates = [
                    row
                    for row in rows
                    if row["case"] == case_name and abs(float(row["alpha"]) - alpha) < 1e-12 and row["variant"] == variant
                ]
                if not candidates:
                    continue
                best = min(candidates, key=lambda row: float(row["stationarity_norm"]))
                specs.append(
                    {
                        "source": source,
                        "base_dir": base_dir,
                        "case": case_name,
                        "alpha": alpha,
                        "variant": variant,
                        "init_name": best["init_name"],
                        "previous_stationarity": float(best["stationarity_norm"]),
                        "previous_boundary": float(best["boundary_norm"]),
                        "previous_position_error": float(best["terminal_position_error"]),
                        "previous_orientation_error_deg": float(best["terminal_so3_error_deg"]),
                    }
                )
    return specs


def _build_case(case_name: str, alpha: float):
    return p2c10t._build_case(case_name, alpha)


def _forward(source: str, params: Any, y: jnp.ndarray, variant: str, steps_per_flex: int):
    if source == "free":
        return p2c10t.forward_interface_slack(params, y, variant=variant, steps_per_flex=steps_per_flex)
    if source == "physical_axis":
        return p2c10u.forward_physical_direction_slack(params, y, variant=variant, steps_per_flex=steps_per_flex)
    raise ValueError(f"Unknown source: {source}")


def _make_block_functions(
    *,
    source: str,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    variant: str,
    steps_per_flex: int,
    position_scale: float,
    so3_scale: float,
    boundary_scale: float,
    slack_scale: float,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def block_yI(y: jnp.ndarray, I: jnp.ndarray, block: str) -> jnp.ndarray:
        out = _forward(source, params_for_I(I), y, variant, steps_per_flex)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        if block == "position":
            return float(position_scale) * (out.tip_p - target_p_j)
        if block == "orientation":
            return float(so3_scale) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        if block == "boundary":
            return float(boundary_scale) * out.boundary
        if block == "slack":
            return float(slack_scale) * y[6:]
        raise ValueError(f"Unknown block: {block}")

    funcs = {}
    for block in ["position", "orientation", "boundary", "slack"]:
        def residual(y: jnp.ndarray, I: jnp.ndarray, *, _block: str = block) -> jnp.ndarray:
            return block_yI(y, I, _block)

        funcs[block] = {
            "residual": jax.jit(residual),
            "jacobian": jax.jit(jax.jacfwd(residual, argnums=0)),
        }

    def total_residual(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [funcs[block]["residual"](y, I).reshape(-1) for block in ["position", "orientation", "boundary", "slack"]],
            axis=0,
        )

    funcs["total"] = {
        "residual": jax.jit(total_residual),
        "jacobian": jax.jit(jax.jacfwd(total_residual, argnums=0)),
    }
    return funcs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-30 or nb <= 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _load_y(spec: dict[str, Any]) -> np.ndarray:
    fname = f"{spec['case']}_alpha_{spec['alpha']:.2f}_{spec['variant']}_branches.npz"
    z = np.load(Path(spec["base_dir"]) / fname)
    return np.asarray(z[f"{spec['init_name']}_y"], dtype=float)


def _analyze_spec(spec: dict[str, Any], failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    params, I0, target_p, target_q, _y_ref, _oracle = _build_case(spec["case"], spec["alpha"])
    y = _load_y(spec)
    funcs = _make_block_functions(
        source=spec["source"],
        params_base=params,
        target_p=target_p,
        target_q=target_q,
        variant=spec["variant"],
        steps_per_flex=4,
        position_scale=120.0,
        so3_scale=2.0,
        boundary_scale=0.5,
        slack_scale=0.05,
    )
    y_j = jnp.asarray(y, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64)
    _ = funcs["total"]["residual"](y_j, I_j).block_until_ready()
    block_rows: list[dict[str, Any]] = []
    gradients: dict[str, np.ndarray] = {}
    residual_norms: dict[str, float] = {}
    for block in ["position", "orientation", "boundary", "slack"]:
        r = np.asarray(jax.device_get(funcs[block]["residual"](y_j, I_j)), dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs[block]["jacobian"](y_j, I_j)), dtype=float)
        g = J.T @ r
        gradients[block] = g
        residual_norms[block] = float(np.linalg.norm(r))
        block_rows.append(
            {
                **{k: v for k, v in spec.items() if k != "base_dir"},
                "block": block,
                "residual_norm": float(np.linalg.norm(r)),
                "gradient_norm": float(np.linalg.norm(g)),
            }
        )
    total_r = np.asarray(jax.device_get(funcs["total"]["residual"](y_j, I_j)), dtype=float)
    total_J = np.asarray(jax.device_get(funcs["total"]["jacobian"](y_j, I_j)), dtype=float)
    total_g = total_J.T @ total_r
    svals = np.linalg.svd(total_J, compute_uv=False)
    total_norm = float(np.linalg.norm(total_g))
    for row in block_rows:
        block = row["block"]
        row["gradient_fraction"] = float(row["gradient_norm"] / max(total_norm, 1e-30))
        row["cosine_with_total"] = _cosine(gradients[block], total_g)
    pair_rows: list[dict[str, Any]] = []
    blocks = ["position", "orientation", "boundary", "slack"]
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            pair_rows.append(
                {
                    **{k: v for k, v in spec.items() if k != "base_dir"},
                    "block_a": a,
                    "block_b": b,
                    "gradient_cosine": _cosine(gradients[a], gradients[b]),
                    "gradient_norm_a": float(np.linalg.norm(gradients[a])),
                    "gradient_norm_b": float(np.linalg.norm(gradients[b])),
                }
            )
    dominant = max(block_rows, key=lambda row: row["gradient_norm"])
    antagonistic_pairs = [
        row
        for row in pair_rows
        if np.isfinite(row["gradient_cosine"])
        and row["gradient_cosine"] < -0.25
        and row["gradient_norm_a"] > 0.05 * max(dominant["gradient_norm"], 1e-30)
        and row["gradient_norm_b"] > 0.05 * max(dominant["gradient_norm"], 1e-30)
    ]
    summary = {
        **{k: v for k, v in spec.items() if k != "base_dir"},
        "state_dim": int(y.size),
        "total_residual_norm": float(np.linalg.norm(total_r)),
        "total_stationarity_norm": total_norm,
        "total_jacobian_rank": int(np.linalg.matrix_rank(total_J)),
        "total_jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "dominant_gradient_block": dominant["block"],
        "dominant_gradient_norm": dominant["gradient_norm"],
        "dominant_residual_block": max(residual_norms, key=residual_norms.get),
        "antagonistic_pair_count": len(antagonistic_pairs),
        "antagonistic_pairs": [f"{row['block_a']}-{row['block_b']}" for row in antagonistic_pairs],
    }
    if total_norm > 1e-5:
        _append_jsonl(
            failures,
            {
                "ts": time.time(),
                "failure": "stationarity_not_publishable",
                "evidence": {
                    "case": summary["case"],
                    "alpha": summary["alpha"],
                    "source": summary["source"],
                    "variant": summary["variant"],
                    "init_name": summary["init_name"],
                    "total_stationarity_norm": summary["total_stationarity_norm"],
                    "dominant_gradient_block": summary["dominant_gradient_block"],
                    "antagonistic_pairs": summary["antagonistic_pairs"],
                },
            },
        )
    return summary, block_rows, pair_rows


def _plot_block_gradients(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['source']}\n{r['case']}@{r['alpha']:.2f}\n{r['variant']}\n{r['block']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.45 * len(rows)), 8), sharex=True)
    axes[0].bar(x, [r["gradient_norm"] for r in rows])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("gradient norm")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [r["cosine_with_total"] for r in rows])
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel("cosine with total")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_gradient_attribution.png", dpi=180)
    plt.close(fig)


def _plot_pairwise_conflicts(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['source']}\n{r['case']}@{r['alpha']:.2f}\n{r['variant']}\n{r['block_a']}-{r['block_b']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(12, 0.36 * len(rows)), 5))
    ax.bar(x, [r["gradient_cosine"] for r in rows])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(-0.25, color="r", lw=0.8, ls="--")
    ax.set_ylabel("pairwise gradient cosine")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_gradient_conflicts.png", dpi=180)
    plt.close(fig)


def _plot_summary(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['source']}\n{r['case']}@{r['alpha']:.2f}\n{r['variant']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.7 * len(rows)), 9), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "total_stationarity_norm", "total stationarity"),
        (axes[1], "total_jacobian_condition", "Jacobian condition"),
        (axes[2], "antagonistic_pair_count", "conflict pairs"),
    ]:
        ax.bar(x, [r[key] for r in rows])
        if key != "antagonistic_pair_count":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stationarity_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for spec in _choose_cases():
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "analyze", **{k: v for k, v in spec.items() if k != "base_dir"}})
        summary, blocks, pairs = _analyze_spec(spec, failures)
        summaries.append(summary)
        block_rows.extend(blocks)
        pair_rows.extend(pairs)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "block_rows.csv", block_rows)
    _write_csv(OUT_DIR / "pairwise_gradient_rows.csv", pair_rows)
    _plot_summary(summaries)
    _plot_block_gradients(block_rows)
    _plot_pairwise_conflicts(pair_rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "summary_count": len(summaries),
        "not_publishable_count": sum(row["total_stationarity_norm"] > 1e-5 for row in summaries),
        "dominant_block_counts": {
            block: sum(row["dominant_gradient_block"] == block for row in summaries)
            for block in ["position", "orientation", "boundary", "slack"]
        },
        "antagonistic_case_count": sum(row["antagonistic_pair_count"] > 0 for row in summaries),
        "case_summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable({k: payload[k] for k in ["summary_count", "not_publishable_count", "dominant_block_counts", "antagonistic_case_count"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
