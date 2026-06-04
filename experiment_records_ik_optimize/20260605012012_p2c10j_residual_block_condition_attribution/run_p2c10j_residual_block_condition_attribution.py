"""Attribute residual-block stationarity and conditioning for P2c10i cases."""

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
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
P2C9F_DIR = ROOT / "experiment_records_ik_optimize" / "20260604141002_p2c9f_exact_stationarity_ift_probe"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, P2C9F_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from inverse_kinematics.ik_terminal_shooting_jax import forward_segment_wrench

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8
import run_p2c9f_exact_stationarity_ift_probe as p2c9f

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
BLOCKS = ("terminal_position", "terminal_orientation", "terminal_boundary", "state_regularization", "interface_continuity")


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


def _make_block_functions(
    *,
    params_base: Any,
    target_p: np.ndarray,
    target_q: np.ndarray,
    y_ref: np.ndarray,
    reg_mu: float,
    continuity_mu: float,
    steps_per_flex: int,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))
    cont_scale = jnp.sqrt(jnp.asarray(continuity_mu, dtype=jnp.float64))

    def params_for_I(I: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def terminal_position(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        return float(p2c8.WEIGHTS.tip_weight) * (out.tip_p - target_p_j)

    def terminal_orientation(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        return float(p2c8.WEIGHTS.so3_weight) * p2c7._so3_vec(out.tip_q, target_q_j)

    def terminal_boundary(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        out = forward_segment_wrench(params_for_I(I), y, steps_per_flex=steps_per_flex)
        return float(p2c8.WEIGHTS.boundary_weight) * out.boundary

    def state_regularization(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        del I
        return reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)

    def interface_continuity(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return cont_scale * p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=steps_per_flex)

    block_funs = {
        "terminal_position": terminal_position,
        "terminal_orientation": terminal_orientation,
        "terminal_boundary": terminal_boundary,
        "state_regularization": state_regularization,
        "interface_continuity": interface_continuity,
    }
    return {
        "block_jit": {name: jax.jit(fun) for name, fun in block_funs.items()},
        "block_jac_jit": {name: jax.jit(jax.jacfwd(fun, argnums=0)) for name, fun in block_funs.items()},
    }


def _build_case_bundle(case: Any, continuity_mu: float, reg_mu: float, max_iter: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I0 = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_oracle = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    residual_jit, task_jit, output_jit, continuity_jit, Jy_jit, _JI_jit, _Ty_jit, _TI_jit = p2c8._make_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    grad_jit, H_jit, _GI_jit, _Ty_exact_jit, _TI_exact_jit = p2c9f._make_exact_functions(
        residual_fun=residual_jit,
        task_fun=task_jit,
    )
    block_bundle = _make_block_functions(
        params_base=params_base,
        target_p=target_p,
        target_q=target_q,
        y_ref=y_oracle,
        reg_mu=float(reg_mu),
        continuity_mu=float(continuity_mu),
        steps_per_flex=4,
    )
    _ = residual_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    _ = H_jit(jnp.asarray(y_oracle), jnp.asarray(I0)).block_until_ready()
    base = p2c7._solve_branch(
        y0=y_oracle,
        I=I0,
        residual_jit=residual_jit,
        output_jit=output_jit,
        Jy_jit=Jy_jit,
        max_iter=max_iter,
    )
    return {
        "engine": engine,
        "oracle": oracle,
        "params_base": params_base,
        "I0": I0,
        "target_p": np.asarray(target_p, dtype=float),
        "target_q": np.asarray(target_q, dtype=float),
        "y_ref": y_oracle,
        "base": base,
        "residual_jit": residual_jit,
        "output_jit": output_jit,
        "continuity_jit": continuity_jit,
        "Jy_jit": Jy_jit,
        "grad_jit": grad_jit,
        "H_jit": H_jit,
        **block_bundle,
    }


def _block_rows(case_name: str, alpha: float, bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_j = jnp.asarray(bundle["base"].y, dtype=jnp.float64)
    I_j = jnp.asarray(bundle["I0"], dtype=jnp.float64)
    full_r = np.asarray(jax.device_get(bundle["residual_jit"](y_j, I_j)), dtype=float)
    full_J = np.asarray(jax.device_get(bundle["Jy_jit"](y_j, I_j)), dtype=float)
    full_grad = full_J.T @ full_r
    full_H_gn = full_J.T @ full_J
    full_svals = np.linalg.svd(full_J, compute_uv=False)
    _u, _s, vh = np.linalg.svd(full_H_gn)
    weak_dir = vh[-1].reshape(-1)
    weak_dir = weak_dir / max(float(np.linalg.norm(weak_dir)), 1.0e-12)
    exact_H = np.asarray(jax.device_get(bundle["H_jit"](y_j, I_j)), dtype=float)
    exact_svals = np.linalg.svd(exact_H, compute_uv=False)
    exact_grad = np.asarray(jax.device_get(bundle["grad_jit"](y_j, I_j)), dtype=float)
    full_summary = {
        "case": case_name,
        "alpha": alpha,
        "base_residual_norm": bundle["base"].residual_norm,
        "base_stationarity_norm": bundle["base"].stationarity_norm,
        "exact_stationarity_norm": float(np.linalg.norm(exact_grad)),
        "base_boundary_norm": bundle["base"].boundary_norm,
        "gn_condition": float(full_svals[0] / max(float(full_svals[-1]), np.finfo(float).tiny)) if full_svals.size else float("inf"),
        "gn_min_singular": float(full_svals[-1]) if full_svals.size else float("nan"),
        "gn_max_singular": float(full_svals[0]) if full_svals.size else float("nan"),
        "exact_hessian_condition": float(exact_svals[0] / max(float(exact_svals[-1]), np.finfo(float).tiny)) if exact_svals.size else float("inf"),
        "exact_hessian_min_singular": float(exact_svals[-1]) if exact_svals.size else float("nan"),
        "exact_hessian_max_singular": float(exact_svals[0]) if exact_svals.size else float("nan"),
    }
    rows = []
    weak_total = float(weak_dir @ full_H_gn @ weak_dir)
    for block_name in BLOCKS:
        r_i = np.asarray(jax.device_get(bundle["block_jit"][block_name](y_j, I_j)), dtype=float)
        J_i = np.asarray(jax.device_get(bundle["block_jac_jit"][block_name](y_j, I_j)), dtype=float)
        grad_i = J_i.T @ r_i
        H_i = J_i.T @ J_i
        svals_i = np.linalg.svd(J_i, compute_uv=False)
        weak_curvature = float(weak_dir @ H_i @ weak_dir)
        rows.append(
            {
                "case": case_name,
                "alpha": alpha,
                "block": block_name,
                "residual_norm": float(np.linalg.norm(r_i)),
                "grad_contribution_norm": float(np.linalg.norm(grad_i)),
                "grad_fraction_of_total": float(np.linalg.norm(grad_i) / max(float(np.linalg.norm(full_grad)), 1.0e-12)),
                "grad_cosine_with_total": float(np.dot(grad_i, full_grad) / max(float(np.linalg.norm(grad_i) * np.linalg.norm(full_grad)), 1.0e-30)),
                "jac_spectral_norm": float(svals_i[0]) if svals_i.size else 0.0,
                "jac_min_singular": float(svals_i[-1]) if svals_i.size else 0.0,
                "jac_rank": int(np.linalg.matrix_rank(J_i)),
                "weak_direction_curvature": weak_curvature,
                "weak_direction_curvature_fraction": weak_curvature / max(weak_total, 1.0e-30),
                "block_length": int(r_i.size),
            }
        )
    return rows, full_summary


def _physical_rows(case_name: str, alpha: float, bundle: dict[str, Any]) -> dict[str, Any]:
    y_j = jnp.asarray(bundle["base"].y, dtype=jnp.float64)
    I_j = jnp.asarray(bundle["I0"], dtype=jnp.float64)
    out = bundle["base"].out
    boundary = np.asarray(out["boundary"], dtype=float).reshape(6)
    continuity = np.asarray(jax.device_get(bundle["continuity_jit"](y_j, I_j)), dtype=float).reshape(-1)
    interfaces = continuity.reshape((-1, 6)) if continuity.size else np.zeros((0, 6))
    row: dict[str, Any] = {
        "case": case_name,
        "alpha": alpha,
        "terminal_boundary_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_boundary_moment_norm": float(np.linalg.norm(boundary[3:])),
        "interface_count": int(interfaces.shape[0]),
        "max_interface_force_jump_norm": float(max([np.linalg.norm(item[:3]) for item in interfaces], default=0.0)),
        "max_interface_moment_jump_norm": float(max([np.linalg.norm(item[3:]) for item in interfaces], default=0.0)),
        "shooting_state_norm": float(np.linalg.norm(bundle["base"].y)),
        "currents": bundle["I0"].tolist(),
        "tip_p": np.asarray(out["tip_p"], dtype=float).reshape(3).tolist(),
        "target_p": bundle["target_p"].tolist(),
    }
    for i, item in enumerate(interfaces):
        row[f"interface_{i}_force_jump_norm"] = float(np.linalg.norm(item[:3]))
        row[f"interface_{i}_moment_jump_norm"] = float(np.linalg.norm(item[3:]))
    return row


def _plot_block_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cases = sorted({row["case"] for row in rows})
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, max(4, 3.8 * len(cases))), squeeze=False)
    for i, case_name in enumerate(cases):
        sub = [row for row in rows if row["case"] == case_name]
        blocks = [row["block"] for row in sub]
        x = np.arange(len(blocks))
        axes[i, 0].bar(x, [row["grad_contribution_norm"] for row in sub])
        axes[i, 0].set_yscale("log")
        axes[i, 0].set_title(f"{case_name}: stationarity contribution")
        axes[i, 1].bar(x, [row["weak_direction_curvature_fraction"] for row in sub])
        axes[i, 1].set_title(f"{case_name}: weak-direction curvature share")
        for ax in axes[i]:
            ax.set_xticks(x)
            ax.set_xticklabels(blocks, rotation=35, ha="right", fontsize=8)
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "block_attribution.png", dpi=180)
    plt.close(fig)


def _plot_condition_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [row["case"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, [row["gn_condition"] for row in rows], width=0.36, label="residual Jacobian")
    ax.bar(x + 0.18, [row["exact_hessian_condition"] for row in rows], width=0.36, label="exact Hessian")
    ax.axhline(1.0e10, color="r", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("condition number")
    ax.set_title("local conditioning summary")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "condition_summary.png", dpi=180)
    plt.close(fig)


def _plot_time_cost(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    labels = [row["case"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, [row.get("base_steady_s", 0.0) for row in rows], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("steady solve time (s)")
    ax.set_title("baseline solve time")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_cost.png", dpi=180)
    plt.close(fig)


def _dominant_block(rows: list[dict[str, Any]], case_name: str, field: str) -> str:
    sub = [row for row in rows if row["case"] == case_name]
    if not sub:
        return ""
    return max(sub, key=lambda row: row[field])["block"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="n2_z_pair_rand12:0.25,n3_mixed_rand20:0.25")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=32)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t_all = time.perf_counter()

    block_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []

    specs: list[tuple[str, float]] = []
    for raw in args.candidates.split(","):
        if not raw.strip():
            continue
        name, alpha_text = raw.split(":", 1)
        specs.append((name.strip(), float(alpha_text)))

    for case_name, alpha in specs:
        target_case = _case_by_name(case_name)
        scaled = _scaled_case(target_case, alpha)
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "build", "case": scaled.name})
        bundle = _build_case_bundle(scaled, args.continuity_mu, args.reg_mu, args.max_iter)
        rows_i, case_summary = _block_rows(case_name, alpha, bundle)
        block_rows.extend(rows_i)
        physical_row = _physical_rows(case_name, alpha, bundle)
        physical_rows.append(physical_row)
        dominant_grad = _dominant_block(rows_i, case_name, "grad_contribution_norm")
        weakest_support = _dominant_block(rows_i, case_name, "weak_direction_curvature_fraction")
        failure_class = (
            "一阶平稳性由残差块主导"
            if case_summary["exact_stationarity_norm"] > 1.0e-3
            else "局部条件数过高"
            if case_summary["exact_hessian_condition"] > 1.0e10
            else "未触发主要失败门禁"
        )
        case_row = {
            **case_summary,
            "base_steady_s": bundle["base"].steady_s,
            "active_magnet_count": _active_magnet_count(target_case),
            "dominant_stationarity_block": dominant_grad,
            "dominant_weak_curvature_block": weakest_support,
            "failure_class": failure_class,
            "terminal_boundary_force_norm": physical_row["terminal_boundary_force_norm"],
            "terminal_boundary_moment_norm": physical_row["terminal_boundary_moment_norm"],
            "max_interface_force_jump_norm": physical_row["max_interface_force_jump_norm"],
            "max_interface_moment_jump_norm": physical_row["max_interface_moment_jump_norm"],
        }
        case_rows.append(case_row)
        if failure_class != "未触发主要失败门禁":
            _append_jsonl(failures, {"ts": time.time(), "case": scaled.name, "failure_class": failure_class, "evidence": case_row})

    _plot_block_rows(block_rows)
    _plot_condition_summary(case_rows)
    _plot_time_cost(case_rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
            "continuity_mu": args.continuity_mu,
            "reg_mu": args.reg_mu,
            "max_iter": args.max_iter,
        },
        "classification": "失败归因审计",
        "case_rows": case_rows,
        "block_rows": block_rows,
        "physical_rows": physical_rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "case_summary.csv", case_rows)
    _write_csv(OUT_DIR / "block_attribution.csv", block_rows)
    _write_csv(OUT_DIR / "physical_interface_summary.csv", physical_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
