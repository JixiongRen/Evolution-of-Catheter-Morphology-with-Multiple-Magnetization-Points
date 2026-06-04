"""Centered magnetic-wrench prototype for P2c sensitivity failures."""

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
P2C10C_DIR = ROOT / "experiment_records_ik_optimize" / "20260604222000_p2c10c_rigid_load_coupling_counterfactual"
P2C2_DIR = ROOT / "experiment_records_ik_optimize" / "20260602103750_p2c2_terminal_pose_constraint"
P2C4_DIR = ROOT / "experiment_records_ik_optimize" / "20260603095445_p2c4_clean_terminal_multi_case"
P2C7_DIR = ROOT / "experiment_records_ik_optimize" / "20260604110544_p2c7_shooting_sensitivity_audit"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, P2C10C_DIR, P2C2_DIR, P2C4_DIR, P2C7_DIR, P2C8_DIR, FK_OPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c7_sensitivity_audit as p2c7
import run_p2c8_continuity_branch_sensitivity as p2c8
import run_p2c10c_rigid_load_coupling_counterfactual as p2c10c

from basics_nondim import quat_normalize, quat_to_rotmat
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from forward_kinematics_optimized.segments_nondim import rigid_state_along_dim
from inverse_kinematics.ik_observations import quaternion_so3_residual
from inverse_kinematics.ik_terminal_shooting_jax import CleanShootingOutput, initial_state, normalize_state

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


def _interpolated_case(base: Any, target: Any, alpha: float) -> Any:
    currents = (1.0 - alpha) * np.asarray(base.currents, dtype=float) + alpha * np.asarray(target.currents, dtype=float)
    return p2c4.CaseSpec(
        name=f"alpha_{alpha:.3f}",
        m_body_list=base.m_body_list,
        currents=tuple(float(x) for x in currents),
    )


def _centered_external_wrench(params: Any, x_prox: jnp.ndarray, i: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    rigid = params.rigid[i]
    pp = x_prox[0:3]
    Qp = quat_normalize(x_prox[3:7])
    R = quat_to_rotmat(Qp)
    center_offset = 0.5 * float(rigid.length) * (R @ rigid.v_star)
    p_center = pp + center_offset

    f_world = jnp.zeros((3,), dtype=jnp.float64)
    tau_world = jnp.zeros((3,), dtype=jnp.float64)

    gravity = params.gravity_rigid_list[i]
    if gravity is not None:
        f_g = gravity.force_world()
        tau_g_world = R @ gravity.torque_body(Qp)
        f_world = f_world + f_g
        tau_world = tau_world + tau_g_world

    magnet_params = params.magnet_params_list[i]
    if (params.magnetic_model is not None) and (magnet_params is not None) and (params.coil_currents is not None):
        f_m, tau_m_center = params.magnetic_model.compute_wrench_world(
            p_world=p_center,
            R_world_from_body=R,
            magnet_params=magnet_params,
            coil_currents=params.coil_currents,
        )
        f_world = f_world + f_m
        tau_world = tau_world + tau_m_center + jnp.cross(center_offset, f_m)

    f_world = f_world + jnp.asarray(params.f_ext_list[i], dtype=jnp.float64).reshape(3)
    tau_world = tau_world + jnp.asarray(params.tau_ext_list[i], dtype=jnp.float64).reshape(3)
    return f_world, tau_world


def _advance_rigid_centered(params: Any, x_prox: jnp.ndarray, i: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rigid = params.rigid[i]
    f_total, tau_total = _centered_external_wrench(params, x_prox, i)
    x_mid = rigid_state_along_dim(
        x_prox_dim=x_prox,
        sigma=0.5 * float(rigid.length),
        rigid=rigid,
        f_ext_total_dim=f_total,
        tau_ext_total_dim=tau_total,
    )
    x_dist = rigid_state_along_dim(
        x_prox_dim=x_prox,
        sigma=float(rigid.length),
        rigid=rigid,
        f_ext_total_dim=f_total,
        tau_ext_total_dim=tau_total,
    )
    return normalize_state(x_dist), x_mid[0:3], quat_normalize(x_mid[3:7])


def _forward_centered(params: Any, y: jnp.ndarray, *, steps_per_flex: int) -> CleanShootingOutput:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    centerline_parts = [x[0:3].reshape(1, 3)]
    rigid_mid_p = []
    rigid_mid_q = []
    rigid_dist_p = []
    rigid_dist_q = []
    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = normalize_state(x)
        x, flex_points = p2c10c._integrate_flex(params, x, i, steps_per_flex)
        centerline_parts.append(flex_points)
        x, mid_p, mid_q = _advance_rigid_centered(params, x, i)
        rigid_mid_p.append(mid_p)
        rigid_mid_q.append(mid_q)
        rigid_dist_p.append(x[0:3])
        rigid_dist_q.append(x[3:7])
        centerline_parts.append(x[0:3].reshape(1, 3))

    boundary = jnp.concatenate(
        [
            x[7:10] / jnp.maximum(float(params.scales.F_ref), 1.0e-12),
            x[10:13] / jnp.maximum(float(params.scales.M_ref), 1.0e-12),
        ],
        axis=0,
    )
    return CleanShootingOutput(
        centerline=jnp.concatenate(centerline_parts, axis=0),
        rigid_mid_p=jnp.stack(rigid_mid_p, axis=0),
        rigid_mid_q=jnp.stack(rigid_mid_q, axis=0),
        rigid_dist_p=jnp.stack(rigid_dist_p, axis=0),
        rigid_dist_q=jnp.stack(rigid_dist_q, axis=0),
        tip_p=x[0:3],
        tip_q=quat_normalize(x[3:7]),
        boundary=boundary,
    )


def _interface_residual_centered(params: Any, y: jnp.ndarray, *, steps_per_flex: int) -> jnp.ndarray:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    residuals = []
    f_ref = jnp.maximum(float(params.scales.F_ref), 1.0e-12)
    m_ref = jnp.maximum(float(params.scales.M_ref), 1.0e-12)
    for i in range(len(params.flex)):
        x = x.at[7:10].set(blocks[i, 0:3] * float(params.scales.F_ref))
        x = x.at[10:13].set(blocks[i, 3:6] * float(params.scales.M_ref))
        x = normalize_state(x)
        x, _flex_points = p2c10c._integrate_flex(params, x, i, steps_per_flex)
        x, _mid_p, _mid_q = _advance_rigid_centered(params, x, i)
        if i + 1 < len(params.flex):
            dist_block = jnp.concatenate([x[7:10] / f_ref, x[10:13] / m_ref], axis=0)
            residuals.append(blocks[i + 1] - dist_block)
    if not residuals:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(residuals, axis=0)


def _build_bundle(case: Any, *, reg_mu: float, continuity_mu: float, steps_per_flex: int) -> dict[str, Any]:
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    I = np.asarray(case.currents, dtype=float)
    oracle = p2c2.p2c1.p2c._solve_oracle(engine, I, z0=None)
    target_p, target_q = p2c2._oracle_terminal_pose(oracle, engine)
    params_base = oracle["params"]
    y_ref = p2c2.p2c1._y_guess_from_oracle(oracle["z"], params_base, "segment_wrench")
    weights = p2c8.WEIGHTS
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    target_q_j = jnp.asarray(target_q, dtype=jnp.float64).reshape(4)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    reg_scale = jnp.sqrt(jnp.asarray(reg_mu, dtype=jnp.float64))
    cont_scale = jnp.sqrt(jnp.asarray(continuity_mu, dtype=jnp.float64))

    def params_for_I(I_j: jnp.ndarray):
        return with_coil_currents(params_base, jnp.asarray(I_j, dtype=jnp.float64).reshape(8))

    def residual_yI(y: jnp.ndarray, I_j: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I_j)
        out = _forward_centered(params, y, steps_per_flex=steps_per_flex)
        r_p = float(weights.tip_weight) * (out.tip_p - target_p_j)
        r_q = float(weights.so3_weight) * quaternion_so3_residual(out.tip_q, target_q_j, sigma_angle=1.0)
        r_b = float(weights.boundary_weight) * out.boundary
        reg = reg_scale * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        continuity = cont_scale * _interface_residual_centered(params, y, steps_per_flex=steps_per_flex)
        return jnp.concatenate([r_p.reshape(-1), r_q.reshape(-1), r_b.reshape(-1), reg.reshape(-1), continuity.reshape(-1)], axis=0)

    def output_yI(y: jnp.ndarray, I_j: jnp.ndarray) -> CleanShootingOutput:
        return _forward_centered(params_for_I(I_j), y, steps_per_flex=steps_per_flex)

    residual_jit = jax.jit(residual_yI)
    output_jit = jax.jit(output_yI)
    Jy_jit = jax.jit(jax.jacfwd(residual_yI, argnums=0))
    _ = residual_jit(jnp.asarray(y_ref), jnp.asarray(I)).block_until_ready()
    return {"I": I, "y_ref": y_ref, "residual_jit": residual_jit, "output_jit": output_jit, "Jy_jit": Jy_jit}


def _solve(bundle: dict[str, Any], y0: np.ndarray, max_iter: int) -> Any:
    return p2c7._solve_branch(
        y0=y0,
        I=bundle["I"],
        residual_jit=bundle["residual_jit"],
        output_jit=bundle["output_jit"],
        Jy_jit=bundle["Jy_jit"],
        max_iter=max_iter,
    )


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    alphas = [row["alpha"] for row in rows]
    ax.plot(alphas, [row["residual_norm"] for row in rows], marker="o", label="total residual")
    ax.plot(alphas, [row["boundary_residual_norm"] for row in rows], marker="o", label="terminal boundary residual")
    ax.plot(alphas, [row["stationarity_norm"] for row in rows], marker="o", label="first-order optimality residual")
    ax.axhline(1.0e-3, color="#c5221f", linestyle="--", label="optimality gate")
    ax.set_yscale("log")
    ax.set_xlabel("continuation alpha")
    ax.set_ylabel("metric value")
    ax.set_title("Centered magnetic-wrench prototype")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "centered_wrench_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_case", default="n3_mixed_base_i4")
    parser.add_argument("--target_case", default="n3_mixed_rand20")
    parser.add_argument("--alphas", default="0.20,0.25")
    parser.add_argument("--continuity_mu", type=float, default=1.0)
    parser.add_argument("--reg_mu", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=24)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")

    base = _case_by_name(args.base_case)
    target = _case_by_name(args.target_case)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    for alpha in alphas:
        _append_jsonl(heartbeat, {"ts": time.time(), "phase": "solve", "alpha": alpha})
        case = _interpolated_case(base, target, alpha)
        try:
            bundle = _build_bundle(case, reg_mu=args.reg_mu, continuity_mu=args.continuity_mu, steps_per_flex=args.steps_per_flex)
            solve = _solve(bundle, bundle["y_ref"], args.max_iter)
            row = {
                "alpha": alpha,
                "success_flag": bool(solve.success),
                "residual_norm": solve.residual_norm,
                "stationarity_norm": solve.stationarity_norm,
                "boundary_residual_norm": solve.boundary_norm,
                "steady_s": solve.steady_s,
                "linearizable": bool(solve.residual_norm <= 1.0e-2 and solve.boundary_norm <= 2.0e-1 and solve.stationarity_norm <= 1.0e-3),
            }
        except Exception as exc:
            row = {
                "alpha": alpha,
                "success_flag": False,
                "residual_norm": float("inf"),
                "stationarity_norm": float("inf"),
                "boundary_residual_norm": float("inf"),
                "steady_s": 0.0,
                "linearizable": False,
                "error": repr(exc),
            }
            _append_jsonl(failures, {"ts": time.time(), "failure_class": "中心磁载原型求解异常", "evidence": row})
        rows.append(row)
        if not row["linearizable"]:
            _append_jsonl(failures, {"ts": time.time(), "failure_class": "中心磁载原型未通过可线性化门禁", "evidence": row})

    _plot(rows)
    summary = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "total_wall_s": float(time.perf_counter() - t_all),
        },
        "classification": "centered-magnetic-wrench-prototype",
        "interpretation_scope": "The magnetic wrench is evaluated at the rigid magnet center and shifted to the proximal point by the force moment arm.",
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "sensitivity_audit.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
