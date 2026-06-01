"""Run non-production physical shooting diagnostics.

This experiment-local script implements a low-dimensional shooting IVP with
proximal internal force and moment as the unknown state. It directly uses the
same elastic, gravity, magnetic, base-pose, and rigid-segment callbacks as the
optimized FK oracle, but it is not connected to production IK.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_artifacts import extract_centerline_dim
from inverse_kinematics.ik_observations import axis_from_quat

from forward_kinematics_optimized.nondim import x_bar_to_dim
from forward_kinematics_optimized.utils_nondim import unpack_z_bar_jax
from segments_nondim import cosserat_rhs_dim, rigid_state_along_dim
from external_wrench_nondim import compute_external_wrench_total_rigid
from basics_nondim import quat_normalize

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


@dataclass
class ShootingResult:
    """Physical shooting solve output."""

    y: np.ndarray
    xi: np.ndarray
    centerline: np.ndarray
    rigid_mid_p: np.ndarray
    rigid_mid_q: np.ndarray
    tip_p: np.ndarray
    tip_q: np.ndarray
    boundary_residual: np.ndarray
    success: bool
    cost: float
    nfev: int
    wall_s: float
    residual_history: list[float]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


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


def _parse_m_list(text: str) -> tuple[int, int, int]:
    vals = [int(float(x.strip())) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--M_list must contain three comma-separated integers")
    return vals[0], vals[1], vals[2]


def _build_engine(M_list: tuple[int, int, int], *, max_iter: int, tol: float) -> ForwardKinematicsEngine:
    calib = ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=M_list,
        L1_min=1e-6,
        L_protrude_max=0.11,
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=str(calib),
        m_body_list=[(0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0)],
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
        scale_include_external=True,
    )


def _directions(active_indices: list[int]) -> list[tuple[str, np.ndarray]]:
    directions = [
        ("dA", np.asarray([0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.1, 0.05], dtype=float)),
        ("dB", np.asarray([-0.1, 0.25, -0.2, 0.15, 0.1, -0.05, 0.3, -0.2], dtype=float)),
    ]
    mask = np.zeros((8,), dtype=float)
    mask[np.asarray(active_indices, dtype=int)] = 1.0
    out = []
    for name, vec in directions:
        vec = vec * mask
        out.append((name, vec / max(float(np.linalg.norm(vec)), 1e-12)))
    return out


def _state_np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float).reshape(-1)


def _normalize_state(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(13).copy()
    q = x[3:7]
    x[3:7] = q / max(float(np.linalg.norm(q)), 1e-12)
    return x


def _rk4_step(rhs, x: np.ndarray, sigma: float, h: float) -> np.ndarray:
    k1 = rhs(x, sigma)
    k2 = rhs(_normalize_state(x + 0.5 * h * k1), sigma + 0.5 * h)
    k3 = rhs(_normalize_state(x + 0.5 * h * k2), sigma + 0.5 * h)
    k4 = rhs(_normalize_state(x + h * k3), sigma + h)
    return _normalize_state(x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _arc_resample(centerline: np.ndarray, count: int) -> np.ndarray:
    centerline = np.asarray(centerline, dtype=float)
    if len(centerline) == count:
        return centerline
    if len(centerline) < 2:
        return np.repeat(centerline, count, axis=0)
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(float(arc[-1]), 1e-12)
    target = np.linspace(0.0, total, int(count))
    return np.stack([np.interp(target, arc, centerline[:, i]) for i in range(3)], axis=1)


def _axis_from_q(q: np.ndarray) -> np.ndarray:
    axis = np.asarray(jax.device_get(axis_from_quat(jnp.asarray(q, dtype=jnp.float64))), dtype=float).reshape(3)
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _initial_state_from_y(y: np.ndarray, params: Any) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(6)
    scales = params.scales
    x = np.zeros((13,), dtype=float)
    x[0:3] = np.asarray(jax.device_get(params.p0_bar), dtype=float).reshape(3) * float(scales.L_ref)
    x[3:7] = _state_np(params.Q0)
    x[7:10] = y[0:3] * float(scales.F_ref)
    x[10:13] = y[3:6] * float(scales.M_ref)
    return _normalize_state(x)


def _proximal_y_from_oracle(z_bar: Any, params: Any) -> np.ndarray:
    """Extract an informed proximal wrench guess from the FK oracle state."""
    x_nodes_list, _k_list, _x_rigid_list = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    x0_dim = x_bar_to_dim(x_nodes_list[0][0], params.scales)
    f0 = np.asarray(jax.device_get(x0_dim[7:10]), dtype=float)
    tau0 = np.asarray(jax.device_get(x0_dim[10:13]), dtype=float)
    return np.concatenate(
        [
            f0 / max(float(params.scales.F_ref), 1e-12),
            tau0 / max(float(params.scales.M_ref), 1e-12),
        ]
    )


def _physical_forward(params: Any, y: np.ndarray, *, steps_per_flex: int = 8) -> dict[str, Any]:
    x = _initial_state_from_y(y, params)
    centerline = [x[0:3].copy()]
    rigid_mid_p = []
    rigid_mid_q = []
    N = len(params.flex)
    for i in range(N):
        flex = params.flex[i]
        if i == 0:
            L = float(np.asarray(jax.device_get(params.L1_dim)).reshape(()))
        else:
            L = float(flex.length)
        n_steps = max(int(steps_per_flex), int(params.M_list[i]) * int(steps_per_flex))
        h = L / max(n_steps, 1)
        f_line = jnp.asarray(params.flex_f_line_world_list[i], dtype=jnp.float64)
        tau_line = jnp.asarray(params.flex_tau_line_world_list[i], dtype=jnp.float64)

        def rhs(x_np: np.ndarray, sigma: float) -> np.ndarray:
            dx = cosserat_rhs_dim(
                jnp.asarray(x_np, dtype=jnp.float64),
                float(sigma),
                Kse_inv=flex.Kse_inv,
                Kbt_inv=flex.Kbt_inv,
                v_star=flex.v_star,
                u_star=flex.u_star,
                fext_density=lambda _x, _s, v=f_line: v,
                tauext_density=lambda _x, _s, v=tau_line: v,
            )
            return _state_np(dx)

        sigma = 0.0
        for _ in range(n_steps):
            x = _rk4_step(rhs, x, sigma, h)
            sigma += h
            centerline.append(x[0:3].copy())

        x_prox = x.copy()
        rigid = params.rigid[i]
        f_total, tau_total = compute_external_wrench_total_rigid(
            x_proximal=jnp.asarray(x_prox, dtype=jnp.float64),
            rigid_length=float(rigid.length),
            gravity=params.gravity_rigid_list[i],
            magnetic_model=params.magnetic_model,
            magnet_params=params.magnet_params_list[i],
            coil_currents=params.coil_currents,
        )
        f_total = f_total + params.f_ext_list[i]
        tau_total = tau_total + params.tau_ext_list[i]
        x_mid = rigid_state_along_dim(
            x_prox_dim=jnp.asarray(x_prox, dtype=jnp.float64),
            sigma=0.5 * float(rigid.length),
            rigid=rigid,
            f_ext_total_dim=f_total,
            tau_ext_total_dim=tau_total,
        )
        x_dist = rigid_state_along_dim(
            x_prox_dim=jnp.asarray(x_prox, dtype=jnp.float64),
            sigma=float(rigid.length),
            rigid=rigid,
            f_ext_total_dim=f_total,
            tau_ext_total_dim=tau_total,
        )
        x_mid_np = _normalize_state(_state_np(x_mid))
        x = _normalize_state(_state_np(x_dist))
        rigid_mid_p.append(x_mid_np[0:3].copy())
        rigid_mid_q.append(x_mid_np[3:7].copy())
        centerline.append(x[0:3].copy())

    boundary = np.concatenate(
        [
            x[7:10] / max(float(params.scales.F_ref), 1e-12),
            x[10:13] / max(float(params.scales.M_ref), 1e-12),
        ]
    )
    return {
        "centerline": np.asarray(centerline, dtype=float),
        "rigid_mid_p": np.asarray(rigid_mid_p, dtype=float),
        "rigid_mid_q": np.asarray(rigid_mid_q, dtype=float),
        "tip_p": x[0:3].copy(),
        "tip_q": x[3:7].copy(),
        "boundary_residual": boundary,
        "xi": np.concatenate([x[7:10], x[10:13]]),
    }


def _solve_shooting(params: Any, *, y0: np.ndarray | None, max_nfev: int, steps_per_flex: int) -> ShootingResult:
    history: list[float] = []
    if y0 is None:
        y0 = np.zeros((6,), dtype=float)

    def residual(y: np.ndarray) -> np.ndarray:
        out = _physical_forward(params, y, steps_per_flex=steps_per_flex)
        r = np.asarray(out["boundary_residual"], dtype=float)
        history.append(float(np.linalg.norm(r)))
        return r

    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        np.asarray(y0, dtype=float).reshape(6),
        method="trf",
        x_scale=np.ones((6,), dtype=float),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=int(max_nfev),
    )
    wall_s = time.perf_counter() - t0
    out = _physical_forward(params, opt.x, steps_per_flex=steps_per_flex)
    xi = np.concatenate(
        [
            opt.x[0:3] * float(params.scales.F_ref),
            opt.x[3:6] * float(params.scales.M_ref),
        ]
    )
    return ShootingResult(
        y=np.asarray(opt.x, dtype=float),
        xi=xi,
        centerline=np.asarray(out["centerline"], dtype=float),
        rigid_mid_p=np.asarray(out["rigid_mid_p"], dtype=float),
        rigid_mid_q=np.asarray(out["rigid_mid_q"], dtype=float),
        tip_p=np.asarray(out["tip_p"], dtype=float),
        tip_q=np.asarray(out["tip_q"], dtype=float),
        boundary_residual=np.asarray(out["boundary_residual"], dtype=float),
        success=bool(opt.success),
        cost=float(opt.cost),
        nfev=int(opt.nfev),
        wall_s=float(wall_s),
        residual_history=history,
    )


def _evaluate_shooting_fixed_y(params: Any, y: np.ndarray, *, steps_per_flex: int) -> ShootingResult:
    """Evaluate the physical IVP without re-solving the shooting residual."""
    out = _physical_forward(params, y, steps_per_flex=steps_per_flex)
    xi = np.concatenate(
        [
            np.asarray(y, dtype=float)[0:3] * float(params.scales.F_ref),
            np.asarray(y, dtype=float)[3:6] * float(params.scales.M_ref),
        ]
    )
    return ShootingResult(
        y=np.asarray(y, dtype=float).reshape(6),
        xi=xi,
        centerline=np.asarray(out["centerline"], dtype=float),
        rigid_mid_p=np.asarray(out["rigid_mid_p"], dtype=float),
        rigid_mid_q=np.asarray(out["rigid_mid_q"], dtype=float),
        tip_p=np.asarray(out["tip_p"], dtype=float),
        tip_q=np.asarray(out["tip_q"], dtype=float),
        boundary_residual=np.asarray(out["boundary_residual"], dtype=float),
        success=True,
        cost=0.5 * float(np.dot(out["boundary_residual"], out["boundary_residual"])),
        nfev=1,
        wall_s=0.0,
        residual_history=[float(np.linalg.norm(out["boundary_residual"]))],
    )


def _solve_oracle(engine: ForwardKinematicsEngine, currents: np.ndarray, *, z0: Any | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=jnp.asarray(currents, dtype=jnp.float64),
        L_protrude=0.11,
        warm_start=True,
        override_z0_bar=z0,
        return_stats=True,
        stats_include_jacobian=False,
    )
    if hasattr(z, "block_until_ready"):
        z.block_until_ready()
    wall_s = time.perf_counter() - t0
    return {
        "z": z,
        "params": params,
        "ok": bool(ok),
        "stats": stats,
        "wall_s": float(wall_s),
        "centerline": extract_centerline_dim(z, M_list=params.M_list, scales=params.scales),
        "sites": engine.query_sites(z_bar=z, scales=params.scales),
    }


def _geometry_metrics(oracle: dict[str, Any], shooting: ShootingResult) -> dict[str, float]:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = _arc_resample(shooting.centerline, len(oracle_cl))
    diffs = np.linalg.norm(shoot_cl - oracle_cl, axis=1)
    sites = oracle["sites"]
    rigid_p_oracle = np.asarray(jax.device_get(sites["rigid_mid_p_dim"]), dtype=float)
    rigid_q_oracle = np.asarray(jax.device_get(sites["rigid_mid_Q_wxyz"]), dtype=float)
    rigid_count = min(len(rigid_p_oracle), len(shooting.rigid_mid_p))
    if rigid_count:
        rigid_pos_err = float(np.sqrt(np.mean(np.linalg.norm(shooting.rigid_mid_p[:rigid_count] - rigid_p_oracle[:rigid_count], axis=1) ** 2)))
        axis_errs = []
        for i in range(rigid_count):
            a = _axis_from_q(shooting.rigid_mid_q[i])
            b = _axis_from_q(rigid_q_oracle[i])
            axis_errs.append(np.linalg.norm(np.cross(b, a)))
        axis_err = float(np.sqrt(np.mean(np.asarray(axis_errs) ** 2)))
    else:
        rigid_pos_err = 0.0
        axis_err = 0.0
    return {
        "tip_error": float(np.linalg.norm(shooting.tip_p - np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float))),
        "rms_centerline_error": float(np.sqrt(np.mean(diffs**2))),
        "max_centerline_error": float(np.max(diffs)),
        "rigid_midpoint_rms_error": rigid_pos_err,
        "axis_proxy_rms_error": axis_err,
        "boundary_residual_norm": float(np.linalg.norm(shooting.boundary_residual)),
    }


def _task_residual_from_oracle(sites: dict[str, Any], residual_type: str, target_tip: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    pieces = []
    if residual_type in ("position", "position_axis"):
        pieces.append(np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float) - target_tip)
    if residual_type in ("axis", "position_axis"):
        cur_axis = np.asarray(jax.device_get(axis_from_quat(sites["rigid_mid_Q_wxyz"][0])), dtype=float)
        cur_axis = cur_axis / max(float(np.linalg.norm(cur_axis)), 1e-12)
        des_axis = target_axis / max(float(np.linalg.norm(target_axis)), 1e-12)
        pieces.append(np.cross(des_axis, cur_axis))
    return np.concatenate([piece.reshape(-1) for piece in pieces], axis=0)


def _task_residual_from_shooting(shooting: ShootingResult, residual_type: str, target_tip: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    pieces = []
    if residual_type in ("position", "position_axis"):
        pieces.append(shooting.tip_p - target_tip)
    if residual_type in ("axis", "position_axis"):
        cur_axis = _axis_from_q(shooting.rigid_mid_q[0])
        des_axis = target_axis / max(float(np.linalg.norm(target_axis)), 1e-12)
        pieces.append(np.cross(des_axis, cur_axis))
    return np.concatenate([piece.reshape(-1) for piece in pieces], axis=0)


def _svd_summary(J: np.ndarray, cutoff_ratio: float) -> dict[str, Any]:
    singular = np.linalg.svd(J, compute_uv=False)
    if singular.size == 0:
        return {"singular_values": [], "effective_rank": 0, "condition_number": None, "truncated_count": 0}
    cutoff = float(cutoff_ratio) * float(np.max(singular))
    kept = singular > cutoff
    kept_vals = singular[kept]
    cond = None if kept_vals.size == 0 else float(np.max(kept_vals) / max(float(np.min(kept_vals)), 1e-300))
    return {
        "singular_values": singular.tolist(),
        "effective_rank": int(np.sum(kept)),
        "condition_number": cond,
        "truncated_count": int(singular.size - np.sum(kept)),
    }


def _grade(rel_error: float) -> str:
    if rel_error < 0.2:
        return "usable"
    if rel_error < 0.5:
        return "diagnostic-only"
    return "untrusted"


def _plot_centerline(case_dir: Path, oracle_cl: np.ndarray, shooting_cl: np.ndarray, title: str) -> None:
    shoot_rs = _arc_resample(shooting_cl, len(oracle_cl))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2, label="oracle FK")
    ax.plot(shoot_rs[:, 0], shoot_rs[:, 1], shoot_rs[:, 2], "r--", lw=2, label="physical shooting")
    ax.scatter(oracle_cl[0, 0], oracle_cl[0, 1], oracle_cl[0, 2], c="blue", s=30, label="base")
    ax.scatter(oracle_cl[-1, 0], oracle_cl[-1, 1], oracle_cl[-1, 2], c="green", s=40, label="oracle tip")
    ax.scatter(shoot_rs[-1, 0], shoot_rs[-1, 1], shoot_rs[-1, 2], c="red", s=30, label="shooting tip")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "oracle_vs_physical_shooting_centerline.png", dpi=180)
    plt.close(fig)


def _plot_residual_history(case_dir: Path, history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(len(history)), history, marker=".")
    ax.set_title("Boundary residual history")
    ax.set_xlabel("residual evaluation")
    ax.set_ylabel("||free-end wrench residual||")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / "boundary_residual_history.png", dpi=180)
    plt.close(fig)


def _plot_singular(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        values = row["singular_values"]
        ax.semilogy(range(1, len(values) + 1), values, marker="o", label=row["residual_type"])
    ax.set_title("Physical shooting task Jacobian singular values")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "singular_values.png", dpi=180)
    plt.close(fig)


def _plot_fd_errors(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['residual_type']}\\n{r['direction']} eps={r['eps']}" for r in rows]
    vals = [r["relative_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 5))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0.2, color="green", linestyle="--", label="usable")
    ax.axhline(0.5, color="orange", linestyle="--", label="diagnostic-only")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title("Physical shooting FD audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "fd_audit_errors.png", dpi=180)
    plt.close(fig)


def run_case(
    *,
    M_list: tuple[int, int, int],
    residual_types: list[str],
    jac_eps: float,
    audit_eps_values: list[float],
    cutoff_ratio: float,
    max_iter: int,
    tol: float,
    shooting_max_nfev: int,
    steps_per_flex: int,
    active_indices: list[int],
    fd_reuse_base_y: bool,
    base_only: bool,
    use_oracle_proximal_guess: bool,
) -> dict[str, Any]:
    case_dir = OUT_DIR / f"M{''.join(str(m) for m in M_list)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    engine = _build_engine(M_list, max_iter=max_iter, tol=tol)
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)

    base = _solve_oracle(engine, I0, z0=None)
    y0_base = _proximal_y_from_oracle(base["z"], base["params"]) if use_oracle_proximal_guess else None
    base_shoot = _solve_shooting(base["params"], y0=y0_base, max_nfev=shooting_max_nfev, steps_per_flex=steps_per_flex)
    geometry = _geometry_metrics(base, base_shoot)

    target_tip = np.asarray(jax.device_get(base["sites"]["tip_p_dim"]), dtype=float)
    target_axis = np.asarray(jax.device_get(axis_from_quat(base["sites"]["rigid_mid_Q_wxyz"][0])), dtype=float)

    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def eval_current(delta: np.ndarray) -> dict[str, Any]:
        key = tuple(np.round(delta, 12).tolist())
        if key not in cache:
            oracle = _solve_oracle(engine, I0 + delta, z0=base["z"])
            if fd_reuse_base_y:
                shooting = _evaluate_shooting_fixed_y(oracle["params"], base_shoot.y, steps_per_flex=steps_per_flex)
            else:
                shooting = _solve_shooting(oracle["params"], y0=base_shoot.y, max_nfev=shooting_max_nfev, steps_per_flex=steps_per_flex)
            cache[key] = {"oracle": oracle, "shooting": shooting}
        return cache[key]

    nI = int(I0.size)
    eye = np.eye(nI, dtype=float)
    jacobian_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    jacobians: dict[str, np.ndarray] = {}
    if not base_only:
        for residual_type in residual_types:
            J_cols = []
            for col in active_indices:
                plus = eval_current(float(jac_eps) * eye[col])["shooting"]
                minus = eval_current(-float(jac_eps) * eye[col])["shooting"]
                rp = _task_residual_from_shooting(plus, residual_type, target_tip, target_axis)
                rm = _task_residual_from_shooting(minus, residual_type, target_tip, target_axis)
                J_cols.append((rp - rm) / (2.0 * float(jac_eps)))
            J = np.stack(J_cols, axis=1)
            jacobians[residual_type] = J
            svd = _svd_summary(J, cutoff_ratio)
            jacobian_rows.append(
                {
                    "M_list": ",".join(str(m) for m in M_list),
                    "residual_type": residual_type,
                    "jac_eps": float(jac_eps),
                    "cutoff_ratio": float(cutoff_ratio),
                    "residual_dim": int(J.shape[0]),
                    "active_current_indices": active_indices,
                    "fd_reuse_base_y": bool(fd_reuse_base_y),
                    **svd,
                }
            )

            for direction_name, direction in _directions(active_indices):
                pred = J @ direction[np.asarray(active_indices, dtype=int)]
                for eps in audit_eps_values:
                    plus = eval_current(float(eps) * direction)["oracle"]
                    minus = eval_current(-float(eps) * direction)["oracle"]
                    rp = _task_residual_from_oracle(plus["sites"], residual_type, target_tip, target_axis)
                    rm = _task_residual_from_oracle(minus["sites"], residual_type, target_tip, target_axis)
                    fd = (rp - rm) / (2.0 * float(eps))
                    diff = float(np.linalg.norm(pred - fd))
                    fd_norm = max(float(np.linalg.norm(fd)), 1e-12)
                    rel = diff / fd_norm
                    audit_rows.append(
                        {
                            "M_list": ",".join(str(m) for m in M_list),
                            "residual_type": residual_type,
                            "direction": direction_name,
                            "eps": float(eps),
                            "jac_eps": float(jac_eps),
                            "relative_error": float(rel),
                            "difference_norm": float(diff),
                            "fd_norm": float(fd_norm),
                            "pred_norm": float(np.linalg.norm(pred)),
                            "grade": _grade(rel),
                            "fd_reuse_base_y": bool(fd_reuse_base_y),
                        }
                    )

    np.save(case_dir / "oracle_centerline.npy", np.asarray(base["centerline"], dtype=float))
    np.save(case_dir / "physical_shooting_centerline.npy", base_shoot.centerline)
    np.save(case_dir / "shooting_state_xi.npy", base_shoot.xi)
    np.savez(
        case_dir / "physical_shooting_data.npz",
        y=base_shoot.y,
        xi=base_shoot.xi,
        boundary_residual=base_shoot.boundary_residual,
        oracle_centerline=np.asarray(base["centerline"], dtype=float),
        shooting_centerline=base_shoot.centerline,
        rigid_mid_p=base_shoot.rigid_mid_p,
        rigid_mid_q=base_shoot.rigid_mid_q,
    )
    with (case_dir / "jacobians.json").open("w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in jacobians.items()}, f, indent=2)
    _plot_centerline(case_dir, np.asarray(base["centerline"], dtype=float), base_shoot.centerline, f"M={M_list} oracle vs physical shooting")
    _plot_residual_history(case_dir, base_shoot.residual_history)
    _plot_singular(case_dir, jacobian_rows)
    _plot_fd_errors(case_dir, audit_rows)

    return {
        "case_dir": str(case_dir.relative_to(OUT_DIR)),
        "M_list": M_list,
        "base_ok": bool(base["ok"]),
        "base_normE": float(base["stats"].normE),
        "base_wall_s": float(base["wall_s"]),
        "shooting_state_dim": 6,
        "shooting_state_type": "proximal_force_moment",
        "active_current_indices": active_indices,
        "steps_per_flex": int(steps_per_flex),
        "fd_reuse_base_y": bool(fd_reuse_base_y),
        "base_only": bool(base_only),
        "use_oracle_proximal_guess": bool(use_oracle_proximal_guess),
        "shooting_success": bool(base_shoot.success),
        "shooting_cost": float(base_shoot.cost),
        "shooting_nfev": int(base_shoot.nfev),
        "shooting_wall_s": float(base_shoot.wall_s),
        "geometry": geometry,
        "jacobian_rows": jacobian_rows,
        "audit_rows": audit_rows,
        "solve_count": int(len(cache) + 1),
        "cache_oracle_wall_s": float(sum(item["oracle"]["wall_s"] for item in cache.values())),
        "cache_shooting_wall_s": float(sum(item["shooting"].wall_s for item in cache.values())),
    }


def _write_outputs(payload: dict[str, Any]) -> None:
    rows = [row for case in payload["cases"] for row in case["audit_rows"]]
    payload["row_count"] = len(rows)
    payload["grade_counts"] = {grade: sum(1 for row in rows if row["grade"] == grade) for grade in ("usable", "diagnostic-only", "untrusted")}
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted(rows[0].keys()) if rows else []
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--M_list", default="1,1,1")
    parser.add_argument("--max_iter", type=int, default=180)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--jac_eps", type=float, default=0.02)
    parser.add_argument("--audit_eps", default="0.02,0.05")
    parser.add_argument("--cutoff_ratio", type=float, default=1e-5)
    parser.add_argument("--shooting_max_nfev", type=int, default=80)
    parser.add_argument("--steps_per_flex", type=int, default=8)
    parser.add_argument("--active_current_indices", default="5,6")
    parser.add_argument("--residual_types", default="position,axis,position_axis")
    parser.add_argument("--fd_reuse_base_y", action="store_true")
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--use_oracle_proximal_guess", action="store_true")
    args = parser.parse_args()

    residual_types = [x.strip() for x in str(args.residual_types).split(",") if x.strip()]
    active_indices = [int(float(x.strip())) for x in str(args.active_current_indices).split(",") if x.strip()]
    cases = [_parse_m_list(args.M_list)]
    if args.mode == "full":
        cases.append((3, 3, 3))

    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "python": sys.executable,
            "mode": args.mode,
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "jac_eps": float(args.jac_eps),
            "audit_eps_values": [float(x.strip()) for x in str(args.audit_eps).split(",") if x.strip()],
            "cutoff_ratio": float(args.cutoff_ratio),
            "shooting_max_nfev": int(args.shooting_max_nfev),
            "steps_per_flex": int(args.steps_per_flex),
            "active_current_indices": active_indices,
            "residual_types": residual_types,
            "non_production": True,
            "physics_approximation": "RK4 IVP with proximal force/moment shooting; rigid magnetic/gravity loads reuse FK callbacks; no distributed magnetic load on flexible segments.",
            "fd_reuse_base_y": bool(args.fd_reuse_base_y),
            "base_only": bool(args.base_only),
            "use_oracle_proximal_guess": bool(args.use_oracle_proximal_guess),
        },
        "cases": [],
    }
    audit_eps_values = payload["metadata"]["audit_eps_values"]
    for M_list in cases:
        payload["cases"].append(
            run_case(
                M_list=M_list,
                residual_types=residual_types,
                jac_eps=float(args.jac_eps),
                audit_eps_values=audit_eps_values,
                cutoff_ratio=float(args.cutoff_ratio),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                shooting_max_nfev=int(args.shooting_max_nfev),
                steps_per_flex=int(args.steps_per_flex),
                active_indices=active_indices,
                fd_reuse_base_y=bool(args.fd_reuse_base_y),
                base_only=bool(args.base_only),
                use_oracle_proximal_guess=bool(args.use_oracle_proximal_guess),
            )
        )
        _write_outputs(payload)
    _write_outputs(payload)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
