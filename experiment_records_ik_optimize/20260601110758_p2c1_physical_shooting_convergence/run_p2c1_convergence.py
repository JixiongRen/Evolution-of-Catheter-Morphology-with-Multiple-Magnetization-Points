"""Run P2c.1 physical shooting convergence candidates.

This experiment-local runner keeps physical shooting out of production IK. It
tests low-dimensional force/moment shooting candidates against the optimized FK
oracle and records whether the base geometry reaches the requested stages.
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
P2C_DIR = ROOT / "experiment_records_ik_optimize" / "20260601100855_p2c_physical_shooting_solver"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))
if str(P2C_DIR) not in sys.path:
    sys.path.insert(0, str(P2C_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_kinematics_optimized.nondim import x_bar_to_dim
from forward_kinematics_optimized.utils_nondim import unpack_z_bar_jax
from segments_nondim import cosserat_rhs_dim, rigid_state_along_dim
from external_wrench_nondim import compute_external_wrench_total_rigid
from inverse_kinematics.ik_observations import axis_from_quat
from basics_nondim import quat_derivative, quat_to_rotmat

import run_p2c_physical_shooting_solver as p2c

jax.config.update("jax_enable_x64", True)
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CandidateSpec:
    """One convergence candidate configuration."""

    name: str
    state_mode: str
    residual_mode: str
    max_nfev: int
    centerline_weight: float
    tip_weight: float
    rigid_weight: float
    axis_weight: float
    boundary_weight: float
    loss: str = "linear"


@dataclass
class CandidateResult:
    """Candidate solve result."""

    spec: CandidateSpec
    y: np.ndarray
    centerline: np.ndarray
    rigid_mid_p: np.ndarray
    rigid_mid_q: np.ndarray
    tip_p: np.ndarray
    tip_q: np.ndarray
    boundary: np.ndarray
    residual_history: list[float]
    success: bool
    cost: float
    nfev: int
    wall_s: float


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


def _stage(metrics: dict[str, float], length: float) -> str:
    """Classify endpoint-first agreement stages.

    The shooting audit prioritizes tip position and rigid-section point
    agreement. Whole-axis centerline mismatch is recorded as morphology drift,
    but it does not veto a candidate by itself.
    """
    scale = max(float(length), 1e-12)
    tip_ratio = metrics["tip_error"] / scale
    rigid_ratio = metrics["rigid_midpoint_rms_error"] / scale
    axis_error = metrics["axis_proxy_rms_error"]
    endpoint_ratio = max(tip_ratio, rigid_ratio)
    if endpoint_ratio < 0.02 and axis_error < 0.15:
        return "C"
    if endpoint_ratio < 0.05 and axis_error < 0.30:
        return "B"
    if endpoint_ratio < 0.10 and axis_error < 0.75:
        return "A"
    return "fail"


def _endpoint_score(row: dict[str, Any], length: float) -> float:
    """Score candidates with endpoint agreement first and morphology second."""
    scale = max(float(length), 1e-12)
    endpoint_ratio = max(
        float(row["tip_error"]) / scale,
        float(row["rigid_midpoint_rms_error"]) / scale,
    )
    morphology_ratio = float(row["rms_centerline_error"]) / scale
    axis_error = min(float(row["axis_proxy_rms_error"]), 1.0)
    boundary_norm = float(row.get("boundary_residual_norm", 0.0) or 0.0)
    return endpoint_ratio + 0.10 * morphology_ratio + 0.05 * axis_error + 0.002 * boundary_norm


def _state_np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float).reshape(-1)


def _normalize_state(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(13).copy()
    q = x[3:7]
    x[3:7] = q / max(float(np.linalg.norm(q)), 1e-12)
    return x


def _initial_state(params: Any, y_block: np.ndarray) -> np.ndarray:
    x = np.zeros((13,), dtype=float)
    x[0:3] = np.asarray(jax.device_get(params.p0_bar), dtype=float).reshape(3) * float(params.scales.L_ref)
    x[3:7] = _state_np(params.Q0)
    x[7:10] = y_block[0:3] * float(params.scales.F_ref)
    x[10:13] = y_block[3:6] * float(params.scales.M_ref)
    return _normalize_state(x)


def _y_guess_from_oracle(z_bar: Any, params: Any, state_mode: str) -> np.ndarray:
    x_nodes_list, _k_list, _x_rigid_list = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    blocks = []
    for i, nodes in enumerate(x_nodes_list):
        x_dim = x_bar_to_dim(nodes[0], params.scales)
        f0 = np.asarray(jax.device_get(x_dim[7:10]), dtype=float)
        tau0 = np.asarray(jax.device_get(x_dim[10:13]), dtype=float)
        blocks.append(np.concatenate([f0 / float(params.scales.F_ref), tau0 / float(params.scales.M_ref)]))
    if state_mode == "proximal":
        return blocks[0]
    if state_mode == "segment_wrench":
        return np.concatenate(blocks)
    if state_mode == "segment_twist":
        return _twist_guess_from_oracle(z_bar, params)
    raise ValueError(f"unknown state_mode: {state_mode}")


def _twist_guess_from_oracle(z_bar: Any, params: Any) -> np.ndarray:
    """Estimate per-segment constant body strain from oracle endpoints."""
    x_nodes_list, _k_list, _x_rigid_list = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    blocks = []
    for i, nodes in enumerate(x_nodes_list):
        x0 = x_bar_to_dim(nodes[0], params.scales)
        x1 = x_bar_to_dim(nodes[-1], params.scales)
        p0 = np.asarray(jax.device_get(x0[0:3]), dtype=float)
        p1 = np.asarray(jax.device_get(x1[0:3]), dtype=float)
        q0 = np.asarray(jax.device_get(x0[3:7]), dtype=float)
        R0 = np.asarray(jax.device_get(quat_to_rotmat(jnp.asarray(q0, dtype=jnp.float64))), dtype=float)
        L = float(np.asarray(jax.device_get(params.L1_dim)).reshape(())) if i == 0 else float(params.flex[i].length)
        v = R0.T @ ((p1 - p0) / max(L, 1e-12))
        # Small initial curvature; optimizer will adjust.
        u = np.zeros(3, dtype=float)
        blocks.append(np.concatenate([v, u]))
    return np.concatenate(blocks)


def _rk4_step(rhs, x: np.ndarray, sigma: float, h: float) -> np.ndarray:
    k1 = rhs(x, sigma)
    k2 = rhs(_normalize_state(x + 0.5 * h * k1), sigma + 0.5 * h)
    k3 = rhs(_normalize_state(x + 0.5 * h * k2), sigma + 0.5 * h)
    k4 = rhs(_normalize_state(x + h * k3), sigma + h)
    return _normalize_state(x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _forward_modes(params: Any, y: np.ndarray, state_mode: str, *, steps_per_flex: int) -> dict[str, np.ndarray]:
    N = len(params.flex)
    y = np.asarray(y, dtype=float).reshape(-1)
    if state_mode == "segment_twist":
        return _forward_segment_twist(params, y.reshape(N, 6), steps_per_flex=steps_per_flex)
    if state_mode == "proximal":
        blocks = np.repeat(y.reshape(1, 6), N, axis=0)
        reset_each_segment = False
    elif state_mode == "segment_wrench":
        blocks = y.reshape(N, 6)
        reset_each_segment = True
    else:
        raise ValueError(f"unknown state_mode: {state_mode}")

    x = _initial_state(params, blocks[0])
    centerline = [x[0:3].copy()]
    rigid_mid_p = []
    rigid_mid_q = []
    for i in range(N):
        if reset_each_segment:
            x[7:10] = blocks[i, 0:3] * float(params.scales.F_ref)
            x[10:13] = blocks[i, 3:6] * float(params.scales.M_ref)
            x = _normalize_state(x)
        flex = params.flex[i]
        L = float(np.asarray(jax.device_get(params.L1_dim)).reshape(())) if i == 0 else float(flex.length)
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
        x_mid = rigid_state_along_dim(
            x_prox_dim=jnp.asarray(x_prox, dtype=jnp.float64),
            sigma=0.5 * float(rigid.length),
            rigid=rigid,
            f_ext_total_dim=f_total + params.f_ext_list[i],
            tau_ext_total_dim=tau_total + params.tau_ext_list[i],
        )
        x_dist = rigid_state_along_dim(
            x_prox_dim=jnp.asarray(x_prox, dtype=jnp.float64),
            sigma=float(rigid.length),
            rigid=rigid,
            f_ext_total_dim=f_total + params.f_ext_list[i],
            tau_ext_total_dim=tau_total + params.tau_ext_list[i],
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
        "boundary": boundary,
    }


def _forward_segment_twist(params: Any, blocks: np.ndarray, *, steps_per_flex: int) -> dict[str, np.ndarray]:
    """Integrate per-segment constant body strain modes."""
    x = np.zeros((13,), dtype=float)
    x[0:3] = np.asarray(jax.device_get(params.p0_bar), dtype=float).reshape(3) * float(params.scales.L_ref)
    x[3:7] = _state_np(params.Q0)
    x = _normalize_state(x)
    centerline = [x[0:3].copy()]
    rigid_mid_p = []
    rigid_mid_q = []
    for i in range(len(params.flex)):
        v_body = blocks[i, 0:3]
        u_body = blocks[i, 3:6]
        L = float(np.asarray(jax.device_get(params.L1_dim)).reshape(())) if i == 0 else float(params.flex[i].length)
        n_steps = max(int(steps_per_flex), int(params.M_list[i]) * int(steps_per_flex))
        h = L / max(n_steps, 1)

        def rhs(x_np: np.ndarray) -> np.ndarray:
            q = x_np[3:7]
            R = np.asarray(jax.device_get(quat_to_rotmat(jnp.asarray(q, dtype=jnp.float64))), dtype=float)
            dx = np.zeros((13,), dtype=float)
            dx[0:3] = R @ v_body
            dx[3:7] = _state_np(quat_derivative(jnp.asarray(q, dtype=jnp.float64), jnp.asarray(R @ u_body, dtype=jnp.float64)))
            return dx

        for _ in range(n_steps):
            k1 = rhs(x)
            k2 = rhs(_normalize_state(x + 0.5 * h * k1))
            k3 = rhs(_normalize_state(x + 0.5 * h * k2))
            k4 = rhs(_normalize_state(x + h * k3))
            x = _normalize_state(x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
            centerline.append(x[0:3].copy())

        R = np.asarray(jax.device_get(quat_to_rotmat(jnp.asarray(x[3:7], dtype=jnp.float64))), dtype=float)
        axis_body = np.asarray(jax.device_get(params.rigid[i].v_star), dtype=float).reshape(3)
        rigid_mid_p.append(x[0:3] + 0.5 * float(params.rigid[i].length) * (R @ axis_body))
        rigid_mid_q.append(x[3:7].copy())
        x[0:3] = x[0:3] + float(params.rigid[i].length) * (R @ axis_body)
        centerline.append(x[0:3].copy())

    return {
        "centerline": np.asarray(centerline, dtype=float),
        "rigid_mid_p": np.asarray(rigid_mid_p, dtype=float),
        "rigid_mid_q": np.asarray(rigid_mid_q, dtype=float),
        "tip_p": x[0:3].copy(),
        "tip_q": x[3:7].copy(),
        "boundary": np.zeros((6,), dtype=float),
    }


def _axis_from_q(q: np.ndarray) -> np.ndarray:
    axis = np.asarray(jax.device_get(axis_from_quat(jnp.asarray(q, dtype=jnp.float64))), dtype=float).reshape(3)
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _arc_resample(centerline: np.ndarray, count: int) -> np.ndarray:
    return p2c._arc_resample(centerline, count)


def _metrics(oracle: dict[str, Any], out: dict[str, np.ndarray]) -> dict[str, float]:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = _arc_resample(out["centerline"], len(oracle_cl))
    diffs = np.linalg.norm(shoot_cl - oracle_cl, axis=1)
    sites = oracle["sites"]
    rigid_p_oracle = np.asarray(jax.device_get(sites["rigid_mid_p_dim"]), dtype=float)
    rigid_q_oracle = np.asarray(jax.device_get(sites["rigid_mid_Q_wxyz"]), dtype=float)
    rigid_count = min(len(rigid_p_oracle), len(out["rigid_mid_p"]))
    rigid_err = float("nan")
    axis_err = float("nan")
    if rigid_count:
        rigid_err = float(np.sqrt(np.mean(np.linalg.norm(out["rigid_mid_p"][:rigid_count] - rigid_p_oracle[:rigid_count], axis=1) ** 2)))
        axis_errs = []
        for i in range(rigid_count):
            axis_errs.append(np.linalg.norm(np.cross(_axis_from_q(rigid_q_oracle[i]), _axis_from_q(out["rigid_mid_q"][i]))))
        axis_err = float(np.sqrt(np.mean(np.asarray(axis_errs) ** 2)))
    return {
        "tip_error": float(np.linalg.norm(out["tip_p"] - np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float))),
        "rms_centerline_error": float(np.sqrt(np.mean(diffs**2))),
        "max_centerline_error": float(np.max(diffs)),
        "rigid_midpoint_rms_error": rigid_err,
        "axis_proxy_rms_error": axis_err,
        "boundary_residual_norm": float(np.linalg.norm(out["boundary"])),
    }


def _residual_vector(spec: CandidateSpec, oracle: dict[str, Any], out: dict[str, np.ndarray]) -> np.ndarray:
    oracle_cl = np.asarray(oracle["centerline"], dtype=float)
    shoot_cl = _arc_resample(out["centerline"], len(oracle_cl))
    sites = oracle["sites"]
    pieces = []
    if spec.centerline_weight:
        pieces.append(spec.centerline_weight * (shoot_cl - oracle_cl).reshape(-1))
    if spec.tip_weight:
        tip_target = np.asarray(jax.device_get(sites["tip_p_dim"]), dtype=float)
        pieces.append(spec.tip_weight * (out["tip_p"] - tip_target))
    if spec.rigid_weight:
        rigid_target = np.asarray(jax.device_get(sites["rigid_mid_p_dim"]), dtype=float)
        n = min(len(rigid_target), len(out["rigid_mid_p"]))
        pieces.append(spec.rigid_weight * (out["rigid_mid_p"][:n] - rigid_target[:n]).reshape(-1))
    if spec.axis_weight:
        rigid_q = np.asarray(jax.device_get(sites["rigid_mid_Q_wxyz"]), dtype=float)
        axis_parts = []
        for i in range(min(len(rigid_q), len(out["rigid_mid_q"]))):
            axis_parts.append(np.cross(_axis_from_q(rigid_q[i]), _axis_from_q(out["rigid_mid_q"][i])))
        if axis_parts:
            pieces.append(spec.axis_weight * np.concatenate(axis_parts))
    if spec.boundary_weight:
        pieces.append(spec.boundary_weight * out["boundary"])
    if not pieces:
        return out["boundary"]
    return np.concatenate([piece.reshape(-1) for piece in pieces])


def _solve_candidate(spec: CandidateSpec, oracle: dict[str, Any], *, steps_per_flex: int) -> CandidateResult:
    params = oracle["params"]
    y0 = _y_guess_from_oracle(oracle["z"], params, spec.state_mode)
    history: list[float] = []

    def residual(y: np.ndarray) -> np.ndarray:
        out = _forward_modes(params, y, spec.state_mode, steps_per_flex=steps_per_flex)
        r = _residual_vector(spec, oracle, out)
        history.append(float(np.linalg.norm(r)))
        return r

    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        y0,
        method="trf",
        x_scale=np.ones_like(y0),
        loss=spec.loss,
        max_nfev=int(spec.max_nfev),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    wall_s = time.perf_counter() - t0
    out = _forward_modes(params, opt.x, spec.state_mode, steps_per_flex=steps_per_flex)
    return CandidateResult(
        spec=spec,
        y=np.asarray(opt.x, dtype=float),
        centerline=out["centerline"],
        rigid_mid_p=out["rigid_mid_p"],
        rigid_mid_q=out["rigid_mid_q"],
        tip_p=out["tip_p"],
        tip_q=out["tip_q"],
        boundary=out["boundary"],
        residual_history=history,
        success=bool(opt.success),
        cost=float(opt.cost),
        nfev=int(opt.nfev),
        wall_s=float(wall_s),
    )


def _candidate_specs(max_nfev: int) -> list[CandidateSpec]:
    return [
        CandidateSpec("c0_free_end_6d", "proximal", "free_end", max_nfev, 0.0, 0.0, 0.0, 0.0, 1.0),
        CandidateSpec("c1_tip_rigid_6d", "proximal", "calib", max_nfev, 0.0, 80.0, 50.0, 2.0, 0.05),
        CandidateSpec("c2_centerline_6d", "proximal", "calib", max_nfev, 40.0, 80.0, 30.0, 0.0, 0.02),
        CandidateSpec("c3_segment_wrench_18d", "segment_wrench", "calib_modes", max_nfev, 40.0, 80.0, 40.0, 1.0, 0.02),
        CandidateSpec("c4_segment_wrench_soft_l1", "segment_wrench", "calib_modes", max_nfev, 40.0, 100.0, 40.0, 1.0, 0.01, loss="soft_l1"),
        CandidateSpec("c5_segment_centerline_strong", "segment_wrench", "calib_modes", max_nfev, 120.0, 120.0, 30.0, 0.3, 0.005),
        CandidateSpec("c6_segment_centerline_only", "segment_wrench", "calib_modes", max_nfev, 180.0, 80.0, 0.0, 0.0, 0.0),
        CandidateSpec("c7_segment_twist_centerline", "segment_twist", "curvature_modes", max_nfev, 180.0, 120.0, 20.0, 0.2, 0.0),
        CandidateSpec("c8_segment_twist_soft_l1", "segment_twist", "curvature_modes", max_nfev, 180.0, 120.0, 20.0, 0.2, 0.0, loss="soft_l1"),
    ]


def _plot_candidate(case_dir: Path, oracle_cl: np.ndarray, result: CandidateResult) -> None:
    shoot = _arc_resample(result.centerline, len(oracle_cl))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2, label="oracle FK")
    ax.plot(shoot[:, 0], shoot[:, 1], shoot[:, 2], "r--", lw=2, label=result.spec.name)
    ax.scatter(oracle_cl[0, 0], oracle_cl[0, 1], oracle_cl[0, 2], c="blue", s=30, label="base")
    ax.scatter(oracle_cl[-1, 0], oracle_cl[-1, 1], oracle_cl[-1, 2], c="green", s=40, label="oracle tip")
    ax.scatter(shoot[-1, 0], shoot[-1, 1], shoot[-1, 2], c="red", s=30, label="shooting tip")
    ax.set_title(result.spec.name)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / f"{result.spec.name}_oracle_vs_shooting.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(len(result.residual_history)), result.residual_history, marker=".")
    ax.set_title(f"{result.spec.name} residual history")
    ax.set_xlabel("eval")
    ax.set_ylabel("||residual||")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / f"{result.spec.name}_residual_history.png", dpi=180)
    plt.close(fig)


def _plot_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    names = [r["candidate"] for r in rows]
    rms = [r["rms_centerline_error"] for r in rows]
    tip = [r["tip_error"] for r in rows]
    rigid = [r["rigid_midpoint_rms_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.24, tip, width=0.24, label="tip")
    ax.bar(x, rigid, width=0.24, label="rigid midpoint")
    ax.bar(x + 0.24, rms, width=0.24, label="centerline RMS")
    ax.axhline(0.0055, color="orange", linestyle="--", label="5% endpoint gate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("error (m)")
    ax.set_title("Endpoint-first candidate comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--candidate_max_nfev", type=int, default=12)
    parser.add_argument("--candidate_names", default="")
    args = parser.parse_args()

    engine = p2c._build_engine((1, 1, 1), max_iter=int(args.max_iter), tol=float(args.tol))
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    oracle = p2c._solve_oracle(engine, I0, z0=None)
    length = float(np.sum(engine.rigid_lengths) + sum(engine._compute_flex_lengths(L_protrude=0.11)))
    case_dir = OUT_DIR / "M111"
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "oracle_centerline.npy", np.asarray(oracle["centerline"], dtype=float))

    rows = []
    best_row = None
    best_result: CandidateResult | None = None
    selected_names = {x.strip() for x in str(args.candidate_names).split(",") if x.strip()}
    specs = _candidate_specs(int(args.candidate_max_nfev))
    if selected_names:
        specs = [spec for spec in specs if spec.name in selected_names]
    for spec in specs:
        result = _solve_candidate(spec, oracle, steps_per_flex=int(args.steps_per_flex))
        out = {
            "centerline": result.centerline,
            "rigid_mid_p": result.rigid_mid_p,
            "rigid_mid_q": result.rigid_mid_q,
            "tip_p": result.tip_p,
            "boundary": result.boundary,
        }
        metrics = _metrics(oracle, out)
        stage = _stage(metrics, length)
        row = {
            "candidate": spec.name,
            "state_mode": spec.state_mode,
            "residual_mode": spec.residual_mode,
            "state_dim": int(result.y.size),
            "state_meaning": (
                "proximal wrench"
                if spec.state_mode == "proximal"
                else "per-segment constant body strain modes"
                if spec.state_mode == "segment_twist"
                else "per-segment proximal wrench modes"
            ),
            "centerline_weight": float(spec.centerline_weight),
            "tip_weight": float(spec.tip_weight),
            "rigid_weight": float(spec.rigid_weight),
            "axis_weight": float(spec.axis_weight),
            "boundary_weight": float(spec.boundary_weight),
            "loss": spec.loss,
            "success": bool(result.success),
            "wall_s": float(result.wall_s),
            "nfev": int(result.nfev),
            "stage": stage,
            "endpoint_first_score": _endpoint_score({**metrics}, length),
            **metrics,
        }
        rows.append(row)
        candidate_dir = case_dir / spec.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        np.save(candidate_dir / "shooting_centerline.npy", result.centerline)
        np.save(candidate_dir / "shooting_state_y.npy", result.y)
        np.save(candidate_dir / "boundary_residual.npy", result.boundary)
        np.save(candidate_dir / "residual_history.npy", np.asarray(result.residual_history, dtype=float))
        _plot_candidate(candidate_dir, np.asarray(oracle["centerline"], dtype=float), result)
        if best_row is None or row["endpoint_first_score"] < best_row["endpoint_first_score"]:
            best_row = row
            best_result = result

    _plot_summary(OUT_DIR, rows)
    payload = {
        "metadata": {
            "branch": _run(["git", "branch", "--show-current"]),
            "git_status_short": _run(["git", "status", "--short"]),
            "jax_devices": [str(device) for device in jax.devices()],
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "steps_per_flex": int(args.steps_per_flex),
            "candidate_max_nfev": int(args.candidate_max_nfev),
            "catheter_length": length,
            "non_production": True,
        },
        "oracle": {
            "base_ok": bool(oracle["ok"]),
            "base_normE": float(oracle["stats"].normE),
            "base_wall_s": float(oracle["wall_s"]),
        },
        "rows": rows,
        "best_candidate": best_row,
        "stage_a_reached": bool(best_row is not None and best_row["stage"] in ("A", "B", "C")),
        "stage_b_reached": bool(best_row is not None and best_row["stage"] in ("B", "C")),
        "fd_audit_run": False,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
