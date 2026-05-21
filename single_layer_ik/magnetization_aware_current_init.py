from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table, interpolate_A_checked

jax.config.update("jax_enable_x64", True)


ArrayLike = Sequence[float] | np.ndarray


def _normalize(v: np.ndarray, *, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1,)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is None:
            return np.zeros_like(v)
        fb = np.asarray(fallback, dtype=np.float64).reshape(v.shape)
        fb_n = float(np.linalg.norm(fb))
        return fb / max(fb_n, 1e-12)
    return v / n


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if str(x).strip() != ""]


def parse_m_body_list(s: str) -> list[np.ndarray]:
    parts = [p.strip() for p in str(s).split(";") if p.strip() != ""]
    out: list[np.ndarray] = []
    for p in parts:
        vals = _parse_csv_floats(p)
        if len(vals) != 3:
            raise ValueError(f"Each m_body entry must contain 3 floats, got {len(vals)} from: {p}")
        out.append(np.asarray(vals, dtype=np.float64).reshape(3,))
    if not out:
        raise ValueError("m_body_list is empty")
    return out


def _quat_to_rot(q_wxyz: ArrayLike) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4,)
    qn = float(np.linalg.norm(q))
    if qn < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / qn
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _skew(v: ArrayLike) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3,)
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _estimate_tip_position(p0: np.ndarray, tangent_world: np.ndarray, x_init: float) -> np.ndarray:
    return p0 + float(x_init) * tangent_world


def _estimate_magnet_centers(
    *,
    p0: np.ndarray,
    tangent_world: np.ndarray,
    x_init: float,
    rigid_lengths: Sequence[float],
    magnet_count: int,
) -> np.ndarray:
    rigid_lengths = [float(v) for v in rigid_lengths]
    if rigid_lengths and len(rigid_lengths) == magnet_count:
        total_rigid = float(sum(rigid_lengths))
        start_s = max(float(x_init) - total_rigid, 0.0)
        centers = []
        s_cursor = start_s
        for L in rigid_lengths:
            centers.append(s_cursor + 0.5 * L)
            s_cursor += L
        return np.asarray([p0 + s * tangent_world for s in centers], dtype=np.float64)

    if magnet_count <= 1:
        return np.asarray([p0 + 0.95 * float(x_init) * tangent_world], dtype=np.float64)

    start = 0.80 * float(x_init)
    end = 0.98 * float(x_init)
    ss = np.linspace(start, end, magnet_count)
    return np.asarray([p0 + s * tangent_world for s in ss], dtype=np.float64)


def _build_segment_sensitivity(
    *,
    A_full: np.ndarray,
    m_world: np.ndarray,
) -> np.ndarray:
    A_B = np.asarray(A_full[:3, :], dtype=np.float64)
    return _skew(m_world) @ A_B


def _solve_regularized_current(
    *,
    S: np.ndarray,
    tau_des: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    gram = S.T @ S + float(ridge_lambda) * np.eye(S.shape[1], dtype=np.float64)
    rhs = S.T @ tau_des
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _apply_top_k(I: np.ndarray, top_k_coils: Optional[int]) -> np.ndarray:
    if top_k_coils is None or int(top_k_coils) >= I.size:
        return I.copy()
    top_k = max(int(top_k_coils), 1)
    idx = np.argsort(np.abs(I))[-top_k:]
    out = np.zeros_like(I)
    out[idx] = I[idx]
    return out


def _rescale_and_clip(
    *,
    I_raw: np.ndarray,
    I_magnitude: float,
    I_max: float,
    min_coil_fraction: float,
) -> np.ndarray:
    I = np.asarray(I_raw, dtype=np.float64).reshape(-1,)
    l1 = float(np.sum(np.abs(I)))
    if l1 > 1e-12 and float(I_magnitude) > 0.0:
        I = I * (float(I_magnitude) / l1)
    I = np.clip(I, -float(I_max), float(I_max))

    l1 = float(np.sum(np.abs(I)))
    if l1 > 1e-12 and float(min_coil_fraction) > 0.0:
        thresh = float(min_coil_fraction) * l1
        I = np.where(np.abs(I) >= thresh, I, 0.0)

    l1 = float(np.sum(np.abs(I)))
    if l1 > 1e-12 and float(I_magnitude) > 0.0:
        I = I * (float(I_magnitude) / l1)
    return np.clip(I, -float(I_max), float(I_max))


def generate_magnetization_aware_current_init(
    *,
    p_des: np.ndarray,
    actuation_table_path: str | Path,
    p0: np.ndarray,
    x_init: float,
    Q0_wxyz: ArrayLike,
    axis_body: ArrayLike,
    m_body_list: str | Sequence[ArrayLike],
    rigid_lengths: Sequence[float],
    I_magnitude: float = 15.0,
    I_max: float = 50.0,
    min_coil_fraction: float = 0.05,
    top_k_coils: Optional[int] = None,
    ridge_lambda: float = 1e-6,
    I_ref_scale: float = 0.5,
    torque_gain: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    p_des = np.asarray(p_des, dtype=np.float64).reshape(3,)
    p0 = np.asarray(p0, dtype=np.float64).reshape(3,)
    R0 = _quat_to_rot(Q0_wxyz)
    tangent_world = _normalize(R0 @ _normalize(np.asarray(axis_body, dtype=np.float64).reshape(3,), fallback=np.array([1.0, 0.0, 0.0])))

    if isinstance(m_body_list, str):
        m_body_vecs = parse_m_body_list(m_body_list)
    else:
        m_body_vecs = [np.asarray(v, dtype=np.float64).reshape(3,) for v in m_body_list]
    if not m_body_vecs:
        raise ValueError("m_body_list is empty")

    p_tip0 = _estimate_tip_position(p0, tangent_world, x_init)
    target_dir = _normalize(p_des - p_tip0, fallback=p_des - p0)
    bend_axis = np.cross(tangent_world, target_dir)
    bend_axis_norm = float(np.linalg.norm(bend_axis))
    tau_des = float(torque_gain) * _normalize(bend_axis, fallback=np.zeros(3, dtype=np.float64))

    magnet_centers = _estimate_magnet_centers(
        p0=p0,
        tangent_world=tangent_world,
        x_init=float(x_init),
        rigid_lengths=rigid_lengths,
        magnet_count=len(m_body_vecs),
    )

    table = load_actuation_table(str(actuation_table_path), dtype=jnp.float64)

    m_world_list = [R0 @ m_body for m_body in m_body_vecs]
    magnitudes = np.asarray([np.linalg.norm(m) for m in m_body_vecs], dtype=np.float64)
    if float(np.sum(magnitudes)) < 1e-12:
        weights = np.ones(len(m_body_vecs), dtype=np.float64) / float(len(m_body_vecs))
    else:
        weights = magnitudes / float(np.sum(magnitudes))

    S_parts = []
    A_samples = []
    for center, m_world, w in zip(magnet_centers, m_world_list, weights):
        A_full = interpolate_A_checked(table, center)
        A_samples.append(np.asarray(A_full, dtype=np.float64))
        S_parts.append(float(w) * _build_segment_sensitivity(A_full=np.asarray(A_full, dtype=np.float64), m_world=np.asarray(m_world, dtype=np.float64)))
    S_total = np.sum(np.stack(S_parts, axis=0), axis=0)

    I_raw = _solve_regularized_current(S=S_total, tau_des=tau_des, ridge_lambda=float(ridge_lambda))

    if float(np.sum(np.abs(I_raw))) < 1e-12:
        column_scores = S_total.T @ tau_des
        I_raw = column_scores

    I_raw = _apply_top_k(I_raw, top_k_coils)
    I_init = _rescale_and_clip(
        I_raw=I_raw,
        I_magnitude=float(I_magnitude),
        I_max=float(I_max),
        min_coil_fraction=float(min_coil_fraction),
    )
    I_ref = float(I_ref_scale) * I_init

    info = {
        "mode": "magnetization_aware_ls",
        "p_des": p_des.tolist(),
        "p0": p0.tolist(),
        "p_tip0": p_tip0.tolist(),
        "target_direction": target_dir.tolist(),
        "tangent_world": tangent_world.tolist(),
        "bend_axis": bend_axis.tolist(),
        "bend_axis_norm": bend_axis_norm,
        "tau_des": tau_des.tolist(),
        "weights": weights.tolist(),
        "magnet_centers": magnet_centers.tolist(),
        "m_body_list": [m.tolist() for m in m_body_vecs],
        "m_world_list": [m.tolist() for m in m_world_list],
        "ridge_lambda": float(ridge_lambda),
        "I_magnitude": float(I_magnitude),
        "I_max": float(I_max),
        "I_raw": I_raw.tolist(),
        "I_init": I_init.tolist(),
        "I_ref": I_ref.tolist(),
        "I_ref_scale": float(I_ref_scale),
        "active_coils": int(np.sum(np.abs(I_init) > 1e-6)),
        "max_current": float(np.max(np.abs(I_init))) if I_init.size else 0.0,
        "S_total": S_total.tolist(),
        "A_samples_first3": [A[:3, :].tolist() for A in A_samples],
    }
    return I_init, info
