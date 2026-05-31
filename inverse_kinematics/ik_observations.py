"""Observation and residual utilities for multi-site IK tasks.

The utilities in this module are intentionally independent from the FK solver.
They operate on pose dictionaries returned by ``ForwardKinematicsEngine`` and
provide small, testable building blocks for future multi-site IK residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


_VALID_SITE_TYPES = {"tip", "rigid_midpoint"}
_VALID_RESIDUAL_TYPES = {"position", "axis", "so3", "pose"}


@dataclass(frozen=True)
class ObservationSpec:
    """Describe one IK observation residual.

    Attributes:
        site_type: Observation site, either ``"tip"`` or ``"rigid_midpoint"``.
        residual_type: Residual type, one of ``"position"``, ``"axis"``,
            ``"so3"``, or ``"pose"``.
        site_index: Rigid midpoint index. Ignored for ``site_type="tip"``.
        target_position: Target position in meters for position residuals.
        target_quat_wxyz: Target quaternion for SO(3) residuals.
        target_axis: Target world-frame axis for axis residuals. If omitted,
            it is derived from ``target_quat_wxyz`` and ``axis_body``.
        axis_body: Body-frame axis used by axis residuals.
        sigma_position: Position residual scale in meters.
        sigma_angle: Angular residual scale in radians.
        name: Optional label used in diagnostics.
    """

    site_type: str
    residual_type: str
    site_index: Optional[int] = None
    target_position: Optional[Sequence[float]] = None
    target_quat_wxyz: Optional[Sequence[float]] = None
    target_axis: Optional[Sequence[float]] = None
    axis_body: Sequence[float] = (0.0, 0.0, 1.0)
    sigma_position: float = 1.0
    sigma_angle: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.site_type not in _VALID_SITE_TYPES:
            raise ValueError(f"Unsupported site_type: {self.site_type}")
        if self.residual_type not in _VALID_RESIDUAL_TYPES:
            raise ValueError(f"Unsupported residual_type: {self.residual_type}")
        if self.site_type == "rigid_midpoint" and self.site_index is None:
            raise ValueError("site_index is required for rigid_midpoint observations")
        if float(self.sigma_position) <= 0.0:
            raise ValueError("sigma_position must be positive")
        if float(self.sigma_angle) <= 0.0:
            raise ValueError("sigma_angle must be positive")


@dataclass(frozen=True)
class ObservationResidual:
    """Computed residual for one observation spec."""

    spec: ObservationSpec
    residual: Array
    position_dim: Array
    quat_wxyz: Array


def normalize_vector(v: Array, eps: float = 1e-12) -> Array:
    """Return a unit vector with a small zero-norm guard."""
    v = jnp.asarray(v, dtype=jnp.float64).reshape(-1)
    norm = jnp.linalg.norm(v)
    return v / jnp.maximum(norm, float(eps))


def quat_normalize_wxyz(q: Array, eps: float = 1e-12) -> Array:
    """Normalize a quaternion in ``[w, x, y, z]`` order."""
    q = jnp.asarray(q, dtype=jnp.float64).reshape(4)
    return q / jnp.maximum(jnp.linalg.norm(q), float(eps))


def quat_to_rotmat_wxyz(q: Array) -> Array:
    """Convert a ``[w, x, y, z]`` quaternion to a rotation matrix."""
    w, x, y, z = quat_normalize_wxyz(q)
    return jnp.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=jnp.float64,
    )


def rotmat_to_so3_log(R: Array, eps: float = 1e-12) -> Array:
    """Map a rotation matrix to the SO(3) tangent vector.

    The implementation uses the standard closed form away from zero and a
    first-order skew approximation for small angles. Rotations close to pi are
    still mathematically ill-conditioned and should be handled by target
    continuation in production IK.
    """
    R = jnp.asarray(R, dtype=jnp.float64).reshape(3, 3)
    cos_theta = (jnp.trace(R) - 1.0) * 0.5
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    vee = jnp.asarray(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=jnp.float64,
    )
    small = theta < 1e-8
    scale = theta / (2.0 * jnp.maximum(jnp.sin(theta), float(eps)))
    return jnp.where(small, 0.5 * vee, scale * vee)


def quaternion_so3_residual(current_quat_wxyz: Array, target_quat_wxyz: Array, *, sigma_angle: float = 1.0) -> Array:
    """Return ``Log(R_target.T @ R_current) / sigma_angle``."""
    R_cur = quat_to_rotmat_wxyz(current_quat_wxyz)
    R_des = quat_to_rotmat_wxyz(target_quat_wxyz)
    return rotmat_to_so3_log(R_des.T @ R_cur) / float(sigma_angle)


def axis_from_quat(current_quat_wxyz: Array, axis_body: Sequence[float] = (0.0, 0.0, 1.0)) -> Array:
    """Return a world-frame axis defined by a body-frame axis and quaternion."""
    R = quat_to_rotmat_wxyz(current_quat_wxyz)
    axis = normalize_vector(jnp.asarray(axis_body, dtype=jnp.float64).reshape(3))
    return normalize_vector(R @ axis)


def axis_residual(
    current_quat_wxyz: Array,
    *,
    target_axis: Optional[Sequence[float]] = None,
    target_quat_wxyz: Optional[Sequence[float]] = None,
    axis_body: Sequence[float] = (0.0, 0.0, 1.0),
    sigma_angle: float = 1.0,
) -> Array:
    """Return a small-angle axis residual.

    The residual is ``cross(axis_desired, axis_current) / sigma_angle``. For a
    small positive rotation about x from the target pose, this convention agrees
    with ``Log(R_target.T @ R_current)``.
    """
    cur_axis = axis_from_quat(current_quat_wxyz, axis_body=axis_body)
    if target_axis is not None:
        des_axis = normalize_vector(jnp.asarray(target_axis, dtype=jnp.float64).reshape(3))
    elif target_quat_wxyz is not None:
        des_axis = axis_from_quat(jnp.asarray(target_quat_wxyz, dtype=jnp.float64), axis_body=axis_body)
    else:
        raise ValueError("axis residual requires target_axis or target_quat_wxyz")
    return jnp.cross(des_axis, cur_axis) / float(sigma_angle)


def extract_site_pose(sites: Mapping[str, Any], spec: ObservationSpec) -> tuple[Array, Array]:
    """Extract one site pose from a ``ForwardKinematicsEngine.query_sites`` dict."""
    if spec.site_type == "tip":
        return (
            jnp.asarray(sites["tip_p_dim"], dtype=jnp.float64).reshape(3),
            quat_normalize_wxyz(jnp.asarray(sites["tip_Q_wxyz"], dtype=jnp.float64)),
        )
    if spec.site_type == "rigid_midpoint":
        idx = int(spec.site_index)
        p_all = jnp.asarray(sites["rigid_mid_p_dim"], dtype=jnp.float64)
        q_all = jnp.asarray(sites["rigid_mid_Q_wxyz"], dtype=jnp.float64)
        if idx < 0 or idx >= int(p_all.shape[0]):
            raise IndexError(f"rigid_midpoint index {idx} out of range for {int(p_all.shape[0])} sites")
        return p_all[idx].reshape(3), quat_normalize_wxyz(q_all[idx])
    raise ValueError(f"Unsupported site_type: {spec.site_type}")


def compute_observation_residual(sites: Mapping[str, Any], spec: ObservationSpec) -> ObservationResidual:
    """Compute one observation residual from FK site poses."""
    p, q = extract_site_pose(sites, spec)
    pieces = []
    if spec.residual_type in ("position", "pose"):
        if spec.target_position is None:
            raise ValueError("position residual requires target_position")
        p_des = jnp.asarray(spec.target_position, dtype=jnp.float64).reshape(3)
        pieces.append((p - p_des) / float(spec.sigma_position))
    if spec.residual_type == "axis":
        pieces.append(
            axis_residual(
                q,
                target_axis=spec.target_axis,
                target_quat_wxyz=spec.target_quat_wxyz,
                axis_body=spec.axis_body,
                sigma_angle=float(spec.sigma_angle),
            )
        )
    if spec.residual_type in ("so3", "pose"):
        if spec.target_quat_wxyz is None:
            raise ValueError("SO(3) residual requires target_quat_wxyz")
        pieces.append(quaternion_so3_residual(q, jnp.asarray(spec.target_quat_wxyz), sigma_angle=float(spec.sigma_angle)))
    residual = jnp.concatenate([piece.reshape(-1) for piece in pieces], axis=0)
    return ObservationResidual(spec=spec, residual=residual, position_dim=p, quat_wxyz=q)


def stack_observation_residuals(sites: Mapping[str, Any], specs: Sequence[ObservationSpec]) -> tuple[Array, tuple[ObservationResidual, ...]]:
    """Compute and concatenate residuals for an observation list."""
    items = tuple(compute_observation_residual(sites, spec) for spec in specs)
    if not items:
        return jnp.zeros((0,), dtype=jnp.float64), items
    return jnp.concatenate([item.residual.reshape(-1) for item in items], axis=0), items
