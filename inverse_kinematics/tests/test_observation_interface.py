"""Tests for multi-site IK observation residuals."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_diff import extract_terminal_p_dim, extract_tip_p_dim
from inverse_kinematics.ik_observations import (
    ObservationSpec,
    axis_residual,
    quat_to_rotmat_wxyz,
    quaternion_so3_residual,
    rotmat_to_so3_log,
    stack_observation_residuals,
)

jax.config.update("jax_enable_x64", True)


def _quat_x(theta: float) -> jnp.ndarray:
    return jnp.asarray([math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0], dtype=jnp.float64)


def _build_small_engine() -> ForwardKinematicsEngine:
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(1, 1, 1),
        L1_min=1e-6,
        L_protrude_max=0.11,
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(1.0, 0.0, 0.0, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=False,
        enable_magnetics=False,
        max_iter=20,
        tol=1e-8,
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
    )


def test_so3_log_small_angle() -> None:
    theta = 1e-4
    residual = quaternion_so3_residual(_quat_x(theta), jnp.asarray([1.0, 0.0, 0.0, 0.0]))
    assert jnp.linalg.norm(residual - jnp.asarray([theta, 0.0, 0.0])) < 1e-8


def test_axis_residual_zero_and_small_angle_direction() -> None:
    q_identity = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    assert float(jnp.linalg.norm(axis_residual(q_identity, target_axis=(0.0, 0.0, 1.0)))) < 1e-12

    theta = 1e-4
    residual = axis_residual(_quat_x(theta), target_axis=(0.0, 0.0, 1.0))
    assert jnp.linalg.norm(residual - jnp.asarray([theta, 0.0, 0.0])) < 1e-8


def test_rotation_matrix_properties() -> None:
    R = quat_to_rotmat_wxyz(jnp.asarray([0.5, 0.5, 0.5, 0.5], dtype=jnp.float64))
    I = jnp.eye(3, dtype=jnp.float64)
    assert float(jnp.linalg.norm(R.T @ R - I)) < 1e-12
    assert abs(float(jnp.linalg.det(R)) - 1.0) < 1e-12
    assert float(jnp.linalg.norm(rotmat_to_so3_log(I))) < 1e-12


def test_query_sites_contract_matches_tip_extractor() -> None:
    engine = _build_small_engine()
    z_star, params, _meshes, ok, _stats = engine.solve_with_stats(
        coil_currents=jnp.zeros((8,), dtype=jnp.float64),
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
    )
    assert ok

    sites = engine.query_sites(z_bar=z_star, scales=params.scales)
    tip_old = extract_tip_p_dim(z_star, M_list=params.M_list, scales=params.scales)
    terminal = extract_terminal_p_dim(z_star, M_list=params.M_list, scales=params.scales)
    assert jnp.linalg.norm(sites["tip_p_dim"] - tip_old) < 1e-12
    assert jnp.linalg.norm(sites["terminal_p_dim"] - terminal) < 1e-12
    assert jnp.linalg.norm(sites["terminal_p_dim"] - sites["rigid_distal_p_dim"][-1]) < 1e-12
    assert sites["rigid_mid_p_dim"].shape == (3, 3)
    assert sites["rigid_mid_Q_wxyz"].shape == (3, 4)
    assert sites["rigid_distal_p_dim"].shape == (3, 3)
    assert sites["rigid_distal_Q_wxyz"].shape == (3, 4)

    specs = [
        ObservationSpec(
            site_type="tip",
            residual_type="position",
            target_position=sites["tip_p_dim"],
            sigma_position=1.0,
        ),
        ObservationSpec(
            site_type="rigid_midpoint",
            site_index=0,
            residual_type="axis",
            target_axis=(0.0, 0.0, 1.0),
            sigma_angle=1.0,
        ),
        ObservationSpec(
            site_type="rigid_midpoint",
            site_index=1,
            residual_type="so3",
            target_quat_wxyz=sites["rigid_mid_Q_wxyz"][1],
            sigma_angle=1.0,
        ),
    ]
    residual, items = stack_observation_residuals(sites, specs)
    assert len(items) == 3
    assert residual.shape == (9,)
    assert float(jnp.linalg.norm(residual)) < 1e-10


if __name__ == "__main__":
    test_so3_log_small_angle()
    test_axis_residual_zero_and_small_angle_direction()
    test_rotation_matrix_properties()
    test_query_sites_contract_matches_tip_extractor()
    print("[observation-interface] PASS")
