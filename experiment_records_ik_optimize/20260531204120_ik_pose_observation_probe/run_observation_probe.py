"""Run P0/P1 observation-contract checks for IK optimization."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_observations import (
    ObservationSpec,
    quat_to_rotmat_wxyz,
    quaternion_so3_residual,
    stack_observation_residuals,
)

jax.config.update("jax_enable_x64", True)


OUT_DIR = Path(__file__).resolve().parent


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
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _build_engine(*, enable_magnetics: bool, m_body_list: list[tuple[float, float, float]]) -> ForwardKinematicsEngine:
    calib = ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(1, 1, 1),
        L1_min=1e-6,
        L_protrude_max=0.11,
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=bool(enable_magnetics),
        calib_file=str(calib) if enable_magnetics else None,
        m_body_list=m_body_list if enable_magnetics else None,
        max_iter=300,
        tol=1e-5,
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
        scale_include_external=True,
    )


def _rotation_quality(q: jnp.ndarray) -> dict[str, float]:
    R = quat_to_rotmat_wxyz(q)
    return {
        "quat_norm": float(jnp.linalg.norm(q)),
        "rot_orth_error": float(jnp.linalg.norm(R.T @ R - jnp.eye(3, dtype=jnp.float64))),
        "rot_det": float(jnp.linalg.det(R)),
    }


def _case_specs(sites: dict[str, Any]) -> list[ObservationSpec]:
    specs: list[ObservationSpec] = [
        ObservationSpec(
            site_type="tip",
            residual_type="pose",
            target_position=sites["tip_p_dim"],
            target_quat_wxyz=sites["tip_Q_wxyz"],
            sigma_position=1.0,
            sigma_angle=1.0,
            name="tip_pose_self",
        )
    ]
    rigid_q = sites["rigid_mid_Q_wxyz"]
    rigid_p = sites["rigid_mid_p_dim"]
    for idx in range(int(rigid_p.shape[0])):
        specs.append(
            ObservationSpec(
                site_type="rigid_midpoint",
                site_index=idx,
                residual_type="pose",
                target_position=rigid_p[idx],
                target_quat_wxyz=rigid_q[idx],
                sigma_position=1.0,
                sigma_angle=1.0,
                name=f"rigid_{idx}_pose_self",
            )
        )
        specs.append(
            ObservationSpec(
                site_type="rigid_midpoint",
                site_index=idx,
                residual_type="axis",
                target_quat_wxyz=rigid_q[idx],
                axis_body=(0.0, 0.0, 1.0),
                sigma_angle=1.0,
                name=f"rigid_{idx}_axis_self",
            )
        )
    return specs


def _solve_case(case: dict[str, Any]) -> dict[str, Any]:
    engine = _build_engine(enable_magnetics=bool(case["enable_magnetics"]), m_body_list=case["m_body_list"])
    currents = jnp.asarray(case["coil_currents"], dtype=jnp.float64)

    t0 = time.perf_counter()
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
    )
    wall_s = time.perf_counter() - t0
    sites = engine.query_sites(z_bar=z, scales=params.scales)
    residual, _items = stack_observation_residuals(sites, _case_specs(sites))

    engine.reset_warm_start()
    t1 = time.perf_counter()
    z_replay, params_replay, _meshes_replay, ok_replay, stats_replay = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
    )
    replay_wall_s = time.perf_counter() - t1
    replay_sites = engine.query_sites(z_bar=z_replay, scales=params_replay.scales)

    rigid_orientation_errors = []
    for q0, q1 in zip(sites["rigid_mid_Q_wxyz"], replay_sites["rigid_mid_Q_wxyz"]):
        rigid_orientation_errors.append(float(jnp.linalg.norm(quaternion_so3_residual(q1, q0))))

    tip_orientation_error = float(jnp.linalg.norm(quaternion_so3_residual(replay_sites["tip_Q_wxyz"], sites["tip_Q_wxyz"])))
    rigid_pos_delta = jnp.asarray(replay_sites["rigid_mid_p_dim"] - sites["rigid_mid_p_dim"], dtype=jnp.float64)

    qualities = [_rotation_quality(sites["tip_Q_wxyz"])]
    qualities.extend(_rotation_quality(q) for q in sites["rigid_mid_Q_wxyz"])
    max_orth = max(item["rot_orth_error"] for item in qualities)
    max_det_error = max(abs(item["rot_det"] - 1.0) for item in qualities)
    max_quat_norm_error = max(abs(item["quat_norm"] - 1.0) for item in qualities)

    return {
        "case": case["name"],
        "enable_magnetics": bool(case["enable_magnetics"]),
        "coil_currents": case["coil_currents"],
        "m_body_list": case["m_body_list"],
        "ok": bool(ok),
        "ok_replay": bool(ok_replay),
        "wall_s": float(wall_s),
        "replay_wall_s": float(replay_wall_s),
        "normE": float(stats.normE),
        "replay_normE": float(stats_replay.normE),
        "residual_self_norm": float(jnp.linalg.norm(residual)),
        "tip_p_dim": _jsonable(sites["tip_p_dim"]),
        "tip_replay_delta_m": float(jnp.linalg.norm(replay_sites["tip_p_dim"] - sites["tip_p_dim"])),
        "tip_orientation_replay_error_rad": tip_orientation_error,
        "rigid_midpoint_count": int(sites["rigid_mid_p_dim"].shape[0]),
        "rigid_replay_max_delta_m": float(jnp.max(jnp.linalg.norm(rigid_pos_delta, axis=1))),
        "rigid_orientation_replay_max_error_rad": max(rigid_orientation_errors) if rigid_orientation_errors else 0.0,
        "max_rot_orth_error": float(max_orth),
        "max_rot_det_error": float(max_det_error),
        "max_quat_norm_error": float(max_quat_norm_error),
    }


def main() -> None:
    default_m = [(0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0)]
    mixed_m = [(0.005301, 0.0, 0.0), (0.0, -0.005301, 0.0), (0.0, 0.0, 0.005301)]
    cases = [
        {
            "name": "zero_current",
            "enable_magnetics": True,
            "coil_currents": [0.0] * 8,
            "m_body_list": default_m,
        },
        {
            "name": "single_coil_current",
            "enable_magnetics": True,
            "coil_currents": [0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            "m_body_list": default_m,
        },
        {
            "name": "mixed_current_mixed_magnet",
            "enable_magnetics": True,
            "coil_currents": [4.0, -6.0, 2.5, 0.0, -3.0, 5.5, -1.5, 3.0],
            "m_body_list": mixed_m,
        },
    ]

    metadata = {
        "branch": _run(["git", "branch", "--show-current"]),
        "git_status_short": _run(["git", "status", "--short"]),
        "fk_module_file": fk_cli.__file__,
        "jax_devices": [str(device) for device in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "python": sys.executable,
    }
    results = [_solve_case(case) for case in cases]
    summary = {"metadata": metadata, "results": results}

    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case",
            "ok",
            "ok_replay",
            "wall_s",
            "replay_wall_s",
            "normE",
            "replay_normE",
            "residual_self_norm",
            "tip_replay_delta_m",
            "tip_orientation_replay_error_rad",
            "rigid_midpoint_count",
            "rigid_replay_max_delta_m",
            "rigid_orientation_replay_max_error_rad",
            "max_rot_orth_error",
            "max_rot_det_error",
            "max_quat_norm_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
