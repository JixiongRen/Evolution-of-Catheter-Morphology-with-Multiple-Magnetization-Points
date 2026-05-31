"""Run a diagnostic dense task-sensitivity finite-difference audit."""

from __future__ import annotations

import json
import os
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

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_diff import compute_task_sensitivities_via_lm_adjoint
from inverse_kinematics.ik_observations import ObservationSpec, stack_observation_residuals

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _build_engine() -> ForwardKinematicsEngine:
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
        enable_magnetics=True,
        calib_file=str(calib),
        m_body_list=[(0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0)],
        max_iter=300,
        tol=1e-5,
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense_fused_cholesky",
        scale_include_external=True,
    )


def main() -> None:
    engine = _build_engine()
    I0 = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=jnp.float64)
    direction = jnp.asarray([0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.1, 0.05], dtype=jnp.float64)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-12)
    eps = 0.25

    t0 = time.perf_counter()
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=I0,
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
    )
    base_wall_s = time.perf_counter() - t0
    sites = engine.query_sites(z_bar=z, scales=params.scales)
    specs = [
        ObservationSpec(
            site_type="tip",
            residual_type="position",
            target_position=sites["tip_p_dim"],
            sigma_position=1.0,
            name="tip_position",
        ),
        ObservationSpec(
            site_type="rigid_midpoint",
            site_index=0,
            residual_type="axis",
            target_quat_wxyz=sites["rigid_mid_Q_wxyz"][0],
            sigma_angle=1.0,
            name="rigid0_axis",
        ),
    ]

    t1 = time.perf_counter()
    sens = compute_task_sensitivities_via_lm_adjoint(
        z_star_bar=z,
        params=params,
        lm_stats=stats,
        coil_currents=I0,
        specs=specs,
        ridge=None,
        compute_conditioning=True,
    )
    sensitivity_wall_s = time.perf_counter() - t1
    pred = sens.J_r_I @ direction

    Ip = I0 + eps * direction
    Im = I0 - eps * direction

    t2 = time.perf_counter()
    zp, pp, _mp, okp = engine.solve(coil_currents=Ip, L_protrude=0.11, warm_start=True, override_z0_bar=z)
    sites_p = engine.query_sites(z_bar=zp, scales=pp.scales)
    rp, _ = stack_observation_residuals(sites_p, specs)
    plus_wall_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    zm, pm, _mm, okm = engine.solve(coil_currents=Im, L_protrude=0.11, warm_start=True, override_z0_bar=z)
    sites_m = engine.query_sites(z_bar=zm, scales=pm.scales)
    rm, _ = stack_observation_residuals(sites_m, specs)
    minus_wall_s = time.perf_counter() - t3

    fd = (rp - rm) / (2.0 * eps)
    diff = float(jnp.linalg.norm(pred - fd))
    denom = max(float(jnp.linalg.norm(fd)), 1e-12)
    rel = diff / denom
    if rel < 0.2:
        grade = "usable"
    elif rel < 0.5:
        grade = "diagnostic-only"
    else:
        grade = "untrusted"

    result = {
        "label": "P2 diagnostic-only dense task sensitivity",
        "ok_base": bool(ok),
        "ok_plus": bool(okp),
        "ok_minus": bool(okm),
        "base_normE": float(stats.normE),
        "stationarity_norm": float(sens.stationarity_norm),
        "condition_number": sens.condition_number,
        "status": str(sens.status),
        "method": str(sens.method),
        "base_wall_s": float(base_wall_s),
        "sensitivity_wall_s": float(sensitivity_wall_s),
        "plus_wall_s": float(plus_wall_s),
        "minus_wall_s": float(minus_wall_s),
        "eps_A": float(eps),
        "direction": direction,
        "predicted_directional_derivative": pred,
        "fd_directional_derivative": fd,
        "difference_norm": float(diff),
        "fd_norm": float(denom),
        "relative_error": float(rel),
        "grade": grade,
        "production_candidate": False,
    }
    (OUT_DIR / "p2_task_sensitivity_probe.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
