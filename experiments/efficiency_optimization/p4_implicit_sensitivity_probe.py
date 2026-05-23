from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "inverse_kinematics"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import jax
import jax.numpy as jnp
import numpy as np

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from ik_diff import compute_tip_sensitivities_via_lm_adjoint


Array = jnp.ndarray


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    enable_gravity: bool
    enable_magnetics: bool
    coil_currents: tuple[float, ...]
    m_body_list: Optional[tuple[tuple[float, float, float], ...]] = None


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _to_list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(jax.device_get(value)).reshape(-1).tolist()]


def _case_set() -> list[CaseConfig]:
    mixed_m = ((0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0))
    return [
        CaseConfig("straight_nomag_nogravity", False, False, (0.0,) * 8),
        CaseConfig("mag_zero_mid", False, True, (0.0,) * 8, mixed_m),
        CaseConfig("mixed_balanced", False, True, (10.0, -8.0, 6.0, -4.0, 2.0, -6.0, 8.0, -10.0), mixed_m),
    ]


def _build_engine(case: CaseConfig, args: argparse.Namespace) -> ForwardKinematicsEngine:
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(3, 2, 2),
        L1_min=1e-6,
        flex_d_outer=(0.0015,),
        flex_E=(1.8e6,),
        flex_G=(0.6e6,),
        flex_rho=(970.0,),
        rigid_d_outer=(0.0015,),
        rigid_rho=(7500.0,),
        p0_dim=(0.0, 0.0, -0.05),
        Q0_wxyz=(1.0, 0.0, 0.0, 0.0),
        axis_body=(1.0, 0.0, 0.0),
        enable_gravity=bool(case.enable_gravity),
        enable_magnetics=bool(case.enable_magnetics),
        calib_file=str(args.calib_file),
        actuation_table_pkl=str(args.actuation_table_pkl),
        m_body_list=case.m_body_list,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        lm_damping=float(args.lm_damping),
        jac_method=str(args.jac_method),
        L_protrude_max=float(args.L_protrude_max),
    )


def _solve_tip(
    engine: ForwardKinematicsEngine,
    *,
    I: Array,
    x: float,
    z0_bar: Optional[Array],
) -> tuple[Array, Array, Any, Any]:
    z, params, _meshes, _ok, stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(x),
        warm_start=True,
        override_z0_bar=z0_bar,
        return_stats=True,
        stats_include_jacobian=False,
    )
    _block(z)
    obs = engine.query_sites(z_bar=z, scales=params.scales)
    p = jnp.asarray(obs["tip_p_dim"], dtype=jnp.float64)
    _block(p)
    return p, z, params, stats


def _direction(seed: int, *, n_coils: int, current_step_a: float, x_step_m: float) -> tuple[Array, float]:
    rng = np.random.default_rng(int(seed))
    dI = rng.normal(size=(n_coils,))
    dI = dI / max(float(np.linalg.norm(dI)), 1e-12) * float(current_step_a)
    dx = float(rng.normal()) * float(x_step_m)
    return jnp.asarray(dI, dtype=jnp.float64), dx


def _run_case(case: CaseConfig, args: argparse.Namespace) -> dict[str, Any]:
    engine = _build_engine(case, args)
    I0 = jnp.asarray(case.coil_currents, dtype=jnp.float64)
    x0 = float(args.L_protrude)
    p0, z0, params0, stats0 = _solve_tip(engine, I=I0, x=x0, z0_bar=None)

    t0 = time.perf_counter()
    Jz0 = engine._solver_cached.jacobian_jit(z0, params0, method=engine.jac_method)
    _block(Jz0)
    jz_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sens = compute_tip_sensitivities_via_lm_adjoint(
        z_star_bar=z0,
        params=params0,
        lm_stats=stats0,
        coil_currents=I0,
        ridge=float(args.ridge),
        J_E_z=Jz0,
        materialize_jacobian=True,
        compute_conditioning=bool(args.conditioning),
        weak_normE=float(args.weak_normE),
    )
    _block(sens.J_p_I)
    _block(sens.J_p_x)
    sensitivity_s = time.perf_counter() - t0

    direction_rows = []
    for idx in range(int(args.num_directions)):
        dI, dx = _direction(
            int(args.seed) + idx,
            n_coils=int(I0.shape[0]),
            current_step_a=float(args.current_step_a),
            x_step_m=float(args.x_step_m),
        )
        xp = min(float(args.x_max), x0 + float(dx))
        xm = max(float(args.x_min), x0 - float(dx))
        if abs(xp - xm) < 1e-12:
            xp = x0
            xm = x0
            dx_eff = 0.0
        else:
            dx_eff = 0.5 * (xp - xm)

        Ip = I0 + dI
        Im = I0 - dI
        pp, _zp, _paramsp, statsp = _solve_tip(engine, I=Ip, x=xp, z0_bar=z0)
        pm, _zm, _paramsm, statsm = _solve_tip(engine, I=Im, x=xm, z0_bar=z0)
        fd_delta = 0.5 * (pp - pm)
        adj_delta = sens.J_p_I @ dI + sens.J_p_x * float(dx_eff)
        err = adj_delta - fd_delta
        abs_err = float(jnp.linalg.norm(err))
        fd_norm = float(jnp.linalg.norm(fd_delta))
        rel_err = abs_err / max(fd_norm, float(args.abs_floor))
        direction_rows.append(
            {
                "direction_idx": int(idx),
                "dI_A": _to_list(dI),
                "dx_m": float(dx_eff),
                "adj_delta_m": _to_list(adj_delta),
                "fd_delta_m": _to_list(fd_delta),
                "abs_err_m": abs_err,
                "rel_err": float(rel_err),
                "fd_norm_m": fd_norm,
                "fd_plus_normE": float(statsp.normE),
                "fd_minus_normE": float(statsm.normE),
                "verdict": "pass" if rel_err <= float(args.rel_tol) or abs_err <= float(args.abs_tol_m) else "warn",
            }
        )

    return {
        "case_id": case.case_id,
        "jax_backend": jax.default_backend(),
        "normE": float(stats0.normE),
        "p_tip_m": _to_list(p0),
        "jz_s": float(jz_s),
        "sensitivity_s": float(sensitivity_s),
        "sensitivity_status": str(sens.status),
        "sensitivity_method": str(sens.method),
        "stationarity_norm": float(sens.stationarity_norm),
        "condition_number": None if sens.condition_number is None else float(sens.condition_number),
        "J_p_x": _to_list(sens.J_p_x),
        "directions": direction_rows,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate implicit adjoint sensitivity with directional FK finite differences.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lm_damping", type=float, default=1e-1)
    parser.add_argument("--jac_method", choices=["fwd", "rev"], default="fwd")
    parser.add_argument("--L_protrude", type=float, default=0.10)
    parser.add_argument("--L_protrude_max", type=float, default=0.18)
    parser.add_argument("--x_min", type=float, default=0.095)
    parser.add_argument("--x_max", type=float, default=0.105)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--weak_normE", type=float, default=1e-1)
    parser.add_argument("--num_directions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--current_step_a", type=float, default=0.2)
    parser.add_argument("--x_step_m", type=float, default=2e-4)
    parser.add_argument("--rel_tol", type=float, default=0.5)
    parser.add_argument("--abs_tol_m", type=float, default=2e-4)
    parser.add_argument("--abs_floor", type=float, default=1e-7)
    parser.add_argument("--calib_file", default="calib/mpem_calibration_file_sp=40_order=1.yaml")
    parser.add_argument("--actuation_table_pkl", default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl")
    parser.add_argument("--case_ids", default="")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _case_set()
    if str(args.case_ids).strip():
        wanted = {v.strip() for v in str(args.case_ids).split(",") if v.strip()}
        cases = [case for case in cases if case.case_id in wanted]

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=False, text=True, capture_output=True).stdout.strip(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for case in cases:
        row = _run_case(case, args)
        rows.append(row)
        (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
