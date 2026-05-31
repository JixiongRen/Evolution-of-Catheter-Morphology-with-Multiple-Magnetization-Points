"""Run P2 task-sensitivity finite-difference audits.

This script is intentionally experiment-local. It keeps the production IK path
unchanged while exercising dense diagnostic sensitivities for observation tasks.
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

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from inverse_kinematics.ik_diff import compute_task_sensitivities_via_lm_adjoint
from inverse_kinematics.ik_observations import ObservationSpec, stack_observation_residuals

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SolveVariant:
    """Linear solve configuration for the diagnostic adjoint."""

    name: str
    ridge: float | None
    solve_mode: str
    svd_rcond: float


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


def _grade(rel_error: float) -> str:
    if rel_error < 0.2:
        return "usable"
    if rel_error < 0.5:
        return "diagnostic-only"
    return "untrusted"


def _parse_m_list(text: str) -> tuple[int, int, int]:
    vals = [int(float(item.strip())) for item in str(text).split(",") if item.strip()]
    if len(vals) != 3:
        raise ValueError("--M_list must contain three comma-separated integers")
    return tuple(vals)  # type: ignore[return-value]


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


def _specs_for_type(sites: dict[str, Any], residual_type: str) -> list[ObservationSpec]:
    if residual_type == "position":
        return [
            ObservationSpec(
                site_type="tip",
                residual_type="position",
                target_position=sites["tip_p_dim"],
                sigma_position=1.0,
                name="tip_position",
            )
        ]
    if residual_type == "axis":
        return [
            ObservationSpec(
                site_type="rigid_midpoint",
                site_index=0,
                residual_type="axis",
                target_quat_wxyz=sites["rigid_mid_Q_wxyz"][0],
                sigma_angle=1.0,
                name="rigid0_axis",
            )
        ]
    if residual_type == "so3":
        return [
            ObservationSpec(
                site_type="rigid_midpoint",
                site_index=0,
                residual_type="so3",
                target_quat_wxyz=sites["rigid_mid_Q_wxyz"][0],
                sigma_angle=1.0,
                name="rigid0_so3",
            )
        ]
    if residual_type == "position_axis":
        return _specs_for_type(sites, "position") + _specs_for_type(sites, "axis")
    raise ValueError(f"Unsupported residual_type: {residual_type}")


def _directions() -> list[tuple[str, jnp.ndarray]]:
    dirs = [
        ("dA", [0.2, -0.3, 0.1, 0.0, -0.15, 0.25, -0.1, 0.05]),
        ("dB", [-0.1, 0.25, -0.2, 0.15, 0.1, -0.05, 0.3, -0.2]),
    ]
    out = []
    for name, vals in dirs:
        vec = jnp.asarray(vals, dtype=jnp.float64)
        out.append((name, vec / jnp.maximum(jnp.linalg.norm(vec), 1e-12)))
    return out


def _solve_observe(
    engine: ForwardKinematicsEngine,
    *,
    currents: jnp.ndarray,
    z0: jnp.ndarray | None,
) -> tuple[jnp.ndarray, Any, dict[str, Any], bool, float]:
    t0 = time.perf_counter()
    z, params, _meshes, ok = engine.solve(
        coil_currents=currents,
        L_protrude=0.11,
        warm_start=True,
        override_z0_bar=z0,
    )
    wall_s = time.perf_counter() - t0
    sites = engine.query_sites(z_bar=z, scales=params.scales)
    return z, params, sites, bool(ok), float(wall_s)


def run_case(
    *,
    M_list: tuple[int, int, int],
    residual_types: list[str],
    eps_values: list[float],
    variants: list[SolveVariant],
    max_iter: int,
    tol: float,
    tol_good: float,
) -> dict[str, Any]:
    engine = _build_engine(M_list, max_iter=max_iter, tol=tol)
    I0 = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=jnp.float64)

    t0 = time.perf_counter()
    z, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=I0,
        L_protrude=0.11,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
    )
    base_solve_wall_s = time.perf_counter() - t0
    base_sites = engine.query_sites(z_bar=z, scales=params.scales)

    t_jz = time.perf_counter()
    Jz = engine._solver_cached.jacobian_jit(z, params, method=engine.jac_method)
    if hasattr(Jz, "block_until_ready"):
        Jz.block_until_ready()
    jz_wall_s = time.perf_counter() - t_jz

    fd_cache: dict[tuple[str, float], dict[str, Any]] = {}
    for direction_name, direction in _directions():
        for eps in eps_values:
            Ip = I0 + float(eps) * direction
            Im = I0 - float(eps) * direction
            _zp, _pp, sites_p, okp, plus_wall_s = _solve_observe(engine, currents=Ip, z0=z)
            _zm, _pm, sites_m, okm, minus_wall_s = _solve_observe(engine, currents=Im, z0=z)
            fd_cache[(direction_name, float(eps))] = {
                "direction": direction,
                "sites_p": sites_p,
                "sites_m": sites_m,
                "ok_plus": bool(okp),
                "ok_minus": bool(okm),
                "plus_wall_s": float(plus_wall_s),
                "minus_wall_s": float(minus_wall_s),
            }

    rows = []
    for residual_type in residual_types:
        specs = _specs_for_type(base_sites, residual_type)
        for variant in variants:
            t_sens = time.perf_counter()
            sens = compute_task_sensitivities_via_lm_adjoint(
                z_star_bar=z,
                params=params,
                lm_stats=stats,
                coil_currents=I0,
                specs=specs,
                ridge=variant.ridge,
                J_E_z=Jz,
                solve_mode=variant.solve_mode,
                svd_rcond=variant.svd_rcond,
                compute_conditioning=True,
            )
            sensitivity_wall_s = time.perf_counter() - t_sens

            for direction_name, direction in _directions():
                pred = sens.J_r_I @ direction
                for eps in eps_values:
                    fd_item = fd_cache[(direction_name, float(eps))]
                    rp, _ = stack_observation_residuals(fd_item["sites_p"], specs)
                    rm, _ = stack_observation_residuals(fd_item["sites_m"], specs)
                    fd = (rp - rm) / (2.0 * float(eps))
                    diff = float(jnp.linalg.norm(pred - fd))
                    fd_norm = max(float(jnp.linalg.norm(fd)), 1e-12)
                    rel = diff / fd_norm
                    rows.append(
                        {
                            "M_list": ",".join(str(v) for v in M_list),
                            "residual_type": residual_type,
                            "variant": variant.name,
                            "solve_mode": variant.solve_mode,
                            "ridge": None if variant.ridge is None else float(variant.ridge),
                            "svd_rcond": float(variant.svd_rcond),
                            "eps": float(eps),
                            "direction_name": direction_name,
                            "base_ok": bool(ok),
                            "base_normE": float(stats.normE),
                            "tol": float(tol),
                            "tol_good": float(tol_good),
                            "base_acceptable": bool(ok or float(stats.normE) <= float(tol_good)),
                            "ok_plus": bool(fd_item["ok_plus"]),
                            "ok_minus": bool(fd_item["ok_minus"]),
                            "stationarity_norm": float(sens.stationarity_norm),
                            "condition_number": None if sens.condition_number is None else float(sens.condition_number),
                            "sv_min": None if sens.sv_min is None else float(sens.sv_min),
                            "sv_max": None if sens.sv_max is None else float(sens.sv_max),
                            "effective_rank": sens.effective_rank,
                            "lm_lam": float(sens.lm_lam),
                            "extra_ridge": float(sens.extra_ridge),
                            "ridge_used": float(sens.ridge_used),
                            "status": str(sens.status),
                            "method": str(sens.method),
                            "base_solve_wall_s": float(base_solve_wall_s),
                            "jz_wall_s": float(jz_wall_s),
                            "sensitivity_wall_s": float(sensitivity_wall_s),
                            "plus_wall_s": float(fd_item["plus_wall_s"]),
                            "minus_wall_s": float(fd_item["minus_wall_s"]),
                            "pred_norm": float(jnp.linalg.norm(pred)),
                            "fd_norm": float(fd_norm),
                            "difference_norm": float(diff),
                            "relative_error": float(rel),
                            "grade": _grade(rel),
                            "production_candidate": False,
                        }
                    )
    return {
        "M_list": M_list,
        "base_ok": bool(ok),
        "base_normE": float(stats.normE),
        "base_solve_wall_s": float(base_solve_wall_s),
        "jz_wall_s": float(jz_wall_s),
        "rows": rows,
    }


def _write_outputs(payload: dict[str, Any]) -> None:
    rows = [row for case in payload["cases"] for row in case["rows"]]
    payload["row_count"] = len(rows)
    payload["grade_counts"] = {
        grade: sum(1 for row in rows if row["grade"] == grade)
        for grade in ("usable", "diagnostic-only", "untrusted")
    }
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
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--tol_good", type=float, default=1e-3)
    args = parser.parse_args()

    variants = [
        SolveVariant(name="solve_base", ridge=None, solve_mode="solve", svd_rcond=1e-10),
        SolveVariant(name="solve_ridge_1e-4", ridge=1e-4, solve_mode="solve", svd_rcond=1e-10),
        SolveVariant(name="svd_rcond_1e-8", ridge=None, solve_mode="svd", svd_rcond=1e-8),
    ]
    eps_values = [0.10, 0.25]
    residual_types = ["position", "axis", "so3", "position_axis"]

    case_specs = [
        {"M_list": (1, 1, 1), "residual_types": residual_types},
    ]
    if args.mode == "full":
        case_specs.append({"M_list": (3, 3, 3), "residual_types": residual_types})
    else:
        case_specs.append({"M_list": (3, 3, 3), "residual_types": ["position", "axis"]})

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
            "tol_good": float(args.tol_good),
        },
        "cases": [],
    }
    for spec in case_specs:
        payload["cases"].append(
            run_case(
                M_list=spec["M_list"],
                residual_types=spec["residual_types"],
                eps_values=eps_values,
                variants=variants,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                tol_good=float(args.tol_good),
            )
        )
        _write_outputs(payload)
    _write_outputs(payload)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
