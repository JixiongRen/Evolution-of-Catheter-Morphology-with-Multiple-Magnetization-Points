from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from single_layer_ik.forward_kinematics_optimized_bak.equilibrium_solver_nondim import residual_bar
from single_layer_ik.forward_kinematics_optimized_bak.fk_engine import ForwardKinematicsEngine
from single_layer_ik.forward_kinematics_optimized_bak.nondim import x_bar_to_dim
from single_layer_ik.forward_kinematics_optimized_bak.utils_nondim import unpack_z_bar_jax
from single_layer_ik.joint_problem import (
    extract_centerline_p_dim,
    extract_interface_position_gaps_dim,
    extract_shape_smooth_residual,
)


Array = jnp.ndarray


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    M_list: tuple[int, int, int]
    L_protrude: float
    enable_gravity: bool
    enable_magnetics: bool
    coil_currents: tuple[float, ...]
    m_body_list: Optional[tuple[tuple[float, float, float], ...]] = None
    diagnostic_only: bool = False


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _time_blocked(fn) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    _block(out)
    return out, time.perf_counter() - t0


def _to_float_list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(jax.device_get(value)).reshape(-1).tolist()]


def _build_engine(case: CaseConfig, *, linear_solver: str, max_iter: int, tol: float, args: argparse.Namespace):
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=case.M_list,
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
        calib_file=args.calib_file,
        actuation_table_pkl=args.actuation_table_pkl,
        m_body_list=case.m_body_list,
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=float(args.lm_damping),
        jac_method=str(args.jac_method),
        linear_solver=str(linear_solver),
        cg_tol=float(args.cg_tol),
        cg_atol=float(args.cg_atol),
        cg_max_iter=None if args.cg_max_iter <= 0 else int(args.cg_max_iter),
        cg_damping_preconditioner=not bool(args.no_cg_damping_preconditioner),
        L_protrude_max=float(args.L_protrude_max),
    )


def _candidate_points(z_bar: Array, params: Any) -> dict[str, Any]:
    x_nodes_list, _k_list, x_rigid_list = unpack_z_bar_jax(z_bar, M_list=params.M_list)
    flex_last = x_bar_to_dim(x_nodes_list[-1][-1], params.scales)
    rigid_last = x_bar_to_dim(x_rigid_list[-1], params.scales)
    return {
        "last_flex_node_m": _to_float_list(flex_last[0:3]),
        "last_rigid_state_m": _to_float_list(rigid_last[0:3]),
        "last_flex_quat_wxyz": _to_float_list(flex_last[3:7]),
        "last_rigid_quat_wxyz": _to_float_list(rigid_last[3:7]),
    }


def _geometry_metrics(z_bar: Array, params: Any) -> dict[str, Any]:
    centerline = extract_centerline_p_dim(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=2,
    )
    gap = extract_interface_position_gaps_dim(z_bar=z_bar, scales=params.scales, M_list=params.M_list)
    smooth = extract_shape_smooth_residual(
        z_bar=z_bar,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=2,
    )
    cl_np = np.asarray(jax.device_get(centerline), dtype=np.float64)
    if cl_np.shape[0] >= 3:
        edges = np.diff(cl_np, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        unit = edges / np.maximum(lengths[:, None], 1e-12)
        turn = np.linalg.norm(np.diff(unit, axis=0), axis=1)
    else:
        turn = np.zeros((0,), dtype=np.float64)
    return {
        "centerline": cl_np,
        "centerline_end_m": cl_np[-1].tolist() if cl_np.size else None,
        "interface_gap_max_m": float(np.max(np.linalg.norm(np.asarray(jax.device_get(gap)), axis=1))) if gap.size else 0.0,
        "smooth_norm": float(np.linalg.norm(np.asarray(jax.device_get(smooth)))) if smooth.size else 0.0,
        "curvature_proxy": turn.tolist(),
    }


def _compare_geometry(ref: dict[str, Any], cand: dict[str, Any], L_ref: float) -> dict[str, Any]:
    ref_cl = np.asarray(ref["centerline"], dtype=np.float64)
    cand_cl = np.asarray(cand["centerline"], dtype=np.float64)
    n = min(ref_cl.shape[0], cand_cl.shape[0])
    if n == 0:
        return {"centerline_rms_m": float("nan"), "centerline_max_m": float("nan"), "curvature_corr": float("nan"), "verdict": "missing_centerline"}
    delta = cand_cl[:n] - ref_cl[:n]
    rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    max_delta = float(np.max(np.linalg.norm(delta, axis=1)))
    a = np.asarray(ref["curvature_proxy"], dtype=np.float64)
    b = np.asarray(cand["curvature_proxy"], dtype=np.float64)
    k = min(a.size, b.size)
    if k >= 2 and np.std(a[:k]) > 1e-12 and np.std(b[:k]) > 1e-12:
        corr = float(np.corrcoef(a[:k], b[:k])[0, 1])
    else:
        corr = 1.0
    hard_fail = rms > 0.10 * float(L_ref) or max_delta > 0.15 * float(L_ref) or corr < 0.8
    return {
        "centerline_rms_m": rms,
        "centerline_max_m": max_delta,
        "curvature_corr": corr,
        "verdict": "hard_fail" if hard_fail else "ok_or_warning",
    }


def _profile_dense_step(engine: ForwardKinematicsEngine, z_bar: Array, params: Any, lam: float) -> dict[str, Any]:
    solver = engine._solver_cached
    E, residual_s = _time_blocked(lambda: solver.residual_jit(z_bar, params))
    J, jacobian_s = _time_blocked(lambda: solver.jacobian_jit(z_bar, params, method=engine.jac_method))
    JtJ, jtj_s = _time_blocked(lambda: jnp.transpose(J) @ J)
    g, g_s = _time_blocked(lambda: jnp.transpose(J) @ E)
    A = JtJ + float(lam) * jnp.eye(J.shape[1], dtype=J.dtype)
    _dz, solve_s = _time_blocked(lambda: jnp.linalg.solve(A, -g))
    return {
        "residual_s": float(residual_s),
        "jacobian_s": float(jacobian_s),
        "jtj_s": float(jtj_s),
        "jt_residual_s": float(g_s),
        "linear_solve_s": float(solve_s),
        "J_shape": [int(J.shape[0]), int(J.shape[1])],
        "J_bytes": int(J.size * J.dtype.itemsize),
    }


def _run_case(case: CaseConfig, *, linear_solver: str, max_iter: int, tol: float, args: argparse.Namespace) -> dict[str, Any]:
    engine = _build_engine(case, linear_solver=linear_solver, max_iter=max_iter, tol=tol, args=args)
    repeat_count = 1 + max(int(args.steady_repeats), 0)
    wall_s_list: list[float] = []
    z_star = None
    params = None
    ok = False
    stats = None
    for _repeat_idx in range(repeat_count):
        start = time.perf_counter()
        z_star, params, _meshes, ok, stats = engine.solve_with_stats(
            coil_currents=case.coil_currents,
            L_protrude=float(case.L_protrude),
            warm_start=False,
            return_stats=True,
            stats_include_jacobian=False,
            verbose=False,
        )
        _block(z_star)
        wall_s_list.append(time.perf_counter() - start)
    assert z_star is not None
    assert params is not None
    wall_s = wall_s_list[0]
    E = residual_bar(z_star, params)
    _block(E)
    geom = _geometry_metrics(z_star, params)
    points = _candidate_points(z_star, params)
    return {
        "case_id": case.case_id,
        "linear_solver": str(linear_solver),
        "ok": bool(ok),
        "diagnostic_only": bool(case.diagnostic_only),
        "wall_s": float(wall_s),
        "first_call_wall_s": float(wall_s_list[0]),
        "steady_wall_s": None if len(wall_s_list) <= 1 else float(wall_s_list[-1]),
        "repeat_wall_s": [float(v) for v in wall_s_list],
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "normE": float(jnp.linalg.norm(E)),
        "cost": 0.5 * float(jnp.linalg.norm(E)) ** 2,
        "linear_info": {} if stats is None or stats.linear_info is None else stats.linear_info,
        "M_list": list(case.M_list),
        "L_protrude": float(case.L_protrude),
        "L_ref": float(params.scales.L_ref),
        "L1_dim": float(params.L1_dim),
        "enable_gravity": bool(case.enable_gravity),
        "enable_magnetics": bool(case.enable_magnetics),
        "coil_currents": list(case.coil_currents),
        "m_body_list": None if case.m_body_list is None else [list(v) for v in case.m_body_list],
        "points": points,
        "geometry": {k: v for k, v in geom.items() if k != "centerline"},
        "_centerline": geom["centerline"],
        "_params": params,
        "_z_star": z_star,
        "_engine": engine,
    }


def _case_set() -> list[CaseConfig]:
    mixed_m = ((0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0))
    return [
        CaseConfig("straight_nomag_nogravity", (3, 2, 2), 0.10, False, False, (0.0,) * 8),
        CaseConfig("gravity_only_mid", (3, 2, 2), 0.10, True, False, (0.0,) * 8),
        CaseConfig("mag_zero_mid", (3, 2, 2), 0.10, False, True, (0.0,) * 8, mixed_m),
        CaseConfig("coil5_pos", (3, 2, 2), 0.10, False, True, (0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0), mixed_m),
        CaseConfig("mixed_balanced", (3, 2, 2), 0.10, False, True, (10.0, -8.0, 6.0, -4.0, 2.0, -6.0, 8.0, -10.0), mixed_m),
        CaseConfig("mesh5_mixed_diagnostic", (5, 5, 5), 0.10, False, True, (10.0, -8.0, 6.0, -4.0, 2.0, -6.0, 8.0, -10.0), mixed_m, True),
    ]


def _minimal_case_set() -> list[CaseConfig]:
    return [
        CaseConfig("minimal_nomag_nogravity", (1, 1, 1), 0.10, False, False, (0.0,) * 8),
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(public, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "dense_wall_s",
        "dense_first_call_wall_s",
        "dense_steady_wall_s",
        "matrix_free_wall_s",
        "matrix_free_first_call_wall_s",
        "matrix_free_steady_wall_s",
        "speedup",
        "dense_normE",
        "matrix_free_normE",
        "centerline_rms_m",
        "centerline_max_m",
        "curvature_corr",
        "geometry_verdict",
        "dense_iter",
        "matrix_free_iter",
        "matrix_free_stop",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run FK correctness, profiling, and matrix-free comparison cases.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_iter", type=int, default=60)
    parser.add_argument("--matrix_free_max_iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lm_damping", type=float, default=1e-1)
    parser.add_argument("--jac_method", choices=["fwd", "rev"], default="fwd")
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--cg_atol", type=float, default=0.0)
    parser.add_argument("--cg_max_iter", type=int, default=60)
    parser.add_argument("--no_cg_damping_preconditioner", action="store_true")
    parser.add_argument("--calib_file", default="calib/mpem_calibration_file_sp=40_order=1.yaml")
    parser.add_argument("--actuation_table_pkl", default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl")
    parser.add_argument("--L_protrude_max", type=float, default=0.18)
    parser.add_argument("--limit_cases", type=int, default=0)
    parser.add_argument("--case_ids", default="", help="Comma-separated case ids to run; empty runs the selected case set")
    parser.add_argument("--minimal_cases", action="store_true", help="Run a tiny shape-only smoke case instead of representative cases")
    parser.add_argument("--steady_repeats", type=int, default=0, help="Repeat the same solve on the same engine to separate compile and steady-state time")
    parser.add_argument("--skip_matrix_free", action="store_true")
    parser.add_argument("--skip_dense_profile", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _minimal_case_set() if bool(args.minimal_cases) else _case_set()
    if str(args.case_ids).strip():
        requested = {v.strip() for v in str(args.case_ids).split(",") if v.strip()}
        cases = [case for case in cases if case.case_id in requested]
    if int(args.limit_cases) > 0:
        cases = cases[: int(args.limit_cases)]

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=False, text=True, capture_output=True).stdout.strip(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "max_iter": int(args.max_iter),
        "matrix_free_max_iter": int(args.matrix_free_max_iter),
        "cg_tol": float(args.cg_tol),
        "cg_max_iter": int(args.cg_max_iter),
        "minimal_cases": bool(args.minimal_cases),
        "steady_repeats": int(args.steady_repeats),
        "skip_matrix_free": bool(args.skip_matrix_free),
        "skip_dense_profile": bool(args.skip_dense_profile),
        "case_ids": [case.case_id for case in cases],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    dense_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    def flush_outputs() -> None:
        _write_jsonl(out_dir / "fk_dense_results.jsonl", dense_rows)
        _write_jsonl(out_dir / "fk_matrix_free_results.jsonl", matrix_rows)
        _write_jsonl(out_dir / "dense_step_profile.jsonl", profile_rows)
        _write_csv(out_dir / "fk_dense_vs_matrix_free_summary.csv", compare_rows)
        (out_dir / "summary.json").write_text(json.dumps(compare_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for case in cases:
        try:
            dense = _run_case(case, linear_solver="dense", max_iter=int(args.max_iter), tol=float(args.tol), args=args)
            dense_rows.append(dense)
            flush_outputs()
            if not bool(args.skip_dense_profile) and not case.diagnostic_only:
                prof = _profile_dense_step(dense["_engine"], dense["_z_star"], dense["_params"], float(args.lm_damping))
                prof.update({"case_id": case.case_id})
                profile_rows.append(prof)
                flush_outputs()

            compare_row = {
                "case_id": case.case_id,
                "dense_wall_s": dense["wall_s"],
                "dense_first_call_wall_s": dense["first_call_wall_s"],
                "dense_steady_wall_s": "" if dense["steady_wall_s"] is None else dense["steady_wall_s"],
                "dense_normE": dense["normE"],
                "dense_iter": dense["n_iter"],
            }
            if bool(args.skip_matrix_free) or int(args.matrix_free_max_iter) <= 0:
                compare_row["geometry_verdict"] = "dense_only"
                compare_rows.append(compare_row)
                flush_outputs()
                continue

            mf = _run_case(
                case,
                linear_solver="matrix_free_cg",
                max_iter=int(args.matrix_free_max_iter),
                tol=float(args.tol),
                args=args,
            )
            matrix_rows.append(mf)
            compare = _compare_geometry(
                {"centerline": dense["_centerline"], "curvature_proxy": dense["geometry"]["curvature_proxy"]},
                {"centerline": mf["_centerline"], "curvature_proxy": mf["geometry"]["curvature_proxy"]},
                float(dense["L_ref"]),
            )
            compare_row.update(
                {
                    "matrix_free_wall_s": mf["wall_s"],
                    "matrix_free_first_call_wall_s": mf["first_call_wall_s"],
                    "matrix_free_steady_wall_s": "" if mf["steady_wall_s"] is None else mf["steady_wall_s"],
                    "speedup": dense["wall_s"] / mf["wall_s"] if mf["wall_s"] > 0 else float("nan"),
                    "matrix_free_normE": mf["normE"],
                    "centerline_rms_m": compare["centerline_rms_m"],
                    "centerline_max_m": compare["centerline_max_m"],
                    "curvature_corr": compare["curvature_corr"],
                    "geometry_verdict": compare["verdict"],
                    "matrix_free_iter": mf["n_iter"],
                    "matrix_free_stop": mf["stop_reason"],
                }
            )
            compare_rows.append(compare_row)
            flush_outputs()
        except Exception as exc:
            compare_rows.append({"case_id": case.case_id, "geometry_verdict": f"error: {exc!r}"})
            flush_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
