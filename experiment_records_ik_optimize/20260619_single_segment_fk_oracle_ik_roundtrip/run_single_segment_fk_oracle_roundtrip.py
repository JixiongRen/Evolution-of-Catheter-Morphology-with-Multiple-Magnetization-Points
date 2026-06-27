from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parents[1]
FK_SOURCE_DIR = REPO / "experiment_records_fk_optimize" / "20260615_fk_single_segment_L50mm_coils1_4_15A_tol1e6"
FK_VIS_SUMMARY = FK_SOURCE_DIR / "visualization" / "visualization_summary.json"
IK_SOURCE_TARGETS = OUT_DIR / "ik_source_fk_oracle_targets.csv"
CALIB_FILE = REPO / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
ACTUATION_TABLE = REPO / "supiee_auto_diff" / "offline_interpolation_data" / "actuation_tables" / "actuation_table.pkl"
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

FLEX_LENGTH = 0.047
RIGID_LENGTH = 0.003
X_REF = FLEX_LENGTH + RIGID_LENGTH
M_LIST = "12"
M_BODY_LIST = "0,0,0.005301"
P0 = "0,0,0"
Q0 = "0.70710678,0,0.70710678,0"
AXIS_BODY = "0,0,1"
G_WORLD = "0,0,-9.81"

PASS_LIMITS = {
    "fk_normE": 1e-6,
    "ik_target_err_mm": 1.0,
    "fk_target_err_mm": 1.0,
    "ik_fk_tip_mm": 0.5,
    "ik_fk_centerline_rms_mm": 1.0,
    "oracle_fk_centerline_rms_mm": 1.0,
    "x_abs_err_mm": 0.05,
}


@dataclass(frozen=True)
class OracleCase:
    name: str
    active_coil: int
    currents: np.ndarray
    p_des: np.ndarray
    fk_normE: float
    iteration_count: int
    source_dir: Path


def _ik_fk_engine(*, max_iter: int, tol: float, lm_damping: float):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from single_layer_ik.forward_kinematics_optimized_bak.fk_engine import ForwardKinematicsEngine

    return ForwardKinematicsEngine(
        flex_lengths_tail=[FLEX_LENGTH],
        rigid_lengths=[RIGID_LENGTH],
        M_list=[int(M_LIST)],
        L1_min=1e-6,
        p0_dim=_parse_csv_floats(P0),
        Q0_wxyz=_parse_csv_floats(Q0),
        axis_body=_parse_csv_floats(AXIS_BODY),
        enable_gravity=True,
        g_world=_parse_csv_floats(G_WORLD),
        enable_magnetics=True,
        calib_file=str(CALIB_FILE),
        actuation_table_pkl=str(ACTUATION_TABLE),
        m_body_list=[_parse_csv_floats(M_BODY_LIST)],
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=float(lm_damping),
        jac_method="fwd",
        L_protrude_max=X_REF,
    )


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).replace(";", ",").split(",") if str(x).strip()]


def _fmt_vec(vals: np.ndarray | list[float], ndigits: int = 12) -> str:
    arr = np.asarray(vals, dtype=float).reshape(-1)
    return ",".join(f"{float(v):.{ndigits}g}" for v in arr)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    return env


def _run_logged(cmd: list[str], log_path: Path, *, cwd: Path = REPO) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=_env(), stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; see {log_path}")


def _load_same_source_targets() -> dict[str, np.ndarray]:
    if not IK_SOURCE_TARGETS.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    for row in _read_csv(IK_SOURCE_TARGETS):
        out[row["case"]] = np.asarray(_parse_csv_floats(row["p_des_m"]), dtype=float)
    return out


def _load_oracle_cases(*, prefer_same_source: bool = True) -> list[OracleCase]:
    same_source_targets = _load_same_source_targets() if prefer_same_source else {}
    summary = json.loads(FK_VIS_SUMMARY.read_text(encoding="utf-8"))
    cases: list[OracleCase] = []
    for rec in summary:
        case_dir = FK_SOURCE_DIR / str(rec["case"])
        cfg = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
        currents = np.asarray(_parse_csv_floats(cfg["coil_currents"]), dtype=float)
        cases.append(
            OracleCase(
                name=str(rec["case"]),
                active_coil=int(rec["active_coil"]),
                currents=currents,
                p_des=same_source_targets.get(str(rec["case"]), np.asarray(rec["final_tip_m"], dtype=float)),
                fk_normE=float(rec["final_normE"]),
                iteration_count=int(rec["iteration_count"]),
                source_dir=case_dir,
            )
        )
    return cases


def _import_ik_helpers():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import jax
    import jax.numpy as jnp
    from single_layer_ik.joint_problem import extract_centerline_p_dim

    return jax, jnp, extract_centerline_p_dim


def _centerline_from_ik_params(z: np.ndarray, params: Any, *, rigid_samples: int = 10) -> np.ndarray:
    jax, jnp, extract_centerline_p_dim = _import_ik_helpers()
    centerline = extract_centerline_p_dim(
        z_bar=jnp.asarray(z, dtype=jnp.float64),
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=int(rigid_samples),
    )
    return np.asarray(jax.device_get(centerline), dtype=float)


def _centerline_from_saved_same_source_z(
    *,
    z: np.ndarray,
    currents: np.ndarray,
    x_m: float,
    rigid_samples: int,
) -> np.ndarray:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import jax
    import jax.numpy as jnp
    from single_layer_ik.joint_problem import build_params_for_y, build_static_fk, extract_centerline_p_dim

    static_fk = build_static_fk(
        flex_lengths_tail=[FLEX_LENGTH],
        rigid_lengths=[RIGID_LENGTH],
        M_list=[int(M_LIST)],
        L_protrude_max=X_REF,
        L1_min=1e-6,
        p0_dim=_parse_csv_floats(P0),
        Q0_wxyz=_parse_csv_floats(Q0),
        axis_body=_parse_csv_floats(AXIS_BODY),
        enable_gravity=True,
        g_world=_parse_csv_floats(G_WORLD),
        enable_magnetics=True,
        calib_file=str(CALIB_FILE),
        actuation_table_pkl=str(ACTUATION_TABLE),
        m_body_list=[_parse_csv_floats(M_BODY_LIST)],
    )
    params = build_params_for_y(
        static_fk,
        I=jnp.asarray(currents, dtype=jnp.float64),
        x=jnp.asarray(float(x_m), dtype=jnp.float64),
    )
    centerline = extract_centerline_p_dim(
        z_bar=jnp.asarray(z, dtype=jnp.float64),
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=int(rigid_samples),
    )
    return np.asarray(jax.device_get(centerline), dtype=float)


def _parse_final_normE(log_path: Path) -> tuple[float, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"Final \|\|E\|\|=([0-9.eE+-]+)", text)
    norm = float(m[-1]) if m else float("nan")
    it = re.findall(r"iter=([0-9]+), \|\|E\|\|=", text)
    n_iter = int(it[-1]) if it else -1
    return norm, n_iter


def _resample_polyline(points: np.ndarray, n: int = 240) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    grid = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(grid, s, pts[:, j]) for j in range(3)])


def _centerline_rms_mm(a: np.ndarray, b: np.ndarray) -> float:
    aa = _resample_polyline(a)
    bb = _resample_polyline(b)
    return float(np.sqrt(np.mean(np.sum((aa - bb) ** 2, axis=1))) * 1e3)


def prepare_oracles(force: bool = False) -> dict[str, np.ndarray]:
    oracle_npz = OUT_DIR / "oracle_centerlines.npz"
    target_csv = IK_SOURCE_TARGETS
    if oracle_npz.exists() and target_csv.exists() and not force:
        loaded = np.load(oracle_npz)
        return {k: loaded[k] for k in loaded.files}

    payload: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for case in _load_oracle_cases(prefer_same_source=False):
        engine = _ik_fk_engine(max_iter=6000, tol=1e-6, lm_damping=0.1)
        z_fk, params, _meshes, ok, stats = engine.solve_with_stats(
            coil_currents=case.currents,
            L_protrude=X_REF,
            warm_start=False,
            return_stats=True,
            stats_include_jacobian=False,
            verbose=False,
        )
        z_np = np.asarray(z_fk, dtype=float)
        centerline = _centerline_from_ik_params(z_np, params, rigid_samples=10)
        tip = centerline[-1]
        external_delta_mm = float(np.linalg.norm(tip - case.p_des) * 1e3)
        oracle_dir = OUT_DIR / "oracles" / case.name
        oracle_dir.mkdir(parents=True, exist_ok=True)
        normE = float("nan") if stats is None else float(getattr(stats, "normE", float("nan")))
        n_iter = -1 if stats is None else int(getattr(stats, "n_iter", -1))
        np.savez_compressed(
            oracle_dir / "same_source_fk_solution.npz",
            z=z_np,
            centerline=centerline,
            coil_currents=case.currents,
            p_tip=tip,
            x=np.asarray([X_REF], dtype=float),
            normE=np.asarray([normE], dtype=float),
            ok=np.asarray([bool(ok)]),
        )
        payload[case.name] = centerline
        rows.append(
            {
                "case": case.name,
                "active_coil": case.active_coil,
                "I_FK_A": _fmt_vec(case.currents, 8),
                "x_FK_m": f"{X_REF:.8f}",
                "L1_FK_m": f"{FLEX_LENGTH:.8f}",
                "p_des_m": _fmt_vec(tip, 12),
                "same_source_fk_normE": f"{normE:.6e}",
                "same_source_ok": bool(ok),
                "same_source_iteration_count": n_iter,
                "same_source_path": str(oracle_dir / "same_source_fk_solution.npz"),
                "external_fk_path": str(case.source_dir),
                "external_fk_tip_delta_mm": f"{external_delta_mm:.6g}",
            }
        )
    np.savez_compressed(oracle_npz, **payload)
    _write_csv(target_csv, rows)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _attempt_specs(case: OracleCase, phase: str) -> list[tuple[str, str, np.ndarray, np.ndarray, float]]:
    specs: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []
    if phase in {"oracle", "full"}:
        specs.append(("oracle_current", "oracle", case.currents.copy(), case.currents.copy(), 0.02))
    if phase in {"multistart", "full"}:
        zero = np.zeros(8, dtype=float)
        specs.append(("zero_current", "blind", zero.copy(), zero.copy(), 0.005))
        for idx in range(8):
            init = np.zeros(8, dtype=float)
            init[idx] = 15.0
            specs.append((f"coil{idx + 1}_15A_start", "blind", init, zero.copy(), 0.005))
    return specs


def _run_ik(
    case: OracleCase,
    attempt: str,
    role: str,
    I_init: np.ndarray,
    I_ref: np.ndarray,
    w_I: float,
    *,
    force: bool,
    x_min: float = 0.04995,
    x_max: float = 0.05005,
    x_init: float = X_REF,
    x_ref: float = X_REF,
    dist_tol_mm: float = 0.25,
    block_clip_dx: float = 0.00005,
    max_iter: int = 220,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    run_dir = OUT_DIR / "runs" / case.name / attempt
    state_path = run_dir / "state_final.npz"
    if state_path.exists() and not force:
        return run_dir
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "-m",
        "single_layer_ik.run_joint_ik",
        "--p_des",
        _fmt_vec(case.p_des, 12),
        "--run_mode",
        "v1_staged",
        "--flex_lengths_tail",
        f"{FLEX_LENGTH:.8f}",
        "--rigid_lengths",
        f"{RIGID_LENGTH:.8f}",
        "--M_list",
        M_LIST,
        "--x_min",
        f"{float(x_min):.12g}",
        "--x_max",
        f"{float(x_max):.12g}",
        "--L_protrude_max",
        f"{X_REF:.8f}",
        "--x_init",
        f"{float(x_init):.12g}",
        "--x_ref",
        f"{float(x_ref):.12g}",
        "--p0_dim",
        P0,
        "--Q0_wxyz",
        Q0,
        "--axis_body",
        AXIS_BODY,
        "--tip_mode",
        "distal_tip",
        "--enable_gravity",
        "--g_world",
        G_WORLD,
        "--enable_magnetics",
        "--calib_file",
        str(CALIB_FILE),
        "--actuation_table_pkl",
        str(ACTUATION_TABLE),
        "--m_body_list",
        M_BODY_LIST,
        "--I_max",
        "50",
        "--I_scale",
        "1.0",
        "--I_init",
        _fmt_vec(I_init, 8),
        "--I_ref",
        _fmt_vec(I_ref, 8),
        "--w_E",
        "100",
        "--sigma_p",
        "0.001",
        "--w_I",
        f"{w_I:.8g}",
        "--w_x",
        "100",
        "--w_smooth",
        "0.2",
        "--w_smooth_baseline",
        "0.2",
        "--w_gap",
        "0",
        "--sigma_gap",
        "0.0005",
        "--init_mode",
        "npz" if role in {"oracle", "perturbed"} else "geometry",
        "--max_iter",
        str(int(max_iter)),
        "--tol",
        "1e-6",
        "--dist_tol_mm",
        f"{float(dist_tol_mm):.12g}",
        "--lm_damping",
        "0.01",
        "--step_norm_clip",
        "50",
        "--block_clip_dz",
        "20",
        "--block_clip_dI",
        "10",
        "--block_clip_dx",
        f"{float(block_clip_dx):.12g}",
        "--stagnation_patience",
        "90",
        "--reject_patience",
        "25",
        "--pose_save_every",
        "20",
        "--out_dir",
        str(run_dir),
    ]
    if role in {"oracle", "perturbed"}:
        cmd.extend(["--init_z_npz", str(OUT_DIR / "oracles" / case.name / "same_source_fk_solution.npz")])
    metadata = {
        "case": case.name,
        "attempt": attempt,
        "role": role,
        "I_init_A": I_init.tolist(),
        "I_ref_A": I_ref.tolist(),
        "I_FK_A": case.currents.tolist(),
        "x_init_m": float(x_init),
        "x_ref_m": float(x_ref),
        "x_min_m": float(x_min),
        "x_max_m": float(x_max),
        "p_des_m": case.p_des.tolist(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (run_dir / "attempt_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _run_logged(cmd, run_dir / "ik_run.log")
    return run_dir


def _run_fk_back(run_dir: Path, *, force: bool) -> dict[str, Any]:
    state = np.load(run_dir / "state_final.npz")
    I_star = np.asarray(state["I_star"], dtype=float).reshape(8)
    x_star = float(np.asarray(state["x_star"], dtype=float).reshape(-1)[0])
    z_star = np.asarray(state["z_star_bar"], dtype=float).reshape(-1)
    fk_dir = run_dir / "fk_back_same_source"
    final_npz = fk_dir / "final_solution.npz"
    log_path = fk_dir / "fk_back.log"
    if force and fk_dir.exists():
        shutil.rmtree(fk_dir)
    fk_dir.mkdir(parents=True, exist_ok=True)
    centerline_path = fk_dir / "centerline.npz"
    if final_npz.exists() and centerline_path.exists() and not force:
        loaded = np.load(final_npz)
        normE = float(np.asarray(loaded["normE"]).reshape(-1)[0]) if "normE" in loaded.files else float("nan")
        n_iter = int(np.asarray(loaded["n_iter"]).reshape(-1)[0]) if "n_iter" in loaded.files else -1
        centerline_fk = np.load(centerline_path)["centerline"]
        return {"fk_normE": normE, "fk_n_iter": n_iter, "centerline": centerline_fk}
    if not final_npz.exists():
        with log_path.open("w", encoding="utf-8") as f:
            f.write("[same-source FK] single_layer_ik.forward_kinematics_optimized_bak.fk_engine\n")
            f.write(f"I_star={_fmt_vec(I_star, 12)}\n")
            f.write(f"x_star={x_star:.12g}\n")
            f.flush()
            engine = _ik_fk_engine(max_iter=6000, tol=1e-6, lm_damping=0.1)
            z_fk, params, _meshes, ok, stats = engine.solve_with_stats(
                coil_currents=I_star,
                L_protrude=x_star,
                warm_start=False,
                override_z0_bar=z_star,
                return_stats=True,
                stats_include_jacobian=False,
                verbose=False,
            )
            normE = float("nan") if stats is None else float(getattr(stats, "normE", float("nan")))
            n_iter = -1 if stats is None else int(getattr(stats, "n_iter", -1))
            centerline_fk = _centerline_from_ik_params(np.asarray(z_fk, dtype=float), params, rigid_samples=10)
            np.savez_compressed(
                final_npz,
                z=np.asarray(z_fk, dtype=float),
                ok=np.asarray([bool(ok)]),
                max_iter=np.asarray([6000]),
                tol=np.asarray([1e-6]),
                normE=np.asarray([normE]),
                n_iter=np.asarray([n_iter]),
            )
            f.write(f"ok={bool(ok)}\n")
            f.write(f"Final ||E||={normE:.12e}\n")
            f.write(f"n_iter={n_iter}\n")
    else:
        raise FileNotFoundError(f"Missing cached centerline for {final_npz}; rerun with --force_fk")
    np.savez_compressed(centerline_path, centerline=centerline_fk)
    return {"fk_normE": normE, "fk_n_iter": n_iter, "centerline": centerline_fk}


def _summarize_attempt(case: OracleCase, run_dir: Path, oracle_centerline: np.ndarray, *, force_fk: bool) -> dict[str, Any]:
    metadata = json.loads((run_dir / "attempt_metadata.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    state = np.load(run_dir / "state_final.npz")
    centerline_ik = np.asarray(state["centerline_ik"], dtype=float)
    p_tip_ik = np.asarray(state["p_tip_ik"], dtype=float).reshape(3)
    p_des = case.p_des
    I_star = np.asarray(state["I_star"], dtype=float).reshape(8)
    x_star = float(np.asarray(state["x_star"], dtype=float).reshape(-1)[0])
    fk_back = _run_fk_back(run_dir, force=force_fk)
    centerline_fk = np.asarray(fk_back["centerline"], dtype=float)
    p_tip_fk = centerline_fk[-1]
    ik_target_err_mm = float(np.linalg.norm(p_tip_ik - p_des) * 1e3)
    fk_target_err_mm = float(np.linalg.norm(p_tip_fk - p_des) * 1e3)
    ik_fk_tip_mm = float(np.linalg.norm(p_tip_ik - p_tip_fk) * 1e3)
    ik_fk_centerline_rms_mm = _centerline_rms_mm(centerline_ik, centerline_fk)
    oracle_fk_centerline_rms_mm = _centerline_rms_mm(oracle_centerline, centerline_fk)
    x_abs_err_mm = abs(x_star - X_REF) * 1e3
    passed = bool(
        fk_back["fk_normE"] <= PASS_LIMITS["fk_normE"]
        and ik_target_err_mm <= PASS_LIMITS["ik_target_err_mm"]
        and fk_target_err_mm <= PASS_LIMITS["fk_target_err_mm"]
        and ik_fk_tip_mm <= PASS_LIMITS["ik_fk_tip_mm"]
        and ik_fk_centerline_rms_mm <= PASS_LIMITS["ik_fk_centerline_rms_mm"]
        and oracle_fk_centerline_rms_mm <= PASS_LIMITS["oracle_fk_centerline_rms_mm"]
        and x_abs_err_mm <= PASS_LIMITS["x_abs_err_mm"] + 1e-9
    )
    row = {
        "case": case.name,
        "active_coil": case.active_coil,
        "attempt": metadata["attempt"],
        "role": metadata["role"],
        "pass": passed,
        "ik_stop_reason": result.get("stop_reason", ""),
        "ik_n_iter": int(result.get("n_iter", -1)),
        "ik_normE_raw": float(result.get("diag", {}).get("norm_E_raw", math.nan)),
        "fk_back_normE": float(fk_back["fk_normE"]),
        "fk_back_n_iter": int(fk_back["fk_n_iter"]),
        "ik_target_err_mm": ik_target_err_mm,
        "fk_target_err_mm": fk_target_err_mm,
        "ik_fk_tip_mm": ik_fk_tip_mm,
        "ik_fk_centerline_rms_mm": ik_fk_centerline_rms_mm,
        "oracle_fk_centerline_rms_mm": oracle_fk_centerline_rms_mm,
        "x_star_m": x_star,
        "x_abs_err_mm": x_abs_err_mm,
        "x_init_m": float(metadata.get("x_init_m", float("nan"))),
        "x_init_rel_delta": float(metadata.get("x_init_rel_delta", float("nan"))),
        "I_FK_A": _fmt_vec(case.currents, 8),
        "I_init_A": _fmt_vec(np.asarray(metadata.get("I_init_A", [float("nan")] * 8), dtype=float), 8),
        "I_star_A": _fmt_vec(I_star, 8),
        "I_init_rel_delta": float(metadata.get("I_init_rel_delta", float("nan"))),
        "I_recovery_l2_A": float(np.linalg.norm(I_star - case.currents)),
        "dominant_I_index_1based": int(np.argmax(np.abs(I_star)) + 1),
        "dominant_I_A": float(I_star[int(np.argmax(np.abs(I_star)))]),
        "run_dir": str(run_dir.relative_to(OUT_DIR)),
    }
    (run_dir / "official_fk_roundtrip_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_suite(phase: str, *, force_ik: bool = False, force_fk: bool = False) -> None:
    oracle_centerlines = prepare_oracles(force=False)
    rows: list[dict[str, Any]] = []
    for case in _load_oracle_cases():
        print(f"[case] {case.name} phase={phase}", flush=True)
        specs = _attempt_specs(case, phase)
        if phase in {"multistart", "full"}:
            oracle_dir = OUT_DIR / "runs" / case.name / "oracle_current"
            if not oracle_dir.exists():
                print(f"[skip blind] missing oracle_current for {case.name}", flush=True)
                continue
            oracle_row = _summarize_attempt(case, oracle_dir, oracle_centerlines[case.name], force_fk=False)
            if not bool(oracle_row["pass"]):
                print(f"[skip blind] oracle_current failed for {case.name}", flush=True)
                rows.append(oracle_row)
                continue
        for attempt, role, I_init, I_ref, w_I in specs:
            print(f"[ik] {case.name}/{attempt}", flush=True)
            run_dir = _run_ik(case, attempt, role, I_init, I_ref, w_I, force=force_ik)
            row = _summarize_attempt(case, run_dir, oracle_centerlines[case.name], force_fk=force_fk)
            rows.append(row)
            print(
                "[summary] {case}/{attempt} pass={passed} ik={ik:.3g}mm fk={fk:.3g}mm rms={rms:.3g}mm".format(
                    case=case.name,
                    attempt=attempt,
                    passed=row["pass"],
                    ik=row["ik_target_err_mm"],
                    fk=row["fk_target_err_mm"],
                    rms=row["oracle_fk_centerline_rms_mm"],
                ),
                flush=True,
            )
    summarize_existing()


def run_perturbed_init_suite(*, force_ik: bool = False, force_fk: bool = False, seed: int = 20260619) -> None:
    oracle_centerlines = prepare_oracles(force=False)
    rng = np.random.default_rng(int(seed))
    near_limit_current_delta = {
        "coil2_15A": -0.098,
        "coil3_15A": 0.098,
        "coil4_15A": -0.098,
    }
    rows: list[dict[str, Any]] = []
    for case in _load_oracle_cases():
        delta_i = float(rng.uniform(-0.10, 0.10))
        delta_x = float(rng.uniform(-0.10, 0.10))
        delta_i = near_limit_current_delta.get(case.name, delta_i)
        I_init = case.currents.copy()
        nz = np.abs(case.currents) > 1e-12
        I_init[nz] = I_init[nz] * (1.0 + delta_i)
        x_init = X_REF * (1.0 + delta_x)
        attempt = "perturbed_init_10pct"
        print(
            f"[perturb] {case.name} delta_I={delta_i:+.4f} delta_x={delta_x:+.4f} "
            f"I_init={_fmt_vec(I_init, 6)} x_init={x_init:.6g}",
            flush=True,
        )
        run_dir = _run_ik(
            case,
            attempt,
            "perturbed",
            I_init,
            case.currents.copy(),
            0.02,
            force=force_ik,
            x_min=X_REF * 0.90,
            x_max=X_REF * 1.10,
            x_init=x_init,
            x_ref=X_REF,
            dist_tol_mm=0.0,
            block_clip_dx=0.001,
            max_iter=45,
            extra_metadata={
                "seed": int(seed),
                "I_init_rel_delta": delta_i,
                "x_init_rel_delta": delta_x,
                "near_limit_current_delta": case.name in near_limit_current_delta,
                "max_iter": 45,
            },
        )
        row = _summarize_attempt(case, run_dir, oracle_centerlines[case.name], force_fk=force_fk)
        rows.append(row)
        print(
            "[perturb-summary] {case} pass={passed} ik={ik:.3g}mm fk={fk:.3g}mm x={x:.6g}m".format(
                case=case.name,
                passed=row["pass"],
                ik=row["ik_target_err_mm"],
                fk=row["fk_target_err_mm"],
                x=row["x_star_m"],
            ),
            flush=True,
        )
    _write_csv(OUT_DIR / "perturbed_init_summary.csv", rows)


def summarize_existing() -> None:
    oracle_centerlines = prepare_oracles(force=False)
    rows: list[dict[str, Any]] = []
    for case in _load_oracle_cases():
        case_dir = OUT_DIR / "runs" / case.name
        if not case_dir.exists():
            continue
        for run_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            if not (run_dir / "state_final.npz").exists():
                continue
            rows.append(_summarize_attempt(case, run_dir, oracle_centerlines[case.name], force_fk=False))
    rows.sort(key=lambda r: (r["case"], 0 if r["attempt"] == "oracle_current" else 1, r["fk_target_err_mm"]))
    _write_csv(OUT_DIR / "case_attempt_summary.csv", rows)
    best_rows: list[dict[str, Any]] = []
    for case in _load_oracle_cases():
        case_rows = [r for r in rows if r["case"] == case.name]
        if not case_rows:
            continue
        blind_pass = [r for r in case_rows if r["role"] == "blind" and bool(r["pass"])]
        oracle_pass = [r for r in case_rows if r["role"] == "oracle" and bool(r["pass"])]
        if blind_pass:
            chosen = min(blind_pass, key=lambda r: (r["fk_target_err_mm"], r["oracle_fk_centerline_rms_mm"]))
            verdict = "blind_pass"
        elif oracle_pass:
            chosen = min(oracle_pass, key=lambda r: (r["fk_target_err_mm"], r["oracle_fk_centerline_rms_mm"]))
            verdict = "oracle_only_pass"
        else:
            chosen = min(case_rows, key=lambda r: (r["fk_target_err_mm"], r["oracle_fk_centerline_rms_mm"]))
            verdict = "failed"
        out = dict(chosen)
        out["selection_verdict"] = verdict
        best_rows.append(out)
    _write_csv(OUT_DIR / "selected_results.csv", best_rows)
    (OUT_DIR / "pass_criteria.json").write_text(json.dumps(PASS_LIMITS, indent=2), encoding="utf-8")


def _font_setup():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    zh_font = REPO / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf"
    if zh_font.exists():
        font_manager.fontManager.addfont(str(zh_font))
        zh_family = font_manager.FontProperties(fname=str(zh_font)).get_name()
    else:
        zh_family = "WenQuanYi Zen Hei"
    en_font = Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
    if en_font.exists():
        font_manager.fontManager.addfont(str(en_font))
        en_family = font_manager.FontProperties(fname=str(en_font)).get_name()
    else:
        en_family = "DejaVu Serif"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [en_family, zh_family, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    cjk_prop = font_manager.FontProperties(fname=str(zh_font)) if zh_font.exists() else font_manager.FontProperties(family=[zh_family])
    latin_prop = font_manager.FontProperties(fname=str(en_font)) if en_font.exists() else font_manager.FontProperties(family=[en_family])
    return cjk_prop, latin_prop


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_selected_centerlines(selection_csv: Path | None = None) -> list[dict[str, Any]]:
    selected = _read_csv(selection_csv if selection_csv is not None else OUT_DIR / "selected_results.csv")
    oracle = prepare_oracles(force=False)
    items: list[dict[str, Any]] = []
    for row in selected:
        run_dir = OUT_DIR / row["run_dir"]
        state = np.load(run_dir / "state_final.npz")
        cfg = json.loads((run_dir / "config_full.json").read_text(encoding="utf-8"))
        rigid_samples = int(cfg.get("diagnostics", {}).get("centerline_rigid_samples", 2))
        fk_back_npz = np.load(run_dir / "fk_back_same_source" / "final_solution.npz")
        fk_back = _centerline_from_saved_same_source_z(
            z=np.asarray(fk_back_npz["z"], dtype=float),
            currents=np.asarray(state["I_star"], dtype=float).reshape(8),
            x_m=float(np.asarray(state["x_star"], dtype=float).reshape(-1)[0]),
            rigid_samples=rigid_samples,
        )
        items.append(
            {
                "row": row,
                "oracle": oracle[row["case"]],
                "ik": np.asarray(state["centerline_ik"], dtype=float),
                "fk_back": np.asarray(fk_back, dtype=float),
                "target": np.asarray(_parse_csv_floats(row.get("p_des_m", "")), dtype=float)
                if row.get("p_des_m")
                else np.asarray(state["p_des"], dtype=float),
            }
        )
    return items


def _font_with_size(prop: Any, size: float) -> Any:
    out = prop.copy()
    out.set_size(float(size))
    return out


def _panel_case_title(row: dict[str, str], *, title_mode: str) -> str:
    if title_mode == "perturbed":
        dI = float(row.get("I_init_rel_delta", "nan")) * 100.0
        dx = float(row.get("x_init_rel_delta", "nan")) * 100.0
        return f"C{row['active_coil']} 15 A (δI={dI:+.1f}%, δx={dx:+.1f}%)"
    if title_mode == "bad_init":
        return f"C{row['active_coil']} 15 A：错误初值"
    return f"C{row['active_coil']} 15 A"


def _pose_error_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ik_pts = np.asarray(item["ik"], dtype=float)
    fk_pts = np.asarray(item["fk_back"], dtype=float)
    if ik_pts.shape != fk_pts.shape:
        fk_cmp = _resample_polyline(fk_pts, n=ik_pts.shape[0])
    else:
        fk_cmp = fk_pts
    x_axis = np.arange(ik_pts.shape[0])
    err_mm = np.linalg.norm(ik_pts - fk_cmp, axis=1) * 1e3
    return x_axis, err_mm, fk_cmp


def _set_compact_3d_limits(ax: Any, points_mm: np.ndarray) -> None:
    pts = np.asarray(points_mm, dtype=float)
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = np.maximum(span * 0.12, 1.2)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])


def _plot_single_pose_panel(
    *,
    item: dict[str, Any],
    out_base: str,
    cjk_prop: Any,
    latin_prop: Any,
    title_mode: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    title_prop = _font_with_size(latin_prop, 31.0)
    label_prop = _font_with_size(latin_prop, 26.0)
    tick_prop = _font_with_size(latin_prop, 23.0)
    legend_prop = _font_with_size(cjk_prop, 25.0)
    fig = plt.figure(figsize=(16.0, 9.0))
    ax = fig.add_axes([0.01, 0.04, 0.81, 0.84], projection="3d")
    row = item["row"]
    for pts, label, color, lw, ls, alpha, zorder in [
        (item["fk_back"], "FK回代", "#0072B2", 4.4, "-", 0.62, 2),
        (item["ik"], "IK终态", "#D55E00", 4.0, "--", 0.98, 3),
    ]:
        mm = pts * 1e3
        ax.plot(
            mm[:, 0],
            mm[:, 1],
            mm[:, 2],
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
    target = item["fk_back"][-1] * 1e3
    ax.scatter([target[0]], [target[1]], [target[2]], marker="*", s=500, color="#C91F37", label="目标点")
    title = _panel_case_title(row, title_mode=title_mode)
    fig.text(0.42, 0.945, title, ha="center", va="center", fontproperties=title_prop)
    ax.set_xlabel("x/mm", fontproperties=label_prop, labelpad=10)
    ax.set_ylabel("y/mm", fontproperties=label_prop, labelpad=16)
    ax.set_zlabel("")
    ax.text2D(0.955, 0.50, "z/mm", transform=ax.transAxes, rotation=90, fontproperties=label_prop)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.35)
    all_pts_mm = np.vstack([np.asarray(item["fk_back"], dtype=float), np.asarray(item["ik"], dtype=float)]) * 1e3
    _set_compact_3d_limits(ax, all_pts_mm)
    ax.set_box_aspect([1.25, 0.72, 0.92])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", pad=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(tick_prop)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=True,
        prop=legend_prop,
        loc="upper left",
        bbox_to_anchor=(0.79, 0.76),
        borderaxespad=0.0,
        handlelength=1.4,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.86)
    legend.get_frame().set_linewidth(0.0)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{out_base}.{suffix}", dpi=360)
    plt.close(fig)


def _plot_single_error_panel(
    *,
    item: dict[str, Any],
    out_base: str,
    cjk_prop: Any,
    latin_prop: Any,
) -> None:
    import matplotlib.pyplot as plt

    title_prop = _font_with_size(cjk_prop, 45.0)
    label_prop = _font_with_size(cjk_prop, 42.0)
    tick_prop = _font_with_size(latin_prop, 36.0)
    note_prop = _font_with_size(latin_prop, 36.0)
    x_axis, err_mm, _ = _pose_error_arrays(item)
    fig, ax = plt.subplots(figsize=(12.0, 8.4), constrained_layout=True)
    ax.plot(x_axis, err_mm, color="#009E73", lw=3.0, marker="o", ms=12.6)
    ax.fill_between(x_axis, 0.0, err_mm, color="#009E73", alpha=0.13)
    ax.set_title("IK-FK对应节点误差", fontproperties=title_prop, pad=18)
    ax.set_xlabel("采样节点编号", fontproperties=label_prop, labelpad=14)
    ax.set_ylabel("距离误差 (mm)", fontproperties=label_prop, labelpad=14)
    ax.grid(True, which="major", color="#b7b7b7", alpha=0.30)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_prop)
    ax.yaxis.get_offset_text().set_fontproperties(tick_prop)
    ax.text(
        0.98,
        0.88,
        f"max={float(np.max(err_mm)):.2e} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontproperties=note_prop,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{out_base}.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_bad_init_failure(row: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cjk_prop, latin_prop = _font_setup()
    title_prop = _font_with_size(cjk_prop, 31.0)
    label_prop = _font_with_size(latin_prop, 26.0)
    tick_prop = _font_with_size(latin_prop, 23.0)
    legend_prop = _font_with_size(cjk_prop, 25.0)

    run_dir = OUT_DIR / row["run_dir"]
    state = np.load(run_dir / "state_final.npz")
    oracle = prepare_oracles(force=False)[row["case"]]
    ik = np.asarray(state["centerline_ik"], dtype=float)
    target = oracle[-1] * 1e3

    fig = plt.figure(figsize=(16.0, 9.0))
    ax = fig.add_axes([0.01, 0.04, 0.81, 0.84], projection="3d")
    for pts, label, color, lw, ls, alpha, zorder in [
        (oracle, "目标FK姿态", "#0072B2", 4.4, "-", 0.72, 2),
        (ik, "错误初值IK终态", "#D55E00", 4.0, "--", 0.98, 3),
    ]:
        mm = np.asarray(pts, dtype=float) * 1e3
        ax.plot(mm[:, 0], mm[:, 1], mm[:, 2], color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder, label=label)
    ax.scatter([target[0]], [target[1]], [target[2]], marker="*", s=500, color="#C91F37", label="目标点")

    title = (
        f"C{row['active_coil']} 15 A 错误初值 "
        f"(末端误差={float(row['ik_target_err_mm']):.2f} mm)"
    )
    fig.text(0.42, 0.945, title, ha="center", va="center", fontproperties=title_prop)
    ax.set_xlabel("x/mm", fontproperties=label_prop, labelpad=10)
    ax.set_ylabel("y/mm", fontproperties=label_prop, labelpad=16)
    ax.set_zlabel("")
    ax.text2D(0.955, 0.50, "z/mm", transform=ax.transAxes, rotation=90, fontproperties=label_prop)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.35)
    all_pts_mm = np.vstack([oracle, ik]) * 1e3
    _set_compact_3d_limits(ax, all_pts_mm)
    ax.set_box_aspect([1.25, 0.72, 0.92])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", pad=5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(tick_prop)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=True,
        prop=legend_prop,
        loc="upper left",
        bbox_to_anchor=(0.79, 0.76),
        borderaxespad=0.0,
        handlelength=1.4,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.86)
    legend.get_frame().set_linewidth(0.0)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"fig_322_single_layer_bad_init_failure_pose.{suffix}", dpi=360)
    plt.close(fig)


def _summarize_bad_init_without_fk(case: OracleCase, run_dir: Path, oracle_centerline: np.ndarray) -> dict[str, Any]:
    metadata = json.loads((run_dir / "attempt_metadata.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    state = np.load(run_dir / "state_final.npz")
    centerline_ik = np.asarray(state["centerline_ik"], dtype=float)
    p_tip_ik = np.asarray(state["p_tip_ik"], dtype=float).reshape(3)
    I_star = np.asarray(state["I_star"], dtype=float).reshape(8)
    x_star = float(np.asarray(state["x_star"], dtype=float).reshape(-1)[0])
    ik_target_err_mm = float(np.linalg.norm(p_tip_ik - case.p_des) * 1e3)
    oracle_ik_centerline_rms_mm = _centerline_rms_mm(oracle_centerline, centerline_ik)
    row = {
        "case": case.name,
        "active_coil": case.active_coil,
        "attempt": metadata["attempt"],
        "role": metadata["role"],
        "pass": False,
        "failure_type": "bad_initialization_branch",
        "ik_stop_reason": result.get("stop_reason", ""),
        "ik_n_iter": int(result.get("n_iter", -1)),
        "ik_normE_raw": float(result.get("diag", {}).get("norm_E_raw", math.nan)),
        "ik_target_err_mm": ik_target_err_mm,
        "oracle_ik_centerline_rms_mm": oracle_ik_centerline_rms_mm,
        "x_star_m": x_star,
        "x_init_m": float(metadata.get("x_init_m", float("nan"))),
        "I_FK_A": _fmt_vec(case.currents, 8),
        "I_init_A": _fmt_vec(np.asarray(metadata.get("I_init_A", [float("nan")] * 8), dtype=float), 8),
        "I_star_A": _fmt_vec(I_star, 8),
        "I_recovery_l2_A": float(np.linalg.norm(I_star - case.currents)),
        "dominant_I_index_1based": int(np.argmax(np.abs(I_star)) + 1),
        "dominant_I_A": float(I_star[int(np.argmax(np.abs(I_star)))]),
        "run_dir": str(run_dir.relative_to(OUT_DIR)),
    }
    return row


def run_bad_init_failure_case(*, force_ik: bool = False, force_fk: bool = False) -> None:
    del force_fk
    oracle_centerlines = prepare_oracles(force=False)
    cases = {case.name: case for case in _load_oracle_cases()}
    case = cases["coil4_15A"]
    I_bad = np.zeros(8, dtype=float)
    I_bad[0] = 15.0
    run_dir = _run_ik(
        case,
        "bad_init_coil1_15A_geometry",
        "blind",
        I_bad,
        I_bad.copy(),
        0.02,
        force=force_ik,
        x_min=X_REF * 0.90,
        x_max=X_REF * 1.10,
        x_init=X_REF * 0.90,
        x_ref=X_REF,
        dist_tol_mm=0.0,
        block_clip_dx=0.001,
        max_iter=45,
        extra_metadata={
            "failure_design": "target is coil4_15A; initial current and current reference use coil1_15A",
            "I_init_description": "C1 15 A",
            "target_description": "C4 15 A same-source FK target",
            "max_iter": 45,
        },
    )
    row = _summarize_bad_init_without_fk(case, run_dir, oracle_centerlines[case.name])
    _write_csv(OUT_DIR / "bad_init_failure_summary.csv", [row])
    (OUT_DIR / "bad_init_failure_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    _plot_bad_init_failure(row)
    print(
        "[bad-init] pass={passed} ik={ik:.3g}mm rms={rms:.3g}mm I*={I}".format(
            passed=row["pass"],
            ik=row["ik_target_err_mm"],
            rms=row["oracle_ik_centerline_rms_mm"],
            I=row["I_star_A"],
        ),
        flush=True,
    )


def _plot_pose_error_single_panels(
    *,
    items: list[dict[str, Any]],
    out_prefix: str,
    cjk_prop: Any,
    latin_prop: Any,
    title_mode: str,
) -> None:
    for idx, item in enumerate(items, 1):
        active = str(item["row"]["active_coil"])
        _plot_single_pose_panel(
            item=item,
            out_base=f"{out_prefix}_panel_{2 * idx - 1:02d}_C{active}_pose",
            cjk_prop=cjk_prop,
            latin_prop=latin_prop,
            title_mode=title_mode,
        )
        _plot_single_error_panel(
            item=item,
            out_base=f"{out_prefix}_panel_{2 * idx:02d}_C{active}_error",
            cjk_prop=cjk_prop,
            latin_prop=latin_prop,
        )


def _plot_pose_error_4x2(
    *,
    items: list[dict[str, Any]],
    out_prefix: str,
    cjk_prop: Any,
    latin_prop: Any,
    title_mode: str = "simple",
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.8, 14.8), constrained_layout=True)
    subfigs = fig.subfigures(4, 1, hspace=0.06)
    for idx, (subfig, item) in enumerate(zip(subfigs, items), 1):
        axes_row = subfig.subplots(1, 2, gridspec_kw={"width_ratios": [1.12, 1.0]})
        ax = axes_row[0]
        ax.remove()
        ax = subfig.add_subplot(1, 2, 1, projection="3d")
        ax_err = axes_row[1]
        row = item["row"]

        for pts, label, color, lw, ls, alpha, zorder in [
            (item["fk_back"], "FK回代", "#0072B2", 2.8, "-", 0.58, 2),
            (item["ik"], "IK终态", "#D55E00", 2.3, "--", 0.98, 3),
        ]:
            mm = pts * 1e3
            ax.plot(
                mm[:, 0],
                mm[:, 1],
                mm[:, 2],
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
        target = item["fk_back"][-1] * 1e3
        ax.scatter([target[0]], [target[1]], [target[2]], marker="*", s=80, color="#C91F37", label="目标点")
        if title_mode == "perturbed":
            dI = float(row.get("I_init_rel_delta", "nan")) * 100.0
            dx = float(row.get("x_init_rel_delta", "nan")) * 100.0
            title = f"C{row['active_coil']} 15 A (δI={dI:+.1f}%, δx={dx:+.1f}%)"
        else:
            title = f"C{row['active_coil']} 15 A"
        ax.set_title(title, fontproperties=latin_prop)
        ax.set_xlabel("x (mm)", fontproperties=latin_prop)
        ax.set_ylabel("y (mm)", fontproperties=latin_prop)
        ax.set_zlabel("z (mm)", fontproperties=latin_prop)
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.35)
        ax.set_box_aspect([1.0, 0.75, 0.85])
        if idx == 1:
            ax.legend(frameon=False, prop=cjk_prop, loc="upper left", bbox_to_anchor=(-0.03, 1.02))

        ik_pts = np.asarray(item["ik"], dtype=float)
        fk_pts = np.asarray(item["fk_back"], dtype=float)
        if ik_pts.shape != fk_pts.shape:
            ik_cmp = ik_pts
            fk_cmp = _resample_polyline(fk_pts, n=ik_pts.shape[0])
        else:
            ik_cmp = ik_pts
            fk_cmp = fk_pts
        x_axis = np.arange(ik_cmp.shape[0])
        err_mm = np.linalg.norm(ik_cmp - fk_cmp, axis=1) * 1e3
        ax_err.plot(x_axis, err_mm, color="#009E73", lw=2.4, marker="o", ms=4.2)
        ax_err.fill_between(x_axis, 0.0, err_mm, color="#009E73", alpha=0.13)
        ax_err.set_title("IK-FK对应节点误差", fontproperties=cjk_prop)
        ax_err.set_xlabel("采样节点编号", fontproperties=cjk_prop)
        ax_err.set_ylabel("距离误差 (mm)", fontproperties=cjk_prop)
        ax_err.grid(True, which="major", color="#b7b7b7", alpha=0.30)
        ax_err.grid(True, which="minor", color="#d0d0d0", alpha=0.15)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)
        ax_err.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax_err.text(
            0.98,
            0.88,
            f"max={float(np.max(err_mm)):.2e} mm",
            transform=ax_err.transAxes,
            ha="right",
            va="top",
            fontproperties=latin_prop,
        )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{out_prefix}.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)
    _plot_pose_error_single_panels(
        items=items,
        out_prefix=out_prefix,
        cjk_prop=cjk_prop,
        latin_prop=latin_prop,
        title_mode=title_mode,
    )


def plot_figures() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    cjk_prop, latin_prop = _font_setup()
    summarize_existing()
    selected_rows = _read_csv(OUT_DIR / "selected_results.csv")
    all_rows = _read_csv(OUT_DIR / "case_attempt_summary.csv")
    if not selected_rows:
        print("[plot] no selected results")
        return

    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "axes.linewidth": 1.1,
        }
    )

    items = _load_selected_centerlines()
    fig = plt.figure(figsize=(11.8, 14.8), constrained_layout=True)
    subfigs = fig.subfigures(4, 1, hspace=0.06)
    for idx, (subfig, item) in enumerate(zip(subfigs, items), 1):
        axes_row = subfig.subplots(1, 2, gridspec_kw={"width_ratios": [1.12, 1.0]})
        ax = axes_row[0]
        ax.remove()
        ax = subfig.add_subplot(1, 2, 1, projection="3d")
        ax_err = axes_row[1]
        row = item["row"]
        for pts, label, color, lw, ls, alpha, zorder in [
            (item["fk_back"], "FK回代", "#0072B2", 2.8, "-", 0.58, 2),
            (item["ik"], "IK终态", "#D55E00", 2.3, "--", 0.98, 3),
        ]:
            mm = pts * 1e3
            ax.plot(
                mm[:, 0],
                mm[:, 1],
                mm[:, 2],
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=zorder,
                label=label,
            )
        target = item["fk_back"][-1] * 1e3
        ax.scatter([target[0]], [target[1]], [target[2]], marker="*", s=80, color="#C91F37", label="目标点")
        ax.set_title(f"C{row['active_coil']} 15 A", fontproperties=latin_prop)
        ax.set_xlabel("x (mm)", fontproperties=latin_prop)
        ax.set_ylabel("y (mm)", fontproperties=latin_prop)
        ax.set_zlabel("z (mm)", fontproperties=latin_prop)
        ax.view_init(elev=24, azim=-58)
        ax.grid(True, alpha=0.35)
        ax.set_box_aspect([1.0, 0.75, 0.85])
        if idx == 1:
            ax.legend(frameon=False, prop=cjk_prop, loc="upper left", bbox_to_anchor=(-0.03, 1.02))

        ik_pts = np.asarray(item["ik"], dtype=float)
        fk_pts = np.asarray(item["fk_back"], dtype=float)
        if ik_pts.shape != fk_pts.shape:
            ik_cmp = ik_pts
            fk_cmp = _resample_polyline(fk_pts, n=ik_pts.shape[0])
        else:
            ik_cmp = ik_pts
            fk_cmp = fk_pts
        x_axis = np.arange(ik_cmp.shape[0])
        err_mm = np.linalg.norm(ik_cmp - fk_cmp, axis=1) * 1e3
        ax_err.plot(x_axis, err_mm, color="#009E73", lw=2.4, marker="o", ms=4.2)
        ax_err.fill_between(x_axis, 0.0, err_mm, color="#009E73", alpha=0.13)
        ax_err.set_title("IK-FK对应节点误差", fontproperties=cjk_prop)
        ax_err.set_xlabel("采样节点编号", fontproperties=cjk_prop)
        ax_err.set_ylabel("距离误差 (mm)", fontproperties=cjk_prop)
        ax_err.grid(True, which="major", color="#b7b7b7", alpha=0.30)
        ax_err.grid(True, which="minor", color="#d0d0d0", alpha=0.15)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)
        ax_err.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax_err.text(
            0.98,
            0.88,
            f"max={float(np.max(err_mm)):.2e} mm",
            transform=ax_err.transAxes,
            ha="right",
            va="top",
            fontproperties=latin_prop,
        )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"fig_322_single_segment_fk_ik_fk_pose.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, row in zip(axes.ravel(), selected_rows):
        I_fk = np.asarray(_parse_csv_floats(row["I_FK_A"]), dtype=float)
        I_star = np.asarray(_parse_csv_floats(row["I_star_A"]), dtype=float)
        x = np.arange(1, 9)
        width = 0.36
        ax.bar(x - width / 2, I_fk, width, color="#222222", alpha=0.82, label="FK输入")
        ax.bar(x + width / 2, I_star, width, color="#0072B2", alpha=0.82, label="IK反解")
        ax.axhline(0.0, color="#555555", lw=0.8)
        ax.set_title(f"C{row['active_coil']} 15 A, x={float(row['x_star_m']) * 1e3:.3f} mm", fontproperties=cjk_prop)
        ax.set_xlabel("线圈编号", fontproperties=cjk_prop)
        ax.set_ylabel("电流 (A)", fontproperties=cjk_prop)
        ax.set_xticks(x)
        ax.grid(True, axis="y", color="#b7b7b7", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes.ravel()[0]:
            ax.legend(frameon=False, prop=cjk_prop)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"fig_322_single_segment_control_recovery.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    labels = [f"C{r['active_coil']}" for r in selected_rows]
    x = np.arange(len(labels))
    metrics = [
        ("IK目标误差", "ik_target_err_mm", "#222222", "o"),
        ("FK回代目标误差", "fk_target_err_mm", "#0072B2", "s"),
        ("IK-FK末端差", "ik_fk_tip_mm", "#D55E00", "^"),
        ("中心线RMS", "oracle_fk_centerline_rms_mm", "#009E73", "D"),
    ]
    for name, key, color, marker in metrics:
        y = np.asarray([max(float(r[key]), 1e-5) for r in selected_rows], dtype=float)
        ax.plot(x, y, marker=marker, ms=7, lw=2.0, color=color, label=name)
    ax.axhline(1.0, color="#999999", lw=1.3, ls="--", label="1 mm 判据")
    ax.axhline(0.5, color="#bbbbbb", lw=1.1, ls=":", label="0.5 mm 判据")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=latin_prop)
    ax.set_ylabel("误差 (mm)", fontproperties=cjk_prop)
    ax.set_xlabel("FK目标工况", fontproperties=cjk_prop)
    ax.grid(True, which="major", color="#b7b7b7", alpha=0.30)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.16)
    ax.legend(frameon=False, prop=cjk_prop, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"fig_322_single_segment_roundtrip_metrics.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)

    blind_counts = {}
    for case in {r["case"] for r in all_rows}:
        blind_counts[case] = sum(1 for r in all_rows if r["case"] == case and r["role"] == "blind" and str(r["pass"]) == "True")
    report = {
        "selected": selected_rows,
        "blind_pass_counts": blind_counts,
        "pass_criteria": PASS_LIMITS,
        "font_note": "Chinese uses Source Han Serif CN fallback; Latin uses Liberation Serif fallback.",
    }
    (OUT_DIR / "figure_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_perturbed_figures() -> None:
    import matplotlib.pyplot as plt

    cjk_prop, latin_prop = _font_setup()
    rows_path = OUT_DIR / "perturbed_init_summary.csv"
    if not rows_path.exists():
        print("[plot] no perturbed_init_summary.csv")
        return
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "axes.linewidth": 1.1,
        }
    )
    items = _load_selected_centerlines(rows_path)
    _plot_pose_error_4x2(
        items=items,
        out_prefix="fig_322_single_segment_perturbed_init_fk_ik_fk_pose",
        cjk_prop=cjk_prop,
        latin_prop=latin_prop,
        title_mode="perturbed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-source FK oracle -> single-layer IK -> same-source FK roundtrip.")
    parser.add_argument(
        "--phase",
        choices=["prepare", "oracle", "multistart", "full", "perturb", "bad_init", "summarize", "plot", "plot_perturb"],
        default="oracle",
    )
    parser.add_argument("--force_ik", action="store_true")
    parser.add_argument("--force_fk", action="store_true")
    parser.add_argument("--force_oracle", action="store_true")
    args = parser.parse_args()

    if args.phase == "prepare":
        prepare_oracles(force=bool(args.force_oracle))
    elif args.phase in {"oracle", "multistart", "full"}:
        prepare_oracles(force=bool(args.force_oracle))
        run_suite(args.phase, force_ik=bool(args.force_ik), force_fk=bool(args.force_fk))
    elif args.phase == "perturb":
        prepare_oracles(force=bool(args.force_oracle))
        run_perturbed_init_suite(force_ik=bool(args.force_ik), force_fk=bool(args.force_fk))
    elif args.phase == "bad_init":
        prepare_oracles(force=bool(args.force_oracle))
        run_bad_init_failure_case(force_ik=bool(args.force_ik), force_fk=bool(args.force_fk))
    elif args.phase == "summarize":
        summarize_existing()
    elif args.phase == "plot":
        plot_figures()
    elif args.phase == "plot_perturb":
        plot_perturbed_figures()


if __name__ == "__main__":
    main()
