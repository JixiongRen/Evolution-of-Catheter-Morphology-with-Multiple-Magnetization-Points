from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FK_OPT_DIR = REPO_ROOT / "forward_kinematics_optimized"
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

import jax
import jax.numpy as jnp
import numpy as np

from equilibrium_solver_nondim import residual_bar
from fk_engine import ForwardKinematicsEngine
from utils_nondim import pack_z_bar_jax, unpack_z_bar_jax, x_bar_to_dim_jax, x_dim_to_bar_jax


TARGET_CURRENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
TARGET_M_BODY = (
    (0.0, 0.0, -0.005301),
    (0.0, 0.0, 0.005301),
    (0.005301, 0.0, 0.0),
)


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _parse_factors(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_float_vector(text: str, *, expected: Optional[int] = None) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if expected is not None and len(values) != int(expected):
        raise ValueError(f"Expected {expected} values, got {len(values)} from {text!r}")
    return values


def _parse_int_tuple(text: str, *, expected: Optional[int] = None) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if expected is not None and len(values) != int(expected):
        raise ValueError(f"Expected {expected} values, got {len(values)} from {text!r}")
    return values


def _parse_m_body_list(text: str) -> tuple[tuple[float, float, float], ...]:
    rows = []
    for row in str(text).split(";"):
        row = row.strip()
        if not row:
            continue
        values = _parse_float_vector(row, expected=3)
        rows.append((values[0], values[1], values[2]))
    if not rows:
        raise ValueError("m_body_list must contain at least one 3-vector")
    return tuple(rows)


def _target_currents(args: argparse.Namespace) -> jnp.ndarray:
    return jnp.asarray(_parse_float_vector(args.coil_currents, expected=8), dtype=jnp.float64)


def _target_m_body(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    return _parse_m_body_list(args.m_body_list)


def _target_M_list(args: argparse.Namespace) -> tuple[int, ...]:
    return _parse_int_tuple(args.M_list, expected=3)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _get_calib_file() -> tuple[str, str]:
    requested = Path("/abs/path/calibration.json")
    if requested.exists():
        return str(requested), "requested_path_exists"
    fallback = REPO_ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    return str(fallback), "requested_path_missing_used_repo_fallback"


def _build_engine(
    args: argparse.Namespace,
    *,
    max_iter: int,
    tol: float,
    M_list: Optional[tuple[int, ...]] = None,
) -> ForwardKinematicsEngine:
    calib_file, _calib_reason = _get_calib_file()
    m_list = _target_M_list(args) if M_list is None else tuple(int(v) for v in M_list)
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=m_list,
        L1_min=1e-6,
        flex_d_outer=(0.0015,),
        flex_E=(1.8e6,),
        flex_G=(0.6e6,),
        flex_rho=(970.0,),
        rigid_d_outer=(0.0015,),
        rigid_rho=(7500.0,),
        p0_dim=(0.0, 0.0, 0.0),
        Q0_wxyz=(0.70710678, 0.0, 0.70710678, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        enable_gravity=True,
        enable_magnetics=True,
        calib_file=calib_file,
        actuation_table_pkl=str(
            REPO_ROOT
            / "supiee_auto_diff"
            / "offline_interpolation_data"
            / "actuation_tables"
            / "actuation_table.pkl"
        ),
        m_body_list=_target_m_body(args),
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=float(args.lm_damping),
        jac_method=str(args.jac_method),
        linear_solver="dense",
        damping_mode=str(args.damping_mode),
        dense_solve_method=str(args.dense_solve_method),
        step_norm_clip=None if args.step_norm_clip is None else float(args.step_norm_clip),
        backtrack_max=int(args.backtrack_max),
        backtrack_factor=float(args.backtrack_factor),
        trust_region_eta=args.trust_region_eta,
        max_damping_attempts=int(args.max_damping_attempts),
        reject_lam_gain=float(args.reject_lam_gain),
        scale_mode=str(args.scale_mode),
        scale_include_external=bool(args.scale_include_external),
        L_protrude_max=0.11,
    )


def _extract_centerline(z_bar: Any, params: Any, *, rigid_sample_count: int) -> np.ndarray:
    x_nodes_list, _k_list, x_rigid_list = unpack_z_bar_jax(jnp.asarray(z_bar, dtype=jnp.float64), M_list=params.M_list)
    points: list[np.ndarray] = []
    for seg_idx, x_nodes_bar in enumerate(x_nodes_list):
        node_pts = np.asarray(jax.device_get(x_nodes_bar[:, 0:3] * float(params.scales.L_ref)), dtype=np.float64)
        if node_pts.size:
            points.append(node_pts)

        sample_count = max(int(rigid_sample_count), 0)
        prox = np.asarray(jax.device_get(x_nodes_bar[-1, 0:3] * float(params.scales.L_ref)), dtype=np.float64)
        dist = np.asarray(jax.device_get(x_rigid_list[seg_idx][0:3] * float(params.scales.L_ref)), dtype=np.float64)
        alphas = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)[1:]
        points.append(prox[None, :] + alphas[:, None] * (dist - prox)[None, :])

    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(points, axis=0)


def _pointwise_metrics(ref: np.ndarray, cand: np.ndarray, *, L_ref: float) -> dict[str, Any]:
    n = min(int(ref.shape[0]), int(cand.shape[0]))
    if n <= 0:
        return {"status": "missing_centerline"}
    ref_n = np.asarray(ref[:n], dtype=np.float64)
    cand_n = np.asarray(cand[:n], dtype=np.float64)
    delta = np.linalg.norm(cand_n - ref_n, axis=1)
    return {
        "status": "ok",
        "point_count": n,
        "max_point_error_m": float(np.max(delta)),
        "rms_point_error_m": float(np.sqrt(np.mean(delta * delta))),
        "mean_point_error_m": float(np.mean(delta)),
        "end_error_m": float(delta[-1]),
        "max_rel_to_L_ref": float(np.max(delta) / max(float(L_ref), 1e-12)),
        "end_rel_to_L_ref": float(delta[-1] / max(float(L_ref), 1e-12)),
    }


def _load_reference_metrics(centerline: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for label, path_text in (
        ("current_new_cli_tol1e-5", args.reference_new_centerline),
        ("legacy_old_cli_tol1e-5", args.reference_old_centerline),
    ):
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            refs[label] = {"status": "missing", "path": str(path)}
            continue
        ref = np.load(path)
        refs[label] = {
            "path": str(path),
            **_pointwise_metrics(ref, centerline, L_ref=0.11),
        }
    return refs


def _plot_centerline(centerline: np.ndarray, out_dir: Path, refs: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="tab:green", linewidth=2.5, label="candidate")
    ax.scatter(centerline[0, 0], centerline[0, 1], centerline[0, 2], color="black", s=35, label="start")
    ax.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2], color="tab:green", s=45, label="end")

    for label, info in refs.items():
        path = info.get("path")
        if not path or not Path(path).exists():
            continue
        ref = np.load(path)
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], linewidth=1.4, linestyle="--", label=label)

    ax.set_title("Fixed FK case centerline")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "centerline_overlay.png", dpi=180)
    plt.close(fig)


def _write_stage_csv(out_dir: Path, stages: Iterable[dict[str, Any]]) -> None:
    rows = list(stages)
    if not rows:
        return
    fields = [
        "stage",
        "current_factor",
        "wall_s",
        "ok",
        "stop_reason",
        "n_iter",
        "normE",
        "max_abs_E",
        "max_iter",
        "tol",
    ]
    with (out_dir / "stages.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _interp_rows(old_s: np.ndarray, old_values: np.ndarray, new_s: np.ndarray) -> np.ndarray:
    values = np.asarray(old_values, dtype=np.float64)
    out = np.zeros((int(new_s.size), int(values.shape[1])), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(new_s, old_s, values[:, col])
    return out


def _normalize_quaternion_columns(states: np.ndarray) -> np.ndarray:
    out = np.asarray(states, dtype=np.float64).copy()
    q = out[:, 3:7]
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    out[:, 3:7] = q / np.maximum(q_norm, 1e-12)
    return out


def _lift_z_bar(
    z_bar: Any,
    *,
    old_params: Any,
    new_params: Any,
    old_M_list: tuple[int, ...],
    new_M_list: tuple[int, ...],
) -> jnp.ndarray:
    """Interpolate a coarse packed FK state to a finer mesh."""
    old_nodes, old_k, old_rigid = unpack_z_bar_jax(jnp.asarray(z_bar, dtype=jnp.float64), M_list=old_M_list)
    new_nodes: list[jnp.ndarray] = []
    new_k: list[jnp.ndarray] = []
    new_rigid: list[jnp.ndarray] = []

    for seg_idx, (m_old, m_new) in enumerate(zip(old_M_list, new_M_list)):
        old_s_nodes = np.linspace(0.0, 1.0, int(m_old) + 1, dtype=np.float64)
        new_s_nodes = np.linspace(0.0, 1.0, int(m_new) + 1, dtype=np.float64)
        old_node_dim = np.asarray(
            jax.device_get(
                jax.vmap(lambda x: x_bar_to_dim_jax(x, old_params.scales))(old_nodes[seg_idx])
            ),
            dtype=np.float64,
        )
        node_dim = _normalize_quaternion_columns(_interp_rows(old_s_nodes, old_node_dim, new_s_nodes))
        node_bar = jax.vmap(lambda x: x_dim_to_bar_jax(x, new_params.scales))(jnp.asarray(node_dim, dtype=jnp.float64))
        new_nodes.append(node_bar)

        old_mid = (np.arange(int(m_old), dtype=np.float64) + 0.5) / max(int(m_old), 1)
        new_mid = (np.arange(int(m_new), dtype=np.float64) + 0.5) / max(int(m_new), 1)
        old_k_np = np.asarray(jax.device_get(old_k[seg_idx]), dtype=np.float64).reshape((int(m_old), 39))
        new_k.append(jnp.asarray(_interp_rows(old_mid, old_k_np, new_mid), dtype=jnp.float64))

        rigid_dim = x_bar_to_dim_jax(old_rigid[seg_idx], old_params.scales)
        new_rigid.append(x_dim_to_bar_jax(rigid_dim, new_params.scales))

    return pack_z_bar_jax(new_nodes, new_k, new_rigid)


def _solve_once(
    engine: ForwardKinematicsEngine,
    *,
    currents: jnp.ndarray,
    L_protrude: float,
    warm_start: bool,
    stage_name: str,
    current_factor: float,
) -> tuple[Any, Any, Any, bool, Any, dict[str, Any]]:
    start = time.perf_counter()
    z_star, params, meshes, ok, stats = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=float(L_protrude),
        warm_start=bool(warm_start),
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    _block(z_star)
    wall_s = time.perf_counter() - start
    E = residual_bar(z_star, params)
    _block(E)
    row = {
        "stage": str(stage_name),
        "current_factor": float(current_factor),
        "wall_s": float(wall_s),
        "ok": bool(ok and float(jnp.linalg.norm(E)) < float(engine.tol)),
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "normE": float(jnp.linalg.norm(E)),
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "max_iter": int(engine.max_iter),
        "tol": float(engine.tol),
    }
    return z_star, params, meshes, bool(ok), stats, row


def run_direct(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    engine = _build_engine(args, max_iter=int(args.max_iter), tol=float(args.tol), M_list=_target_M_list(args))
    target = _target_currents(args)
    warmup = engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}
    z_star, params, _meshes, ok, stats, row = _solve_once(
        engine,
        currents=target,
        L_protrude=0.11,
        warm_start=False,
        stage_name="direct",
        current_factor=1.0,
    )
    return _finalize_result(args, out_dir, z_star, params, ok, stats, [row], warmup)


def run_continuation(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    target = _target_currents(args)
    engine = _build_engine(args, max_iter=int(args.stage_max_iter), tol=float(args.stage_tol), M_list=_target_M_list(args))

    # Compile and initialize static scale using target current, not the zero-current stage.
    warmup = engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}
    stages: list[dict[str, Any]] = []
    z_star = params = stats = None
    ok = False

    for idx, factor in enumerate(_parse_factors(args.current_factors)):
        engine.max_iter = int(args.stage_max_iter)
        engine.tol = float(args.stage_tol)
        z_star, params, _meshes, ok, stats, row = _solve_once(
            engine,
            currents=target * float(factor),
            L_protrude=0.11,
            warm_start=(idx > 0),
            stage_name=f"continuation_{idx:02d}",
            current_factor=float(factor),
        )
        stages.append(row)

    engine.max_iter = int(args.final_max_iter)
    engine.tol = float(args.final_tol)
    z_star, params, _meshes, ok, stats, row = _solve_once(
        engine,
        currents=target,
        L_protrude=0.11,
        warm_start=True,
        stage_name="final_refine",
        current_factor=1.0,
    )
    stages.append(row)
    if z_star is None or params is None:
        raise RuntimeError("Continuation did not produce a solution.")
    return _finalize_result(args, out_dir, z_star, params, ok, stats, stages, warmup)


def run_coarse_to_fine(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    target = jnp.asarray(TARGET_CURRENTS, dtype=jnp.float64)
    coarse_M = tuple(int(v) for v in str(args.coarse_M_list).split(","))
    fine_M = _target_M_list(args)
    if len(coarse_M) != 3:
        raise ValueError("coarse_M_list must contain three integers")

    coarse_engine = _build_engine(args, max_iter=int(args.stage_max_iter), tol=float(args.stage_tol), M_list=coarse_M)
    coarse_warmup = coarse_engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}
    stages: list[dict[str, Any]] = []
    coarse_z = coarse_params = None
    ok = False
    stats = None

    for idx, factor in enumerate(_parse_factors(args.current_factors)):
        coarse_engine.max_iter = int(args.stage_max_iter)
        coarse_engine.tol = float(args.stage_tol)
        coarse_z, coarse_params, _meshes, ok, stats, row = _solve_once(
            coarse_engine,
            currents=target * float(factor),
            L_protrude=0.11,
            warm_start=(idx > 0),
            stage_name=f"coarse_continuation_{idx:02d}",
            current_factor=float(factor),
        )
        stages.append(row)

    if coarse_z is None or coarse_params is None:
        raise RuntimeError("Coarse solve did not produce a state.")

    fine_engine = _build_engine(args, max_iter=int(args.final_max_iter), tol=float(args.final_tol), M_list=fine_M)
    fine_warmup = fine_engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}
    fine_engine._ensure_static_initialized(L_protrude=0.11, coil_currents=target)
    if fine_engine._params_static_base is None:
        raise RuntimeError("Fine engine static params were not initialized.")
    fine_params0 = fine_engine._params_static_base
    z0_fine = _lift_z_bar(
        coarse_z,
        old_params=coarse_params,
        new_params=fine_params0,
        old_M_list=coarse_M,
        new_M_list=fine_M,
    )

    fine_engine.max_iter = int(args.final_max_iter)
    fine_engine.tol = float(args.final_tol)
    start = time.perf_counter()
    z_star, params, _meshes, ok, stats = fine_engine.solve_with_stats(
        coil_currents=target,
        L_protrude=0.11,
        warm_start=False,
        override_z0_bar=z0_fine,
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    _block(z_star)
    wall_s = time.perf_counter() - start
    E = residual_bar(z_star, params)
    _block(E)
    fine_normE = float(jnp.linalg.norm(E))
    stages.append(
        {
            "stage": "fine_refine",
            "current_factor": 1.0,
            "wall_s": float(wall_s),
            "ok": bool(ok and fine_normE < float(fine_engine.tol)),
            "stop_reason": None if stats is None else str(stats.stop_reason),
            "n_iter": None if stats is None else int(stats.n_iter),
            "normE": fine_normE,
            "max_abs_E": float(jnp.max(jnp.abs(E))),
            "max_iter": int(fine_engine.max_iter),
            "tol": float(fine_engine.tol),
        }
    )

    warmup = {
        "coarse": coarse_warmup,
        "fine": fine_warmup,
        "total_s": float(coarse_warmup.get("total_s", 0.0) + fine_warmup.get("total_s", 0.0)),
    }
    return _finalize_result(args, out_dir, z_star, params, bool(ok), stats, stages, warmup)


def run_scipy_trust_region(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Run a SciPy trust-region least-squares solve on the fixed FK residual."""
    from scipy.optimize import least_squares
    from scipy import sparse

    target = _target_currents(args)
    engine = _build_engine(args, max_iter=int(args.stage_max_iter), tol=float(args.stage_tol), M_list=_target_M_list(args))
    warmup = engine.warm_up(coil_currents=target, L_protrude=0.11) if bool(args.warmup) else {}
    stages: list[dict[str, Any]] = []

    if str(args.scipy_seed_mode) == "continuation":
        z_seed = params = stats = None
        ok = False
        for idx, factor in enumerate(_parse_factors(args.current_factors)):
            engine.max_iter = int(args.stage_max_iter)
            engine.tol = float(args.stage_tol)
            z_seed, params, _meshes, ok, stats, row = _solve_once(
                engine,
                currents=target * float(factor),
                L_protrude=0.11,
                warm_start=(idx > 0),
                stage_name=f"seed_continuation_{idx:02d}",
                current_factor=float(factor),
            )
            stages.append(row)
        if z_seed is None or params is None:
            raise RuntimeError("Continuation seed did not produce a state.")
        if int(args.scipy_seed_final_lm_iter) > 0:
            engine.max_iter = int(args.scipy_seed_final_lm_iter)
            engine.tol = float(args.tol)
            z_seed, params, _meshes, ok, stats, row = _solve_once(
                engine,
                currents=target,
                L_protrude=0.11,
                warm_start=True,
                stage_name="seed_final_lm",
                current_factor=1.0,
            )
            stages.append(row)
    else:
        old_max_iter = int(engine.max_iter)
        engine.max_iter = 0
        z_seed, params, _meshes, ok, stats, row = _solve_once(
            engine,
            currents=target,
            L_protrude=0.11,
            warm_start=False,
            stage_name="seed_initial",
            current_factor=1.0,
        )
        engine.max_iter = old_max_iter
        stages.append(row)

    if params is None:
        raise RuntimeError("Target params are unavailable.")

    z0_np = np.asarray(jax.device_get(z_seed), dtype=np.float64).reshape(-1)
    profile = {
        "fun_calls": 0,
        "jac_calls": 0,
        "fun_s": 0.0,
        "jac_s": 0.0,
        "last_normE": None,
        "first_jac_sparsity": None,
    }
    history: list[dict[str, Any]] = []

    def fun(z_np: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        E = engine._solver_cached.residual_jit(jnp.asarray(z_np, dtype=jnp.float64), params)
        _block(E)
        arr = np.asarray(jax.device_get(E), dtype=np.float64)
        elapsed = time.perf_counter() - t0
        profile["fun_calls"] += 1
        profile["fun_s"] += float(elapsed)
        profile["last_normE"] = float(np.linalg.norm(arr))
        return arr

    def jac(z_np: np.ndarray) -> Any:
        t0 = time.perf_counter()
        J = engine._solver_cached.jacobian_jit(
            jnp.asarray(z_np, dtype=jnp.float64),
            params,
            method=str(args.jac_method),
        )
        _block(J)
        arr = np.asarray(jax.device_get(J), dtype=np.float64)
        elapsed = time.perf_counter() - t0
        profile["jac_calls"] += 1
        profile["jac_s"] += float(elapsed)
        if profile["first_jac_sparsity"] is None:
            nnz = int(np.count_nonzero(arr))
            profile["first_jac_sparsity"] = {
                "shape": list(arr.shape),
                "nnz": nnz,
                "density": float(nnz / max(arr.size, 1)),
            }
        if str(args.scipy_jac_format) == "csr_exact":
            return sparse.csr_matrix(arr)
        if str(args.scipy_jac_format) == "csr_threshold":
            arr[np.abs(arr) < float(args.scipy_jac_threshold)] = 0.0
            return sparse.csr_matrix(arr)
        return arr

    def callback(x: np.ndarray, *callback_args: Any, **callback_kwargs: Any) -> None:
        normE = profile["last_normE"]
        history.append(
            {
                "callback_index": len(history),
                "normE": None if normE is None else float(normE),
                "fun_calls": int(profile["fun_calls"]),
                "jac_calls": int(profile["jac_calls"]),
            }
        )
        if normE is not None and float(normE) < float(args.tol):
            raise StopIteration

    tr_solver = None if str(args.scipy_tr_solver) == "none" else str(args.scipy_tr_solver)
    x_scale = str(args.scipy_x_scale)
    scipy_start = time.perf_counter()
    result = least_squares(
        fun,
        z0_np,
        jac=jac,
        method=str(args.scipy_method),
        ftol=float(args.scipy_ftol),
        xtol=float(args.scipy_xtol),
        gtol=float(args.scipy_gtol),
        x_scale=x_scale,
        tr_solver=tr_solver,
        tr_options={"regularize": bool(args.scipy_regularize)} if tr_solver == "lsmr" else None,
        max_nfev=int(args.scipy_max_nfev),
        verbose=0,
        callback=callback,
    )
    scipy_wall = time.perf_counter() - scipy_start

    z_star = jnp.asarray(result.x, dtype=jnp.float64)
    E_final = residual_bar(z_star, params)
    _block(E_final)
    final_norm = float(jnp.linalg.norm(E_final))
    ok_final = bool(final_norm < float(args.tol))
    stop_reason = "strict_converged" if ok_final else f"scipy_status_{int(result.status)}"
    stats_final = SimpleNamespace(
        stop_reason=stop_reason,
        n_iter=int(getattr(result, "njev", 0) or getattr(result, "nfev", 0) or 0),
        lam=float("nan"),
    )
    stages.append(
        {
            "stage": "scipy_trust_region_refine",
            "current_factor": 1.0,
            "wall_s": float(scipy_wall),
            "ok": bool(ok_final),
            "stop_reason": stop_reason,
            "n_iter": int(stats_final.n_iter),
            "normE": final_norm,
            "max_abs_E": float(jnp.max(jnp.abs(E_final))),
            "max_iter": int(args.scipy_max_nfev),
            "tol": float(args.tol),
        }
    )
    (out_dir / "scipy_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "scipy_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "scipy_result.json").write_text(
        json.dumps(
            {
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "final_normE": final_norm,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return _finalize_result(args, out_dir, z_star, params, ok_final, stats_final, stages, warmup)


def _finalize_result(
    args: argparse.Namespace,
    out_dir: Path,
    z_star: Any,
    params: Any,
    ok: bool,
    stats: Any,
    stages: list[dict[str, Any]],
    warmup: dict[str, Any],
) -> dict[str, Any]:
    E = residual_bar(z_star, params)
    _block(E)
    final_normE = float(jnp.linalg.norm(E))
    ok_strict = bool(ok and final_normE < float(args.tol))
    centerline = _extract_centerline(z_star, params, rigid_sample_count=int(args.rigid_sample_count))
    refs = _load_reference_metrics(centerline, args)
    _plot_centerline(centerline, out_dir, refs)
    _write_stage_csv(out_dir, stages)

    np.save(out_dir / "centerline.npy", centerline)
    np.savez_compressed(
        out_dir / "solution.npz",
        z_bar=np.asarray(jax.device_get(z_star), dtype=np.float64),
        E=np.asarray(jax.device_get(E), dtype=np.float64),
        centerline=centerline,
    )

    stage_wall_s = float(sum(float(row["wall_s"]) for row in stages))
    first_compile_s = float(warmup.get("total_s", 0.0)) if warmup else 0.0
    summary = {
        "case_name": str(args.case_name),
        "mode": str(args.mode),
        "classification": str(args.classification),
        "ok": ok_strict,
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "final_lam": None if stats is None else float(stats.lam),
        "wall_s": stage_wall_s,
        "first_call_compile_s": first_compile_s,
        "steady_solve_s": stage_wall_s,
        "wall_s_including_warmup": float(stage_wall_s + first_compile_s),
        "final_normE": final_normE,
        "max_abs_E": float(jnp.max(jnp.abs(E))),
        "centerline_shape": list(centerline.shape),
        "centerline_start_m": centerline[0].tolist() if centerline.size else None,
        "centerline_end_m": centerline[-1].tolist() if centerline.size else None,
        "scale": {
            "mode": str(args.scale_mode),
            "include_external": bool(args.scale_include_external),
            "L_ref": float(params.scales.L_ref),
            "F_ref": float(params.scales.F_ref),
            "M_ref": float(params.scales.M_ref),
        },
        "warmup": warmup,
        "stages": stages,
        "references": refs,
        "rss_mb": _rss_mb(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-case FK benchmark driver for under-60s optimization.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case_name", default="fk_under_60s_candidate")
    parser.add_argument("--mode", choices=["direct", "continuation", "coarse_to_fine", "scipy_trust_region"], default="direct")
    parser.add_argument("--classification", default="speed-only result")
    parser.add_argument("--coil_currents", default=",".join(str(v) for v in TARGET_CURRENTS))
    parser.add_argument(
        "--m_body_list",
        default="; ".join(",".join(str(v) for v in row) for row in TARGET_M_BODY),
    )
    parser.add_argument("--M_list", default="5,5,5")
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lm_damping", type=float, default=1e-1)
    parser.add_argument("--jac_method", choices=["fwd", "rev"], default="fwd")
    parser.add_argument("--damping_mode", choices=["identity", "diagH"], default="identity")
    parser.add_argument("--dense_solve_method", choices=["solve", "pos", "cholesky"], default="solve")
    parser.add_argument("--step_norm_clip", type=float, default=None)
    parser.add_argument("--backtrack_max", type=int, default=0)
    parser.add_argument("--backtrack_factor", type=float, default=0.5)
    parser.add_argument("--trust_region_eta", type=float, default=None)
    parser.add_argument("--max_damping_attempts", type=int, default=1)
    parser.add_argument("--reject_lam_gain", type=float, default=10.0)
    parser.add_argument("--scale_mode", choices=["optimized", "legacy"], default="optimized")
    parser.add_argument("--scale_include_external", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--current_factors", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--coarse_M_list", default="2,2,2")
    parser.add_argument("--stage_max_iter", type=int, default=300)
    parser.add_argument("--stage_tol", type=float, default=1e-4)
    parser.add_argument("--final_max_iter", type=int, default=1000)
    parser.add_argument("--final_tol", type=float, default=1e-5)
    parser.add_argument("--rigid_sample_count", type=int, default=10)
    parser.add_argument("--scipy_seed_mode", choices=["initial", "continuation"], default="continuation")
    parser.add_argument("--scipy_seed_final_lm_iter", type=int, default=0)
    parser.add_argument("--scipy_method", choices=["trf", "dogbox", "lm"], default="trf")
    parser.add_argument("--scipy_tr_solver", choices=["none", "exact", "lsmr"], default="exact")
    parser.add_argument("--scipy_jac_format", choices=["dense", "csr_exact", "csr_threshold"], default="dense")
    parser.add_argument("--scipy_jac_threshold", type=float, default=1e-12)
    parser.add_argument("--scipy_x_scale", default="jac")
    parser.add_argument("--scipy_ftol", type=float, default=1e-12)
    parser.add_argument("--scipy_xtol", type=float, default=1e-12)
    parser.add_argument("--scipy_gtol", type=float, default=1e-12)
    parser.add_argument("--scipy_max_nfev", type=int, default=200)
    parser.add_argument("--scipy_regularize", action="store_true")
    parser.add_argument(
        "--reference_new_centerline",
        default=str(
            REPO_ROOT
            / "docs"
            / "efficiency_optimization_experiment"
            / "eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data"
            / "new_fk_tol1e-5"
            / "centerline_last_callback.npy"
        ),
    )
    parser.add_argument(
        "--reference_old_centerline",
        default=str(
            REPO_ROOT
            / "docs"
            / "efficiency_optimization_experiment"
            / "eff_opt_exp_20260523192724_fk_old_new_same_params_compare_logs_data"
            / "old_fk_tol1e-5"
            / "centerline_last_callback.npy"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_file, calib_reason = _get_calib_file()
    target_currents = _parse_float_vector(args.coil_currents, expected=8)
    target_m_body = _target_m_body(args)
    target_M_list = _target_M_list(args)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "git_commit": _repo_commit(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "calib_file": calib_file,
        "calib_reason": calib_reason,
        "fixed_case": {
            "flex_lengths_tail": [0.03, 0.03],
            "rigid_lengths": [0.003, 0.003, 0.003],
            "L_protrude": 0.11,
            "M_list": list(target_M_list),
            "enable_gravity": True,
            "enable_magnetics": True,
            "coil_currents": list(target_currents),
            "m_body_list": [list(item) for item in target_m_body],
            "p0": [0.0, 0.0, 0.0],
            "Q0": [0.70710678, 0.0, 0.70710678, 0.0],
            "axis_body": [0.0, 0.0, 1.0],
        },
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    if args.mode == "direct":
        summary = run_direct(args, out_dir)
    elif args.mode == "continuation":
        summary = run_continuation(args, out_dir)
    elif args.mode == "coarse_to_fine":
        summary = run_coarse_to_fine(args, out_dir)
    elif args.mode == "scipy_trust_region":
        summary = run_scipy_trust_region(args, out_dir)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    summary["process_wall_s"] = float(time.perf_counter() - start)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
