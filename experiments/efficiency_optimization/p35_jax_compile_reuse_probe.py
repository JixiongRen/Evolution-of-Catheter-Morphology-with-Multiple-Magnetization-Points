from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

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
from single_layer_ik.joint_problem import extract_centerline_p_dim


def _build_engine(*, max_iter: int, tol: float, enable_magnetics: bool = False) -> ForwardKinematicsEngine:
    m_body_list = ((0.0, 0.0, -0.005301), (0.0, 0.0, 0.005301), (0.005301, 0.0, 0.0))
    return ForwardKinematicsEngine(
        flex_lengths_tail=(0.03, 0.03),
        rigid_lengths=(0.003, 0.003, 0.003),
        M_list=(3, 2, 2),
        L1_min=1e-6,
        p0_dim=(0.0, 0.0, -0.05),
        Q0_wxyz=(1.0, 0.0, 0.0, 0.0),
        axis_body=(1.0, 0.0, 0.0),
        enable_gravity=False,
        enable_magnetics=bool(enable_magnetics),
        calib_file=None,
        actuation_table_pkl="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl",
        m_body_list=m_body_list if enable_magnetics else None,
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=1e-1,
        jac_method="fwd",
        linear_solver="dense",
        L_protrude_max=0.18,
    )


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _to_float_list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1).tolist()]


def _candidate_points(z_bar: jnp.ndarray, engine: ForwardKinematicsEngine, params: Any) -> dict[str, Any]:
    x_nodes_list, _k_list, x_rigid_list = unpack_z_bar_jax(z_bar, M_list=engine.M_list)
    flex_last = x_bar_to_dim(x_nodes_list[-1][-1], params.scales)
    rigid_last = x_bar_to_dim(x_rigid_list[-1], params.scales)
    return {
        "last_flex_node_m": _to_float_list(flex_last[0:3]),
        "last_rigid_state_m": _to_float_list(rigid_last[0:3]),
    }


def _run_solve(
    engine: ForwardKinematicsEngine,
    *,
    label: str,
    currents: jnp.ndarray,
    L_protrude: float,
    warm_start: bool,
    stats_include_jacobian: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    z_star, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=currents,
        L_protrude=float(L_protrude),
        warm_start=bool(warm_start),
        return_stats=True,
        stats_include_jacobian=bool(stats_include_jacobian),
        verbose=False,
    )
    _block(z_star)
    wall_s = time.perf_counter() - start
    E = residual_bar(z_star, params)
    _block(E)
    centerline = extract_centerline_p_dim(
        z_bar=z_star,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=2,
    )
    centerline_np = np.asarray(jax.device_get(centerline), dtype=np.float64)
    return {
        "label": str(label),
        "wall_s": float(wall_s),
        "ok": bool(ok),
        "stop_reason": None if stats is None else str(stats.stop_reason),
        "n_iter": None if stats is None else int(stats.n_iter),
        "normE": float(jnp.linalg.norm(E)),
        "stats_has_jacobian": bool(stats is not None and getattr(stats, "J", None) is not None),
        "centerline_end_m": centerline_np[-1].tolist() if centerline_np.size else None,
        "points": _candidate_points(z_star, engine, params),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Probe JAX compile reuse across repeated FK calls and engine rebuilds.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_iter", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--rebuild_count", type=int, default=2)
    parser.add_argument("--enable_magnetics", action="store_true")
    parser.add_argument("--stats_include_jacobian", action="store_true")
    parser.add_argument("--warmup_first", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=False, text=True, capture_output=True).stdout.strip(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
        "max_iter": int(args.max_iter),
        "rebuild_count": int(args.rebuild_count),
        "module_level_jit_wrappers": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    currents = jnp.asarray((10.0, -8.0, 6.0, -4.0, 2.0, -6.0, 8.0, -10.0), dtype=jnp.float64)
    rows: list[dict[str, Any]] = []

    engine = _build_engine(max_iter=int(args.max_iter), tol=float(args.tol), enable_magnetics=bool(args.enable_magnetics))
    if bool(args.warmup_first):
        rows.append({"label": "engine0_warmup", "warmup": engine.warm_up(coil_currents=currents, L_protrude=0.10)})
    rows.append(
        _run_solve(
            engine,
            label="engine0_first_solve",
            currents=currents,
            L_protrude=0.10,
            warm_start=False,
            stats_include_jacobian=bool(args.stats_include_jacobian),
        )
    )
    rows.append(
        _run_solve(
            engine,
            label="engine0_repeat_solve",
            currents=currents,
            L_protrude=0.10,
            warm_start=False,
            stats_include_jacobian=bool(args.stats_include_jacobian),
        )
    )

    for i in range(int(args.rebuild_count)):
        start = time.perf_counter()
        rebuilt = _build_engine(max_iter=int(args.max_iter), tol=float(args.tol), enable_magnetics=bool(args.enable_magnetics))
        build_s = time.perf_counter() - start
        row = _run_solve(
            rebuilt,
            label=f"engine_rebuild_{i + 1}_solve",
            currents=currents,
            L_protrude=0.10,
            warm_start=False,
            stats_include_jacobian=bool(args.stats_include_jacobian),
        )
        row["engine_build_s"] = float(build_s)
        rows.append(row)
        (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
