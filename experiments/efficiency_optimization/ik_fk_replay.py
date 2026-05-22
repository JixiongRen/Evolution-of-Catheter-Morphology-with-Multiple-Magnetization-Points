from __future__ import annotations

import argparse
import json
import os
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


def _parse_csv_floats(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(v) for v in str(value).replace(";", ",").split(",") if str(v).strip()]


def _parse_vec3(value: Any) -> tuple[float, float, float]:
    vals = _parse_csv_floats(value)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {len(vals)} from {value!r}")
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def _parse_list_vec3(value: Optional[str]) -> Optional[list[list[float]]]:
    if value is None or str(value).strip() == "":
        return None
    out: list[list[float]] = []
    for item in str(value).split(";"):
        if not item.strip():
            continue
        vals = _parse_csv_floats(item)
        if len(vals) != 3:
            raise ValueError(f"Invalid vec3 entry: {item!r}")
        out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_last_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def _to_list(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1).tolist()]


def _candidate_points(z_bar: jnp.ndarray, engine: ForwardKinematicsEngine, params: Any) -> dict[str, Any]:
    x_nodes_list, _k_list, x_rigid_list = unpack_z_bar_jax(z_bar, M_list=engine.M_list)
    flex_last = x_bar_to_dim(x_nodes_list[-1][-1], params.scales)
    rigid_last = x_bar_to_dim(x_rigid_list[-1], params.scales)
    return {
        "last_flex_node_m": _to_list(flex_last[0:3]),
        "last_rigid_state_m": _to_list(rigid_last[0:3]),
        "last_flex_quat_wxyz": _to_list(flex_last[3:7]),
        "last_rigid_quat_wxyz": _to_list(rigid_last[3:7]),
    }


def _norm_delta(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return None
    return float(np.linalg.norm(aa - bb))


def replay_run(run_dir: Path, *, max_iter: int, tol: float, lm_damping: float) -> dict[str, Any]:
    result = _read_json(run_dir / "result.json")
    config = _read_json(run_dir / "run_config.json")
    trace_last = _read_last_trace(run_dir / "trace.jsonl")
    argv = config.get("argv", {})

    engine = ForwardKinematicsEngine(
        flex_lengths_tail=_parse_csv_floats(argv.get("flex_lengths_tail", "0.03,0.03")),
        rigid_lengths=_parse_csv_floats(argv.get("rigid_lengths", "0.003,0.003,0.003")),
        M_list=[int(v) for v in _parse_csv_floats(argv.get("M_list", "3,2,2"))],
        L1_min=float(argv.get("L1_min", 1e-6)),
        p0_dim=_parse_vec3(argv.get("p0_dim", "0,0,-0.05")),
        Q0_wxyz=tuple(_parse_csv_floats(argv.get("Q0_wxyz", "1,0,0,0"))),
        axis_body=_parse_vec3(argv.get("axis_body", "1,0,0")),
        enable_gravity=bool(argv.get("enable_gravity", False)),
        g_world=_parse_vec3(argv.get("g_world", "0,0,-9.81")),
        enable_magnetics=bool(argv.get("enable_magnetics", False)),
        calib_file=argv.get("calib_file"),
        actuation_table_pkl=argv.get("actuation_table_pkl"),
        m_body_list=_parse_list_vec3(argv.get("m_body_list")),
        max_iter=int(max_iter),
        tol=float(tol),
        lm_damping=float(lm_damping),
        jac_method=str(argv.get("jac_method", "fwd")),
        L_protrude_max=float(argv.get("L_protrude_max", 0.18)),
    )

    I = jnp.asarray(result["I_A"], dtype=jnp.float64).reshape(-1)
    x = float(result["x_m"])
    start = time.perf_counter()
    z_star, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=x,
        warm_start=False,
        return_stats=True,
        stats_include_jacobian=False,
        verbose=False,
    )
    if hasattr(z_star, "block_until_ready"):
        z_star.block_until_ready()
    wall_s = time.perf_counter() - start

    E = residual_bar(z_star, params)
    if hasattr(E, "block_until_ready"):
        E.block_until_ready()
    obs = engine.query_sites(z_bar=z_star, scales=params.scales)
    centerline = extract_centerline_p_dim(
        z_bar=z_star,
        scales=params.scales,
        M_list=params.M_list,
        include_rigid=True,
        rigid_sample_count=int(argv.get("centerline_rigid_samples", 2)),
    )
    centerline_np = np.asarray(jax.device_get(centerline), dtype=np.float64)

    ik_tip = result.get("diag", {}).get("p_tip_m")
    trace_tip = trace_last.get("p_tip_m")
    trace_centerline_end = trace_last.get("p_centerline_end_m")
    fk_tip = _to_list(obs["tip_p_dim"])
    fk_centerline_end = centerline_np[-1].tolist() if centerline_np.size else None

    return {
        "run_dir": str(run_dir),
        "success": bool(result.get("success", False)),
        "stop_reason": result.get("stop_reason"),
        "I_A": [float(v) for v in np.asarray(jax.device_get(I), dtype=np.float64).reshape(-1).tolist()],
        "x_m": float(x),
        "p_des_m": result.get("p_des_m"),
        "ik_reported_tip_m": ik_tip,
        "trace_tip_m": trace_tip,
        "trace_centerline_end_m": trace_centerline_end,
        "fk_replay_tip_m": fk_tip,
        "fk_replay_centerline_end_m": fk_centerline_end,
        "ik_tip_delta_m": _norm_delta(ik_tip, fk_tip),
        "trace_tip_delta_m": _norm_delta(trace_tip, fk_tip),
        "trace_centerline_end_delta_m": _norm_delta(trace_centerline_end, fk_centerline_end),
        "fk_ok": bool(ok),
        "fk_stop_reason": None if stats is None else str(stats.stop_reason),
        "fk_n_iter": None if stats is None else int(stats.n_iter),
        "fk_normE": float(jnp.linalg.norm(E)),
        "fk_wall_s": float(wall_s),
        "candidate_points": _candidate_points(z_star, engine, params),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay an existing IK output through the FK oracle.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--lm_damping", type=float, default=1e-1)
    args = parser.parse_args(argv)

    out = replay_run(Path(args.run_dir), max_iter=int(args.max_iter), tol=float(args.tol), lm_damping=float(args.lm_damping))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
