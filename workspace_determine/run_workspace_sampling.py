
from __future__ import annotations

import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp

from forward_kinematics_nondim_optimized.fk import build_solver_params, compute_scales_from_flex
from forward_api import ForwardModel
from workspace_analyzer import WorkspaceAnalyzer


def _first(x):
    return x[0] if isinstance(x, list) else x


def _to_jsonable(obj):
    """Recursively convert numpy/jax arrays and scalars into JSON-serializable Python types.

    - np.ndarray / jnp.ndarray -> list
    - numpy/jax scalar -> Python scalar via .item()
    - dict/list/tuple/set -> recursively converted (set -> list)
    - other objects are returned as-is
    """
    # numpy / jax array
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    # numpy scalar (e.g., np.float64, np.int32)
    if isinstance(obj, np.generic):
        return obj.item()
    # common containers
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in obj]
    # leave other types (str, int, float, bool, None) as-is
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to a JSON config file (same style as example's arguments).")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--I_max", type=float, default=5.0, help="Current bound (A): sample in [-I_max, I_max]^8")
    ap.add_argument("--method", type=str, default="lhs", choices=["lhs", "uniform"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voxel", type=float, default=0.002, help="Voxel size (m). Set <=0 to disable voxelization.")
    ap.add_argument("--out_dir", type=str, default="workspace_out")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)


    # --- Base pose and axis ---
    p0_dim = jnp.asarray(cfg.get("p0_dim", [0.0, 0.0, 0.0]), dtype=jnp.float64).reshape(3,)
    Q0 = jnp.asarray(cfg.get("Q0", [1.0, 0.0, 0.0, 0.0]), dtype=jnp.float64).reshape(4,)
    axis_body = jnp.asarray(cfg.get("axis_body", [0.0, 0.0, 1.0]), dtype=jnp.float64).reshape(3,)

    enable_magnetics = bool(cfg.get("enable_magnetics", True))
    coil_currents = jnp.zeros((8,), dtype=jnp.float64)
    m_body_list = None
    if enable_magnetics:
        m_body_list = [jnp.asarray(v, dtype=jnp.float64).reshape(3,) for v in cfg["m_body_list"]]


    # --- Scales ---
    L_ref = float(cfg.get("L_ref", sum(cfg["flex_lengths"]) + sum(cfg["rigid_lengths"])))
    scales = compute_scales_from_flex(
        L_ref=L_ref,
        d_outer=float(_first(cfg["flex_d_outer"])),
        E=float(_first(cfg["flex_E"])),
        G=float(_first(cfg["flex_G"])),
        enable_magnetics=bool(cfg.get("enable_magnetics", True)),
        calib_file=cfg.get("calib_file", None),
        coil_currents=jnp.zeros((8,), dtype=jnp.float64),
        m_body_list=m_body_list,
        p0_dim=p0_dim,
        Q0=Q0,
        axis_body=axis_body,
        flex_lengths=cfg["flex_lengths"],
        rigid_lengths=cfg["rigid_lengths"],
    )


    params, meshes = build_solver_params(
        flex_lengths=cfg["flex_lengths"],
        rigid_lengths=cfg["rigid_lengths"],
        M_list=cfg["M_list"],
        flex_d_outer=cfg["flex_d_outer"],
        flex_E=cfg["flex_E"],
        flex_G=cfg["flex_G"],
        flex_rho=cfg["flex_rho"],
        rigid_d_outer=cfg["rigid_d_outer"],
        rigid_rho=cfg["rigid_rho"],
        scales=scales,
        p0_dim=p0_dim,
        Q0=Q0,
        axis_body=axis_body,
        enable_gravity=bool(cfg.get("enable_gravity", True)),
        g_world=jnp.asarray(cfg.get("g_world", [0.0, 0.0, -9.81]), dtype=jnp.float64),
        enable_magnetics=enable_magnetics,
        calib_file=cfg.get("calib_file", None),
        actuation_table_pkl=cfg.get("actuation_table_pkl", None),
        coil_currents=coil_currents,
        m_body_list=m_body_list,
    )

    fwd = ForwardModel(
        params=params,
        meshes=meshes,
        max_iter=int(cfg.get("max_iter", 50000)),
        lm_lambda_init=float(cfg.get("lm_lambda_init", 1e-1)),
        verbose=bool(cfg.get("verbose", True)),
    )
    wa = WorkspaceAnalyzer(fwd)

    voxel = None if args.voxel <= 0 else float(args.voxel)
    out = wa.sample_forward(
        n_samples=int(args.n_samples),
        I_max=float(args.I_max),
        method=args.method,
        seed=int(args.seed),
        save_npz=os.path.join(args.out_dir, "samples.npz"),
        voxel=voxel,
        save_voxels_npz=os.path.join(args.out_dir, "workspace_voxels.npz") if voxel is not None else None,
    )

    summary = out["summary"]
    # Convert all possible ndarray or numpy/jax scalar values to JSON-serializable types
    summary_json = _to_jsonable(summary)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("[workspace] Done.")
    print("  success_rate:", summary["success_rate"])
    print("  p_min (m):", summary["p_min"])
    print("  p_max (m):", summary["p_max"])
    print("  outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
