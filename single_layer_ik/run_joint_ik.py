from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp


from .joint_problem import (
    Ix_to_u,
    build_params_for_y,
    build_static_fk,
    extract_tip_p_dim,
    joint_residual,
    make_initial_z0_bar,
    residual_bar,
    u_to_Ix,
)
from .joint_lm_solver import JointLMSolver

from .forward_kinematics_optimized_bak.nondim import x_bar_to_dim
from .forward_kinematics_optimized_bak.utils_nondim import unpack_z_bar_jax


jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


def _parse_csv_floats(s: str) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(";", ",").split(",") if str(x).strip() != ""]


def _parse_vec3(s: str) -> Array:
    vals = _parse_csv_floats(s)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {len(vals)} from: {s}")
    return jnp.asarray(vals, dtype=jnp.float64).reshape(3,)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-layer joint IK (penalty) solver")

    # Desired tip position
    p.add_argument("--p_des", type=str, required=True, help="Desired tip position in meters: x,y,z")

    # Geometry
    p.add_argument("--flex_lengths_tail", type=str, required=True, help="L2..LN in meters, comma-separated")
    p.add_argument("--rigid_lengths", type=str, required=True, help="Rigid lengths in meters, comma-separated (N values)")
    p.add_argument("--M_list", type=str, required=True, help="Intervals for each flexible segment, comma-separated (N values)")

    # Length bounds
    p.add_argument("--x_min", type=float, required=True, help="Min protruding length L_protrude [m]")
    p.add_argument("--x_max", type=float, required=True, help="Max protruding length L_protrude [m]")
    p.add_argument("--L_protrude_max", type=float, default=None, help="Reference length for scales; default uses x_max")
    p.add_argument("--L1_min", type=float, default=1e-6, help="Min L1 for advancer safety [m]")

    # Actuation
    p.add_argument("--I_max", type=float, default=5.0, help="Max coil current magnitude [A]")
    p.add_argument("--I_init", type=str, default="0,0,0,0,0,0,0,0", help="Initial currents [A], 8 values")
    p.add_argument("--x_init", type=float, default=None, help="Initial protruding length [m]; default midpoint of [x_min,x_max]")

    # Base pose
    p.add_argument("--p0_dim", type=str, default="0,0,-0.05", help="Base position [m]")
    p.add_argument("--Q0_wxyz", type=str, default="1,0,0,0", help="Base quaternion w,x,y,z")
    p.add_argument("--axis_body", type=str, default="0,0,1", help="Body axis direction")

    # Environment
    p.add_argument("--enable_gravity", action="store_true")
    p.add_argument("--g_world", type=str, default="0,0,-9.81")

    # Magnetics
    p.add_argument("--enable_magnetics", action="store_true")
    p.add_argument("--calib_file", type=str, default=None)
    p.add_argument("--actuation_table_pkl", type=str, default=None)
    p.add_argument("--m_body_list", type=str, default=None, help="N magnets as ';' separated vec3: 'x,y,z; x,y,z; ...'")

    # Joint residual weights
    p.add_argument("--w_E", type=float, default=1.0, help="Weight for equilibrium residual")
    p.add_argument("--sigma_p", type=float, default=1e-3, help="Tip position sigma [m]")
    p.add_argument("--w_I", type=float, default=0.0, help="Current regularization weight")
    p.add_argument("--w_x", type=float, default=0.0, help="Insertion regularization weight")
    p.add_argument("--x_ref", type=float, default=0.0, help="Reference x for regularization")

    # LM solver
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--lm_damping", type=float, default=1e-3)
    p.add_argument("--lam_max", type=float, default=1e10)
    p.add_argument("--gtol", type=float, default=1e-12)
    p.add_argument("--xtol", type=float, default=1e-12)
    p.add_argument("--jac_method", type=str, default="fwd", choices=["fwd", "rev"])
    p.add_argument("--step_norm_clip", type=float, default=0.0, help="Global 2-norm clip for LM step; 0 disables")
    p.add_argument("--backtrack_max", type=int, default=6, help="Max backtracking attempts after LM step")
    p.add_argument("--backtrack_factor", type=float, default=0.5, help="Backtracking shrink factor")

    # Output
    p.add_argument("--out_dir", type=str, default=None, help="Output directory; default ik_out/run_YYYYMMDD_HHMMSS")

    return p


def _parse_list_vec3(s: Optional[str]) -> Optional[list[list[float]]]:
    if s is None or str(s).strip() == "":
        return None
    out: list[list[float]] = []
    for item in str(s).split(";"):
        item = item.strip()
        if not item:
            continue
        vals = _parse_csv_floats(item)
        if len(vals) != 3:
            raise ValueError(f"Invalid vec3 in m_body_list: {item}")
        out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return out


def main() -> None:
    args = build_argparser().parse_args()

    p_des = _parse_vec3(args.p_des)

    flex_lengths_tail = _parse_csv_floats(args.flex_lengths_tail)
    rigid_lengths = _parse_csv_floats(args.rigid_lengths)
    M_list = [int(float(v)) for v in _parse_csv_floats(args.M_list)]

    if len(rigid_lengths) <= 0:
        raise ValueError("rigid_lengths empty")

    if args.L_protrude_max is None:
        L_protrude_max = float(args.x_max)
    else:
        L_protrude_max = float(args.L_protrude_max)

    I_init = jnp.asarray(_parse_csv_floats(args.I_init), dtype=jnp.float64).reshape(8,)
    if args.x_init is None:
        x_init = 0.5 * (float(args.x_min) + float(args.x_max))
    else:
        x_init = float(args.x_init)

    static_fk = build_static_fk(
        flex_lengths_tail=flex_lengths_tail,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        L_protrude_max=float(L_protrude_max),
        L1_min=float(args.L1_min),
        p0_dim=_parse_vec3(args.p0_dim),
        Q0_wxyz=jnp.asarray(_parse_csv_floats(args.Q0_wxyz), dtype=jnp.float64).reshape(4,),
        axis_body=_parse_vec3(args.axis_body),
        enable_gravity=bool(args.enable_gravity),
        g_world=_parse_vec3(args.g_world),
        enable_magnetics=bool(args.enable_magnetics),
        calib_file=args.calib_file,
        actuation_table_pkl=args.actuation_table_pkl,
        m_body_list=_parse_list_vec3(args.m_body_list),
    )

    z0_bar = make_initial_z0_bar(static_fk)
    u0 = Ix_to_u(I_init, x_init, I_max=float(args.I_max), x_min=float(args.x_min), x_max=float(args.x_max))
    y0 = jnp.concatenate([z0_bar.reshape(-1,), u0.reshape(-1,)], axis=0)

    # Output directory
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent.parent / "ik_out" / f"run_{ts}_single_layer"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    trace_path = out_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    def _centerline_from_z(z_bar: Array) -> np.ndarray:
        x_nodes_list_bar, _, x_rigid_list_bar = unpack_z_bar_jax(
            jnp.asarray(z_bar, dtype=jnp.float64),
            M_list=tuple(int(m) for m in static_fk.M_list),
        )
        pts: list[np.ndarray] = []
        for i in range(len(x_nodes_list_bar)):
            nodes = np.asarray(jax.device_get(x_nodes_list_bar[i]))
            for n in range(nodes.shape[0]):
                x_dim = np.asarray(jax.device_get(x_bar_to_dim(jnp.asarray(nodes[n]), static_fk.scales))).reshape(13,)
                pts.append(np.asarray(x_dim[0:3], dtype=np.float64))

            xR_bar = np.asarray(jax.device_get(x_rigid_list_bar[i])).reshape(13,)
            xR_dim = np.asarray(jax.device_get(x_bar_to_dim(jnp.asarray(xR_bar), static_fk.scales))).reshape(13,)
            pts.append(np.asarray(xR_dim[0:3], dtype=np.float64))

        if not pts:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(pts, axis=0)

    z0 = jnp.asarray(y0[: int(static_fk.z_len)], dtype=jnp.float64)
    np.savez_compressed(poses_dir / "iter_0000.npz", centerline=_centerline_from_z(z0))

    def _res(y: Array) -> Array:
        return joint_residual(
            y,
            static_fk=static_fk,
            p_des=p_des,
            I_max=float(args.I_max),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            w_E=float(args.w_E),
            sigma_p=float(args.sigma_p),
            w_I=float(args.w_I),
            w_x=float(args.w_x),
            x_ref=float(args.x_ref),
        )

    solver = JointLMSolver(
        residual_fn=_res,
        jac_method=str(args.jac_method),
        step_norm_clip=float(args.step_norm_clip),
        backtrack_max=int(args.backtrack_max),
        backtrack_factor=float(args.backtrack_factor),
    )

    nZ = int(static_fk.z_len)

    prev_I = np.asarray(jax.device_get(I_init)).reshape(8,)
    prev_x = float(x_init)

    def _callback(rec: Dict[str, Any]) -> None:
        nonlocal prev_I, prev_x

        it = int(rec.get("iter", -1))
        y = jnp.asarray(rec["y"], dtype=jnp.float64).reshape(-1,)
        z = jnp.asarray(y[:nZ], dtype=jnp.float64)
        u = jnp.asarray(y[nZ:], dtype=jnp.float64)
        I, x = u_to_Ix(u, I_max=float(args.I_max), x_min=float(args.x_min), x_max=float(args.x_max))

        I_np = np.asarray(jax.device_get(I)).reshape(8,)
        x_f = float(jax.device_get(x))
        dI = I_np - prev_I
        dx = x_f - prev_x

        p_tip = extract_tip_p_dim(z_bar=z, scales=static_fk.scales, M_list=static_fk.M_list)
        p_tip_np = np.asarray(jax.device_get(p_tip)).reshape(3,)
        p_des_np = np.asarray(jax.device_get(p_des)).reshape(3,)
        pos_err = p_tip_np - p_des_np
        dist_m = float(np.linalg.norm(pos_err))

        centerline = _centerline_from_z(z)
        np.savez_compressed(poses_dir / f"iter_{it:04d}.npz", centerline=centerline)

        r = {
            "iter": int(it),
            "accepted": bool(rec.get("accepted", False)),
            "lam": float(rec.get("lam", float("nan"))),
            "rho": float(rec.get("rho", float("nan"))),
            "pred": float(rec.get("pred", float("nan"))),
            "act": float(rec.get("act", float("nan"))),
            "dy_norm": float(rec.get("dy_norm", float("nan"))),
            "normR": float(rec.get("normR", float("nan"))),
            "cost": float(rec.get("cost", float("nan"))),
            "x_m": float(x_f),
            "dx_m": float(dx),
            "I_A": [float(v) for v in I_np.tolist()],
            "dI_A": [float(v) for v in dI.tolist()],
            "p_tip_m": [float(v) for v in p_tip_np.tolist()],
            "pos_err_m": [float(v) for v in pos_err.tolist()],
            "dist_to_target_m": float(dist_m),
            "dist_to_target_mm": float(dist_m * 1e3),
        }
        trace_path.open("a", encoding="utf-8").write(json.dumps(r) + "\n")

        prev_I = I_np
        prev_x = x_f
    y_star, stats, history = solver.solve_lm(
        y0,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        lm_damping=float(args.lm_damping),
        lam_max=float(args.lam_max),
        gtol=float(args.gtol),
        xtol=float(args.xtol),
        verbose=True,
        return_history=True,
        callback=_callback,
    )

    # Decode final I/x from u
    u_star = jnp.asarray(y_star[nZ:], dtype=jnp.float64)
    I_star, x_star = u_to_Ix(u_star, I_max=float(args.I_max), x_min=float(args.x_min), x_max=float(args.x_max))

    # Diagnostics: tip position & residual component norms
    z_star = jnp.asarray(y_star[:nZ], dtype=jnp.float64)
    params_star = build_params_for_y(static_fk, I=I_star, x=x_star)
    E_star = residual_bar(z_star, params_star)
    r_E_star = jnp.asarray(args.w_E, dtype=jnp.float64) * E_star
    p_tip_star = extract_tip_p_dim(z_bar=z_star, scales=params_star.scales, M_list=params_star.M_list)
    r_p_star = (p_tip_star - p_des) / jnp.asarray(args.sigma_p, dtype=jnp.float64)
    r_I_star = jnp.asarray(args.w_I, dtype=jnp.float64) * I_star
    r_x_star = (jnp.asarray(args.w_x, dtype=jnp.float64) * (x_star - jnp.asarray(args.x_ref, dtype=jnp.float64))).reshape((1,))

    pos_err = p_tip_star - p_des
    pos_err_norm = jnp.linalg.norm(pos_err)

    diag = {
        "p_tip_m": [float(v) for v in np.asarray(jax.device_get(p_tip_star)).reshape(3,).tolist()],
        "pos_err_m": [float(v) for v in np.asarray(jax.device_get(pos_err)).reshape(3,).tolist()],
        "pos_err_norm_m": float(jax.device_get(pos_err_norm)),
        "pos_err_norm_mm": float(jax.device_get(pos_err_norm)) * 1e3,
        "norm_r_E": float(jax.device_get(jnp.linalg.norm(r_E_star))),
        "norm_r_p": float(jax.device_get(jnp.linalg.norm(r_p_star))),
        "norm_r_I": float(jax.device_get(jnp.linalg.norm(r_I_star))),
        "norm_r_x": float(jax.device_get(jnp.linalg.norm(r_x_star))),
    }

    # Save history and result
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "success": bool(stats.ok),
                "stop_reason": str(stats.stop_reason),
                "n_iter": int(stats.n_iter),
                "normR": float(stats.normR),
                "cost": float(stats.cost),
                "I_A": [float(v) for v in np.asarray(jax.device_get(I_star)).reshape(-1,).tolist()],
                "x_m": float(jax.device_get(x_star)),
                "p_des_m": [float(v) for v in np.asarray(jax.device_get(p_des)).reshape(3,).tolist()],
                "diag": diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    init_tip = extract_tip_p_dim(z_bar=jnp.asarray(y0[:nZ], dtype=jnp.float64), scales=static_fk.scales, M_list=static_fk.M_list)
    init_tip_np = np.asarray(jax.device_get(init_tip)).reshape(3,)

    np.savez_compressed(poses_dir / "final.npz", centerline=_centerline_from_z(z_star))

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "p_des": [float(v) for v in np.asarray(jax.device_get(p_des)).reshape(3,).tolist()],
                "target": {
                    "p_des_m": [float(v) for v in np.asarray(jax.device_get(p_des)).reshape(3,).tolist()],
                },
                "init": {
                    "I_A": [float(v) for v in np.asarray(jax.device_get(I_init)).reshape(-1,).tolist()],
                    "x_m": float(x_init),
                    "p_tip_m": [float(v) for v in init_tip_np.reshape(3,).tolist()],
                    "centerline_npz": str(Path("poses") / "iter_0000.npz"),
                },
                "final": {
                    "I_A": [float(v) for v in np.asarray(jax.device_get(I_star)).reshape(-1,).tolist()],
                    "x_m": float(jax.device_get(x_star)),
                    "p_tip_m": [float(v) for v in np.asarray(jax.device_get(p_tip_star)).reshape(3,).tolist()],
                    "centerline_npz": str(Path("poses") / "final.npz"),
                },
                "flex_lengths_tail": [float(v) for v in flex_lengths_tail],
                "rigid_lengths": [float(v) for v in rigid_lengths],
                "M_list": [int(v) for v in M_list],
                "bounds": {"x_min": float(args.x_min), "x_max": float(args.x_max), "I_max": float(args.I_max)},
                "weights": {
                    "w_E": float(args.w_E),
                    "sigma_p": float(args.sigma_p),
                    "w_I": float(args.w_I),
                    "w_x": float(args.w_x),
                    "x_ref": float(args.x_ref),
                },
                "lm": {
                    "max_iter": int(args.max_iter),
                    "tol": float(args.tol),
                    "lm_damping": float(args.lm_damping),
                    "lam_max": float(args.lam_max),
                    "gtol": float(args.gtol),
                    "xtol": float(args.xtol),
                    "jac_method": str(args.jac_method),
                },
                "trace": {"path": "trace.jsonl", "poses_dir": "poses"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Joint-LM] wrote: {out_dir}")


if __name__ == "__main__":
    main()
