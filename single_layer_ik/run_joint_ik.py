from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import jax
import jax.numpy as jnp

from .jax_config import configure_jax_enable_x64
from .forward_kinematics_optimized_bak.equilibrium_solver_nondim import summarize_top_blocks_jax
from .forward_kinematics_optimized_bak.fk_engine import ForwardKinematicsEngine
from .joint_lm_solver import JointLMSolver
from .joint_problem import (
    Ix_to_u,
    build_static_fk,
    extract_centerline_p_dim,
    extract_tip_p_dim,
    joint_residual,
    joint_residual_components,
    make_initial_z0_bar,
    u_to_Ix,
)

JAX_ENABLE_X64 = configure_jax_enable_x64()
Array = jnp.ndarray


@dataclass(frozen=True)
class StageConfig:
    name: str
    sigma_p: float
    w_smooth: float
    max_iter: int
    block_clip_dz: float
    block_clip_dI: float
    block_clip_dx: float
    enable_acceptance_guard: bool
    accept_kappa_E: float
    accept_eps_E: float
    accept_eps_smooth: float


def _parse_csv_floats(s: str) -> list[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(";", ",").split(",") if str(x).strip() != ""]


def _parse_vec3(s: str) -> Array:
    vals = _parse_csv_floats(s)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {len(vals)} from: {s}")
    return jnp.asarray(vals, dtype=jnp.float64).reshape(3,)


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


def _to_numpy(a: Any, *, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    arr = np.asarray(jax.device_get(a), dtype=np.float64)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _block_scale(base: float, scale: float) -> float:
    if base <= 0.0:
        return 0.0
    return float(base) * float(scale)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _is_success_stop_reason(stop_reason: str) -> bool:
    return str(stop_reason) in {"converged", "dist_tol"}


def _build_stage_configs(args: argparse.Namespace) -> list[StageConfig]:
    if str(args.run_mode) == "baseline":
        return [
            StageConfig(
                name="baseline",
                sigma_p=float(args.sigma_p),
                w_smooth=float(args.w_smooth_baseline),
                max_iter=int(args.max_iter),
                block_clip_dz=0.0,
                block_clip_dI=0.0,
                block_clip_dx=0.0,
                enable_acceptance_guard=False,
                accept_kappa_E=float(args.accept_kappa_E),
                accept_eps_E=float(args.accept_eps_E),
                accept_eps_smooth=float(args.accept_eps_smooth),
            )
        ]

    stage1_max_iter = int(args.stage1_max_iter) if int(args.stage1_max_iter) > 0 else max(10, int(args.max_iter) // 3)
    return [
        StageConfig(
            name="stage1_feasible",
            sigma_p=float(args.sigma_p) * float(args.stage1_sigma_scale),
            w_smooth=float(args.w_smooth) * float(args.stage1_w_smooth_scale),
            max_iter=int(stage1_max_iter),
            block_clip_dz=_block_scale(float(args.block_clip_dz), float(args.stage1_block_clip_scale)),
            block_clip_dI=_block_scale(float(args.block_clip_dI), float(args.stage1_block_clip_scale)),
            block_clip_dx=_block_scale(float(args.block_clip_dx), float(args.stage1_block_clip_scale)),
            enable_acceptance_guard=True,
            accept_kappa_E=float(args.accept_kappa_E),
            accept_eps_E=float(args.accept_eps_E),
            accept_eps_smooth=float(args.accept_eps_smooth),
        ),
        StageConfig(
            name="stage2_refine",
            sigma_p=float(args.sigma_p),
            w_smooth=float(args.w_smooth),
            max_iter=int(args.max_iter),
            block_clip_dz=float(args.block_clip_dz),
            block_clip_dI=float(args.block_clip_dI),
            block_clip_dx=float(args.block_clip_dx),
            enable_acceptance_guard=True,
            accept_kappa_E=float(args.accept_kappa_E),
            accept_eps_E=float(args.accept_eps_E),
            accept_eps_smooth=float(args.accept_eps_smooth),
        ),
    ]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-layer joint IK (penalty) solver")

    p.add_argument("--p_des", type=str, required=True, help="Desired tip position in meters: x,y,z")
    p.add_argument("--run_mode", type=str, default="baseline", choices=["baseline", "v1_staged"])

    p.add_argument("--flex_lengths_tail", type=str, required=True, help="L2..LN in meters, comma-separated")
    p.add_argument("--rigid_lengths", type=str, required=True, help="Rigid lengths in meters, comma-separated (N values)")
    p.add_argument("--M_list", type=str, required=True, help="Intervals for each flexible segment, comma-separated (N values)")

    p.add_argument("--x_min", type=float, required=True, help="Min protruding length L_protrude [m]")
    p.add_argument("--x_max", type=float, required=True, help="Max protruding length L_protrude [m]")
    p.add_argument("--L_protrude_max", type=float, default=None, help="Reference length for scales; default uses x_max")
    p.add_argument("--L1_min", type=float, default=1e-6, help="Min L1 for advancer safety [m]")
    p.add_argument("--enforce_feasible_x_bounds", dest="enforce_feasible_x_bounds", action="store_true", help="Clamp optimization x_min to L_fixed + L1_min")
    p.add_argument("--no_enforce_feasible_x_bounds", dest="enforce_feasible_x_bounds", action="store_false", help="Use the raw user-provided x_min even if it is infeasible")
    p.set_defaults(enforce_feasible_x_bounds=True)

    p.add_argument("--I_max", type=float, default=5.0, help="Max coil current magnitude [A]")
    p.add_argument("--I_scale", type=float, default=1.0, help="Amplify Jacobian sensitivity for coil currents (>1 helps optimizer use I more)")
    p.add_argument("--I_ref", type=str, default=None, help="Reference currents [A] for regularisation r_I=w_I*(I-I_ref); default=I_init when magnetics enabled, else zeros")
    p.add_argument("--I_init", type=str, default="0,0,0,0,0,0,0,0", help="Initial currents [A], 8 values")
    p.add_argument("--x_init", type=float, default=None, help="Initial protruding length [m]; default midpoint of [x_min,x_max]")
    p.add_argument("--init_mode", type=str, default="geometry", choices=["geometry", "equilibrium"], help="Initialization mode for z0")
    p.add_argument("--init_fk_max_iter", type=int, default=400, help="FK warm-start max_iter when init_mode=equilibrium")
    p.add_argument("--init_fk_tol", type=float, default=1e-5, help="FK warm-start tol when init_mode=equilibrium")
    p.add_argument("--init_fk_lm_damping", type=float, default=1e-2, help="FK warm-start LM damping when init_mode=equilibrium")

    p.add_argument("--p0_dim", type=str, default="0,0,-0.05", help="Base position [m]")
    p.add_argument("--Q0_wxyz", type=str, default="1,0,0,0", help="Base quaternion w,x,y,z")
    p.add_argument("--axis_body", type=str, default="0,0,1", help="Body axis direction")

    p.add_argument("--enable_gravity", action="store_true")
    p.add_argument("--g_world", type=str, default="0,0,-9.81")

    p.add_argument("--enable_magnetics", action="store_true")
    p.add_argument("--calib_file", type=str, default=None)
    p.add_argument("--actuation_table_pkl", type=str, default=None)
    p.add_argument("--m_body_list", type=str, default=None, help="N magnets as ';' separated vec3: 'x,y,z; x,y,z; ...'")

    p.add_argument("--w_E", type=float, default=1.0, help="Weight for equilibrium residual")
    p.add_argument("--sigma_p", type=float, default=1e-3, help="Tip position sigma [m]")
    p.add_argument("--w_I", type=float, default=0.0, help="Current regularization weight")
    p.add_argument("--w_x", type=float, default=0.0, help="Insertion regularization weight")
    p.add_argument("--x_ref", type=float, default=0.0, help="Reference x for regularization")
    p.add_argument("--w_smooth", type=float, default=0.0, help="Centerline smoothness residual weight for V1 stages")
    p.add_argument("--w_smooth_baseline", type=float, default=0.0, help="Centerline smoothness residual weight for baseline mode")
    p.add_argument("--smooth_include_rigid", dest="smooth_include_rigid", action="store_true", help="Include rigid endpoints in centerline smoothing and plotting")
    p.add_argument("--no_smooth_include_rigid", dest="smooth_include_rigid", action="store_false")
    p.set_defaults(smooth_include_rigid=True)
    p.add_argument("--w_gap", type=float, default=0.0, help="Rigid-flex interface position continuity weight")
    p.add_argument("--sigma_gap", type=float, default=1e-3, help="Interface position continuity scale [m]")
    p.add_argument("--centerline_rigid_samples", type=int, default=2, help="Internal samples per rigid segment for plotting/smoothing centerline")

    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--dist_tol_mm", type=float, default=1.0, help="Stop early when dist_to_target_mm is below this threshold; <=0 disables")
    p.add_argument("--bound_stall_patience", type=int, default=20, help="Stop if x stays pinned at a bound with low current utilization for too many accepted steps")
    p.add_argument("--bound_stall_dist_mm", type=float, default=20.0, help="Only trigger bound-stall stop when distance remains above this threshold [mm]")
    p.add_argument("--bound_stall_current_util", type=float, default=0.15, help="Only trigger bound-stall stop when max current utilization stays below this threshold")
    p.add_argument("--bound_stall_eps_x", type=float, default=1e-4, help="Tolerance for treating x as pinned at a bound [m]")
    p.add_argument("--progress_stall_patience", type=int, default=30, help="Accepted-step window for detecting slow progress in stage-2 rescue scenarios")
    p.add_argument("--progress_stall_dist_mm", type=float, default=20.0, help="Only trigger progress-stall stop when distance remains above this threshold [mm]")
    p.add_argument("--progress_stall_min_delta_mm", type=float, default=6.0, help="Minimum distance reduction required across the progress-stall window [mm]")
    p.add_argument("--progress_stall_min_x_ratio", type=float, default=0.75, help="Only trigger progress-stall stop when x has entered this fraction of the feasible insertion range")
    p.add_argument("--progress_stall_max_current_util", type=float, default=0.1, help="Only trigger progress-stall stop when current utilization remains below this threshold")
    p.add_argument("--lm_damping", type=float, default=1e-3)
    p.add_argument("--lam_max", type=float, default=1e10)
    p.add_argument("--gtol", type=float, default=1e-12)
    p.add_argument("--xtol", type=float, default=1e-12)
    p.add_argument("--stagnation_patience", type=int, default=80, help="Stop if accepted steps no longer improve cost")
    p.add_argument("--stagnation_rtol", type=float, default=1e-6, help="Relative improvement threshold for stagnation")
    p.add_argument("--reject_patience", type=int, default=20, help="Stop after too many consecutive rejected steps")
    p.add_argument("--high_rho_lam_scale", type=float, default=0.3, help="Lambda scale when rho > 0.75")
    p.add_argument("--mid_rho_lam_scale", type=float, default=0.5, help="Lambda scale when 0.25 < rho <= 0.75")
    p.add_argument("--low_rho_lam_gain", type=float, default=10.0, help="Lambda gain when accepted step has rho <= 0.25")
    p.add_argument("--reject_lam_gain", type=float, default=10.0, help="Lambda gain after rejected step")
    p.add_argument("--jac_method", type=str, default="fwd", choices=["fwd", "rev"])
    p.add_argument("--step_norm_clip", type=float, default=0.0, help="Global 2-norm clip for LM step; 0 disables")
    p.add_argument("--backtrack_max", type=int, default=6, help="Max backtracking attempts after LM step")
    p.add_argument("--backtrack_factor", type=float, default=0.5, help="Backtracking shrink factor")
    p.add_argument("--block_clip_dz", type=float, default=20.0, help="Per-step clip for ||Δz|| in joint state space; <=0 disables")
    p.add_argument("--block_clip_dI", type=float, default=5.0, help="Per-step clip for ||ΔI|| in amperes; <=0 disables")
    p.add_argument("--block_clip_dx", type=float, default=0.015, help="Per-step clip for |Δx| in meters; <=0 disables")
    p.add_argument("--accept_kappa_E", type=float, default=1.05, help="Allowable multiplicative increase for raw equilibrium residual")
    p.add_argument("--accept_eps_E", type=float, default=1e-4, help="Allowable additive increase for raw equilibrium residual")
    p.add_argument("--accept_eps_smooth", type=float, default=0.0, help="Allowable additive increase for raw smoothness metric; <=0 disables this extra check")
    p.add_argument("--stage1_sigma_scale", type=float, default=2.5, help="Stage-1 sigma_p multiplier in v1_staged mode")
    p.add_argument("--stage1_w_smooth_scale", type=float, default=0.8, help="Stage-1 smoothness weight multiplier in v1_staged mode")
    p.add_argument("--stage1_block_clip_scale", type=float, default=0.75, help="Stage-1 block clip scale factor in v1_staged mode")
    p.add_argument("--stage1_max_iter", type=int, default=0, help="Stage-1 max_iter in v1_staged mode; <=0 uses max(10, max_iter//3)")
    p.add_argument("--trace_block_topk", type=int, default=0, help="If >0, record top residual blocks into trace/result")
    p.add_argument("--trace_block_every", type=int, default=0, help="Iteration interval for block summaries; 0 disables per-iter block logging")
    p.add_argument("--pose_save_every", type=int, default=10, help="Save intermediate centerline NPZ every N solver callbacks; <=0 disables intermediate saves")

    p.add_argument("--out_dir", type=str, default=None, help="Output directory; default ik_out/run_YYYYMMDD_HHMMSS")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    p_des = _parse_vec3(args.p_des)
    p0_dim = _parse_vec3(args.p0_dim)
    Q0_wxyz = jnp.asarray(_parse_csv_floats(args.Q0_wxyz), dtype=jnp.float64).reshape(4,)
    axis_body = _parse_vec3(args.axis_body)
    g_world = _parse_vec3(args.g_world)
    m_body_list = _parse_list_vec3(args.m_body_list)

    flex_lengths_tail = _parse_csv_floats(args.flex_lengths_tail)
    rigid_lengths = _parse_csv_floats(args.rigid_lengths)
    M_list = [int(float(v)) for v in _parse_csv_floats(args.M_list)]
    if len(rigid_lengths) <= 0:
        raise ValueError("rigid_lengths empty")

    L_protrude_max = float(args.x_max) if args.L_protrude_max is None else float(args.L_protrude_max)
    I_init = jnp.asarray(_parse_csv_floats(args.I_init), dtype=jnp.float64).reshape(8,)
    x_init = 0.5 * (float(args.x_min) + float(args.x_max)) if args.x_init is None else float(args.x_init)

    if args.I_ref is not None:
        I_ref = jnp.asarray(_parse_csv_floats(args.I_ref), dtype=jnp.float64).reshape(8,)
    elif bool(args.enable_magnetics):
        I_ref = I_init
    else:
        I_ref = None

    static_fk = build_static_fk(
        flex_lengths_tail=flex_lengths_tail,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        L_protrude_max=float(L_protrude_max),
        L1_min=float(args.L1_min),
        p0_dim=p0_dim,
        Q0_wxyz=Q0_wxyz,
        axis_body=axis_body,
        enable_gravity=bool(args.enable_gravity),
        g_world=g_world,
        enable_magnetics=bool(args.enable_magnetics),
        calib_file=args.calib_file,
        actuation_table_pkl=args.actuation_table_pkl,
        m_body_list=m_body_list,
    )

    feasible_x_min = float(static_fk.L_fixed + static_fk.L1_min)
    x_min_opt = max(float(args.x_min), feasible_x_min) if bool(args.enforce_feasible_x_bounds) else float(args.x_min)
    x_max_opt = float(args.x_max)
    if x_max_opt <= x_min_opt:
        raise ValueError(f"Infeasible optimization bounds: x_min_opt={x_min_opt:.6g}, x_max={x_max_opt:.6g}")
    if x_min_opt > float(args.x_min):
        print(
            f"[Joint-LM] tighten x_min from {float(args.x_min):.6g} to feasible {x_min_opt:.6g} "
            f"(L_fixed + L1_min = {feasible_x_min:.6g})"
        )

    init_summary: Dict[str, Any] = {"mode": str(args.init_mode)}
    if str(args.init_mode) == "equilibrium":
        fk_engine = ForwardKinematicsEngine(
            flex_lengths_tail=flex_lengths_tail,
            rigid_lengths=rigid_lengths,
            M_list=M_list,
            L1_min=float(args.L1_min),
            p0_dim=p0_dim,
            Q0_wxyz=Q0_wxyz,
            axis_body=axis_body,
            enable_gravity=bool(args.enable_gravity),
            g_world=g_world,
            enable_magnetics=bool(args.enable_magnetics),
            calib_file=args.calib_file,
            actuation_table_pkl=args.actuation_table_pkl,
            m_body_list=m_body_list,
            max_iter=int(args.init_fk_max_iter),
            tol=float(args.init_fk_tol),
            lm_damping=float(args.init_fk_lm_damping),
            jac_method=str(args.jac_method),
            L_protrude_max=float(L_protrude_max),
        )
        z0_bar, _params_init, _meshes_init, ok_init, fk_stats = fk_engine.solve_with_stats(
            coil_currents=I_init,
            L_protrude=float(x_init),
            warm_start=False,
            return_stats=True,
        )
        init_summary.update(
            {
                "fk_ok": bool(ok_init),
                "fk_stop_reason": None if fk_stats is None else str(getattr(fk_stats, "stop_reason", "")),
                "fk_normE": None if fk_stats is None else float(getattr(fk_stats, "normE", float("nan"))),
                "fk_n_iter": None if fk_stats is None else int(getattr(fk_stats, "n_iter", -1)),
            }
        )
    else:
        z0_bar = make_initial_z0_bar(static_fk)

    u0 = Ix_to_u(I_init, x_init, I_max=float(args.I_max), x_min=float(x_min_opt), x_max=float(x_max_opt), I_scale=float(args.I_scale))
    y0 = jnp.concatenate([z0_bar.reshape(-1,), u0.reshape(-1,)], axis=0)

    stage_configs = _build_stage_configs(args)

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

    nZ = int(static_fk.z_len)
    prev_I = _to_numpy(I_init, shape=(8,))
    prev_x = float(np.clip(x_init, x_min_opt, x_max_opt))
    global_iter = 0

    def _centerline_from_z(z_bar: Array) -> np.ndarray:
        return _to_numpy(
            extract_centerline_p_dim(
                z_bar=jnp.asarray(z_bar, dtype=jnp.float64),
                scales=static_fk.scales,
                M_list=static_fk.M_list,
                include_rigid=bool(args.smooth_include_rigid),
                rigid_sample_count=int(args.centerline_rigid_samples),
            )
        )

    def _parts_for_stage(y: Array, stage: StageConfig) -> Dict[str, Any]:
        return joint_residual_components(
            y,
            static_fk=static_fk,
            p_des=p_des,
            I_max=float(args.I_max),
            x_min=float(x_min_opt),
            x_max=float(x_max_opt),
            w_E=float(args.w_E),
            sigma_p=float(stage.sigma_p),
            w_I=float(args.w_I),
            w_x=float(args.w_x),
            w_smooth=float(stage.w_smooth),
            x_ref=float(args.x_ref),
            smooth_include_rigid=bool(args.smooth_include_rigid),
            w_gap=float(args.w_gap),
            sigma_gap=float(args.sigma_gap),
            centerline_rigid_samples=int(args.centerline_rigid_samples),
            I_scale=float(args.I_scale),
            I_ref=I_ref,
        )

    def _make_step_filter(stage: StageConfig):
        def _filter(y: Array, dy: Array) -> tuple[Array, Dict[str, Any]]:
            y = jnp.asarray(y, dtype=jnp.float64).reshape(-1,)
            dy = jnp.asarray(dy, dtype=jnp.float64).reshape(-1,)

            z_step = dy[:nZ]
            dz_norm_pre = float(jnp.linalg.norm(z_step))

            u_cur = y[nZ:]
            I_cur, x_cur = u_to_Ix(u_cur, I_max=float(args.I_max), x_min=float(x_min_opt), x_max=float(x_max_opt), I_scale=float(args.I_scale))
            I_trial, x_trial = u_to_Ix((y + dy)[nZ:], I_max=float(args.I_max), x_min=float(x_min_opt), x_max=float(x_max_opt), I_scale=float(args.I_scale))
            dI_norm_pre = float(jnp.linalg.norm(I_trial - I_cur))
            dx_abs_pre = abs(float(x_trial - x_cur))

            scales = [1.0]
            if stage.block_clip_dz > 0.0 and dz_norm_pre > stage.block_clip_dz:
                scales.append(stage.block_clip_dz / max(dz_norm_pre, 1e-18))
            if stage.block_clip_dI > 0.0 and dI_norm_pre > stage.block_clip_dI:
                scales.append(stage.block_clip_dI / max(dI_norm_pre, 1e-18))
            if stage.block_clip_dx > 0.0 and dx_abs_pre > stage.block_clip_dx:
                scales.append(stage.block_clip_dx / max(dx_abs_pre, 1e-18))

            step_scale = float(min(scales))
            dy_scaled = dy * step_scale

            I_new, x_new = u_to_Ix((y + dy_scaled)[nZ:], I_max=float(args.I_max), x_min=float(x_min_opt), x_max=float(x_max_opt), I_scale=float(args.I_scale))
            info = {
                "step_scale": float(step_scale),
                "dz_norm_pre": float(dz_norm_pre),
                "dI_norm_pre": float(dI_norm_pre),
                "dx_abs_pre": float(dx_abs_pre),
                "dz_norm": float(jnp.linalg.norm(dy_scaled[:nZ])),
                "dI_norm": float(jnp.linalg.norm(I_new - I_cur)),
                "dx_abs": abs(float(x_new - x_cur)),
                "clip_dz": float(stage.block_clip_dz),
                "clip_dI": float(stage.block_clip_dI),
                "clip_dx": float(stage.block_clip_dx),
            }
            return dy_scaled, info

        return _filter

    def _make_acceptance_fn(stage: StageConfig):
        if not stage.enable_acceptance_guard:
            return None

        def _accept(ctx: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
            old_parts = _parts_for_stage(ctx["y"], stage)
            new_parts = _parts_for_stage(ctx["y_new"], stage)

            norm_E_old = float(jnp.linalg.norm(old_parts["E"]))
            norm_E_new = float(jnp.linalg.norm(new_parts["E"]))
            smooth_old = float(old_parts["smooth_metric"])
            smooth_new = float(new_parts["smooth_metric"])

            e_limit = max(float(stage.accept_kappa_E) * norm_E_old, norm_E_old + float(stage.accept_eps_E))
            smooth_enabled = float(stage.accept_eps_smooth) > 0.0
            smooth_limit = smooth_old + float(stage.accept_eps_smooth) if smooth_enabled else float("inf")

            eq_ok = norm_E_new <= e_limit
            smooth_ok = (smooth_new <= smooth_limit) if smooth_enabled else True
            accepted = bool(eq_ok and smooth_ok)

            return accepted, {
                "eq_guard": bool(eq_ok),
                "smooth_guard": bool(smooth_ok),
                "norm_E_old": float(norm_E_old),
                "norm_E_new": float(norm_E_new),
                "norm_E_limit": float(e_limit),
                "smooth_old": float(smooth_old),
                "smooth_new": float(smooth_new),
                "smooth_limit": None if not smooth_enabled else float(smooth_limit),
            }

        return _accept

    def _make_stop_fn(stage: StageConfig):
        dist_tol_mm = float(args.dist_tol_mm)
        bound_stall_patience = int(args.bound_stall_patience)
        bound_stall_dist_mm = float(args.bound_stall_dist_mm)
        bound_stall_current_util = float(args.bound_stall_current_util)
        bound_stall_eps_x = float(args.bound_stall_eps_x)
        progress_stall_patience = int(args.progress_stall_patience)
        progress_stall_dist_mm = float(args.progress_stall_dist_mm)
        progress_stall_min_delta_mm = float(args.progress_stall_min_delta_mm)
        progress_stall_min_x_ratio = float(args.progress_stall_min_x_ratio)
        progress_stall_max_current_util = float(args.progress_stall_max_current_util)
        if dist_tol_mm <= 0.0 and bound_stall_patience <= 0 and progress_stall_patience <= 0:
            return None

        bound_stall_count = 0
        progress_window: list[float] = []

        def _stop(rec: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
            y = jnp.asarray(rec["y"], dtype=jnp.float64).reshape(-1,)
            parts = _parts_for_stage(y, stage)
            p_tip_np = _to_numpy(parts["p_tip"], shape=(3,))
            p_des_np = _to_numpy(p_des, shape=(3,))
            dist_m = float(np.linalg.norm(p_tip_np - p_des_np))
            dist_mm = float(dist_m * 1e3)
            if dist_mm < dist_tol_mm:
                return True, "dist_tol", {
                    "dist_to_target_mm": float(dist_mm),
                    "dist_tol_mm": float(dist_tol_mm),
                    "stage": stage.name,
                }
            nonlocal bound_stall_count
            x_now = float(jax.device_get(parts["x"]))
            I_now = _to_numpy(parts["I"], shape=(8,))
            current_util = float(np.max(np.abs(I_now)) / max(float(args.I_max), 1e-12))
            pinned_low = abs(x_now - float(x_min_opt)) <= bound_stall_eps_x
            pinned_high = abs(x_now - float(x_max_opt)) <= bound_stall_eps_x
            pinned = pinned_low or pinned_high
            if (
                bound_stall_patience > 0
                and pinned
                and current_util <= bound_stall_current_util
                and dist_mm >= bound_stall_dist_mm
            ):
                bound_stall_count += 1
                if bound_stall_count >= bound_stall_patience:
                    return True, "bound_stall", {
                        "dist_to_target_mm": float(dist_mm),
                        "x_m": float(x_now),
                        "x_min_m": float(x_min_opt),
                        "x_max_m": float(x_max_opt),
                        "current_utilization": float(current_util),
                        "bound_side": "low" if pinned_low else "high",
                        "stage": stage.name,
                    }
            else:
                bound_stall_count = 0

            if bool(rec.get("accepted", False)):
                progress_window.append(float(dist_mm))
                if len(progress_window) > max(progress_stall_patience, 1):
                    progress_window.pop(0)
            if (
                progress_stall_patience > 0
                and stage.name == "stage2_refine"
                and len(progress_window) >= progress_stall_patience
            ):
                x_ratio = (x_now - float(x_min_opt)) / max(float(x_max_opt - x_min_opt), 1e-12)
                recent_delta_mm = float(progress_window[0] - progress_window[-1])
                if (
                    dist_mm >= progress_stall_dist_mm
                    and x_ratio >= progress_stall_min_x_ratio
                    and current_util <= progress_stall_max_current_util
                    and recent_delta_mm < progress_stall_min_delta_mm
                ):
                    return True, "progress_stall", {
                        "dist_to_target_mm": float(dist_mm),
                        "recent_delta_mm": float(recent_delta_mm),
                        "window": int(progress_stall_patience),
                        "x_m": float(x_now),
                        "x_ratio": float(x_ratio),
                        "current_utilization": float(current_util),
                        "stage": stage.name,
                    }
            return False, "", {}

        return _stop

    def _maybe_block_summary(stage_iter: int, parts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if int(args.trace_block_topk) <= 0:
            return None
        if int(args.trace_block_every) > 0 and stage_iter % int(args.trace_block_every) != 0:
            return None
        return summarize_top_blocks_jax(
            parts["E"],
            parts["params"],
            topk_flex=max(int(args.trace_block_topk), 3),
            topk_rigid=max(int(args.trace_block_topk), 3),
        )

    np.savez_compressed(poses_dir / "iter_0000.npz", centerline=_centerline_from_z(y0[:nZ]))

    stage_outputs: list[Dict[str, Any]] = []
    y_current = y0

    for stage in stage_configs:
        def _res(y: Array, stage_cfg: StageConfig = stage) -> Array:
            return joint_residual(
                y,
                static_fk=static_fk,
                p_des=p_des,
                I_max=float(args.I_max),
                x_min=float(x_min_opt),
                x_max=float(x_max_opt),
                w_E=float(args.w_E),
                sigma_p=float(stage_cfg.sigma_p),
                w_I=float(args.w_I),
                w_x=float(args.w_x),
                w_smooth=float(stage_cfg.w_smooth),
                x_ref=float(args.x_ref),
                smooth_include_rigid=bool(args.smooth_include_rigid),
                w_gap=float(args.w_gap),
                sigma_gap=float(args.sigma_gap),
                centerline_rigid_samples=int(args.centerline_rigid_samples),
                I_scale=float(args.I_scale),
                I_ref=I_ref,
            )

        solver = JointLMSolver(
            residual_fn=_res,
            jac_method=str(args.jac_method),
            step_norm_clip=float(args.step_norm_clip),
            backtrack_max=int(args.backtrack_max),
            backtrack_factor=float(args.backtrack_factor),
        )

        def _callback(rec: Dict[str, Any], stage_cfg: StageConfig = stage) -> None:
            nonlocal prev_I, prev_x, global_iter

            stage_iter = int(rec.get("iter", -1))
            y = jnp.asarray(rec["y"], dtype=jnp.float64).reshape(-1,)
            parts = _parts_for_stage(y, stage_cfg)

            z = jnp.asarray(parts["z_bar"], dtype=jnp.float64)
            I_np = _to_numpy(parts["I"], shape=(8,))
            x_f = float(jax.device_get(parts["x"]))
            dI = I_np - prev_I
            dx = x_f - prev_x

            p_tip_np = _to_numpy(parts["p_tip"], shape=(3,))
            p_des_np = _to_numpy(p_des, shape=(3,))
            pos_err = p_tip_np - p_des_np
            dist_m = float(np.linalg.norm(pos_err))

            centerline = _to_numpy(parts["centerline_p"])
            centerline_end = centerline[-1].tolist() if centerline.size else None
            interface_gap_mm = _to_numpy(parts["interface_gap_norms"]) * 1e3 if parts["interface_gap_norms"].size else np.zeros((0,), dtype=np.float64)

            block_summary = _maybe_block_summary(stage_iter, parts)

            global_iter += 1
            pose_save_every = int(args.pose_save_every)
            if pose_save_every > 0 and global_iter % pose_save_every == 0:
                np.savez_compressed(poses_dir / f"iter_{global_iter:04d}.npz", centerline=centerline)

            record = {
                "iter": int(global_iter),
                "stage": stage_cfg.name,
                "stage_iter": int(stage_iter),
                "accepted": bool(rec.get("accepted", False)),
                "acceptance_ok": bool(rec.get("acceptance_ok", True)),
                "lam": _safe_float(rec.get("lam")),
                "rho": _safe_float(rec.get("rho")),
                "pred": _safe_float(rec.get("pred")),
                "act": _safe_float(rec.get("act")),
                "dy_norm": _safe_float(rec.get("dy_norm")),
                "g_norm": _safe_float(rec.get("g_norm")),
                "g_norm_inf": _safe_float(rec.get("g_norm_inf")),
                "backtrack_count": int(rec.get("backtrack_count", 0)),
                "damping_attempts": int(rec.get("damping_attempts", 0)),
                "normR": _safe_float(rec.get("normR")),
                "cost": _safe_float(rec.get("cost")),
                "sigma_p": float(stage_cfg.sigma_p),
                "w_smooth": float(stage_cfg.w_smooth),
                "norm_E_raw": float(jnp.linalg.norm(parts["E"])),
                "norm_r_E": float(jnp.linalg.norm(parts["r_E"])),
                "norm_r_p": float(jnp.linalg.norm(parts["r_p"])),
                "norm_r_I": float(jnp.linalg.norm(parts["r_I"])),
                "norm_r_x": float(jnp.linalg.norm(parts["r_x"])),
                "norm_r_gap": float(jnp.linalg.norm(parts["r_gap"])),
                "norm_r_smooth": float(jnp.linalg.norm(parts["r_smooth"])),
                "smooth_metric_raw": float(parts["smooth_metric"]),
                "interface_gap_mm": [float(v) for v in interface_gap_mm.tolist()],
                "max_interface_gap_mm": float(jax.device_get(parts["max_interface_gap"]) * 1e3),
                "x_m": float(x_f),
                "dx_m": float(dx),
                "L1_raw_m": float(jax.device_get(parts["L1_raw_dim"])),
                "L1_dim_m": float(jax.device_get(parts["L1_dim"])),
                "L1_clipped": bool(float(jax.device_get(parts["L1_dim"])) > float(jax.device_get(parts["L1_raw_dim"])) + 1e-12),
                "I_A": [float(v) for v in I_np.tolist()],
                "dI_A": [float(v) for v in dI.tolist()],
                "current_utilization": float(np.max(np.abs(I_np)) / max(float(args.I_max), 1e-12)),
                "p_tip_m": [float(v) for v in p_tip_np.tolist()],
                "p_centerline_end_m": centerline_end,
                "pos_err_m": [float(v) for v in pos_err.tolist()],
                "dist_to_target_m": float(dist_m),
                "dist_to_target_mm": float(dist_m * 1e3),
                "step_info": rec.get("step_info", {}),
                "acceptance_meta": rec.get("acceptance_meta", {}),
            }
            if block_summary is not None:
                record["residual_blocks"] = block_summary
            trace_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")

            prev_I = I_np
            prev_x = x_f

        y_star, stats, history = solver.solve_lm(
            y_current,
            max_iter=int(stage.max_iter),
            tol=float(args.tol),
            lm_damping=float(args.lm_damping),
            lam_max=float(args.lam_max),
            gtol=float(args.gtol),
            xtol=float(args.xtol),
            stagnation_patience=int(args.stagnation_patience),
            stagnation_rtol=float(args.stagnation_rtol),
            reject_patience=int(args.reject_patience),
            high_rho_lam_scale=float(args.high_rho_lam_scale),
            mid_rho_lam_scale=float(args.mid_rho_lam_scale),
            low_rho_lam_gain=float(args.low_rho_lam_gain),
            reject_lam_gain=float(args.reject_lam_gain),
            verbose=True,
            return_history=True,
            step_filter=_make_step_filter(stage),
            acceptance_fn=_make_acceptance_fn(stage),
            stop_fn=_make_stop_fn(stage),
            callback=_callback,
        )

        parts_star = _parts_for_stage(y_star, stage)
        I_star = _to_numpy(parts_star["I"], shape=(8,))
        x_star = float(jax.device_get(parts_star["x"]))
        pos_err = _to_numpy(parts_star["p_tip"], shape=(3,)) - _to_numpy(p_des, shape=(3,))
        stage_outputs.append(
            {
                "name": stage.name,
                "config": {
                    "sigma_p": float(stage.sigma_p),
                    "w_smooth": float(stage.w_smooth),
                    "max_iter": int(stage.max_iter),
                    "block_clip_dz": float(stage.block_clip_dz),
                    "block_clip_dI": float(stage.block_clip_dI),
                    "block_clip_dx": float(stage.block_clip_dx),
                    "enable_acceptance_guard": bool(stage.enable_acceptance_guard),
                    "accept_kappa_E": float(stage.accept_kappa_E),
                    "accept_eps_E": float(stage.accept_eps_E),
                    "accept_eps_smooth": float(stage.accept_eps_smooth),
                },
                "stats": {
                    "success": bool(_is_success_stop_reason(str(stats.stop_reason))),
                    "stop_reason": str(stats.stop_reason),
                    "n_iter": int(stats.n_iter),
                    "normR": float(stats.normR),
                    "cost": float(stats.cost),
                    "stop_meta": history.get("stop_meta", {}),
                    "I_A": [float(v) for v in I_star.tolist()],
                    "x_m": float(x_star),
                    "pos_err_norm_mm": float(np.linalg.norm(pos_err) * 1e3),
                    "norm_E_raw": float(jnp.linalg.norm(parts_star["E"])),
                    "smooth_metric_raw": float(parts_star["smooth_metric"]),
                },
                "history": history,
            }
        )
        y_current = y_star

    final_stage = stage_configs[-1]
    final_output = stage_outputs[-1]
    parts_final = _parts_for_stage(y_current, final_stage)
    z_star = jnp.asarray(parts_final["z_bar"], dtype=jnp.float64)
    I_star = _to_numpy(parts_final["I"], shape=(8,))
    x_star = float(jax.device_get(parts_final["x"]))
    p_tip_star = _to_numpy(parts_final["p_tip"], shape=(3,))
    pos_err = p_tip_star - _to_numpy(p_des, shape=(3,))
    pos_err_norm = float(np.linalg.norm(pos_err))

    final_block_summary = summarize_top_blocks_jax(
        parts_final["E"],
        parts_final["params"],
        topk_flex=max(int(args.trace_block_topk), 3),
        topk_rigid=max(int(args.trace_block_topk), 3),
    )

    diag = {
        "p_tip_m": [float(v) for v in p_tip_star.tolist()],
        "pos_err_m": [float(v) for v in pos_err.tolist()],
        "pos_err_norm_m": float(pos_err_norm),
        "pos_err_norm_mm": float(pos_err_norm * 1e3),
        "norm_E_raw": float(jnp.linalg.norm(parts_final["E"])),
        "norm_r_E": float(jnp.linalg.norm(parts_final["r_E"])),
        "norm_r_p": float(jnp.linalg.norm(parts_final["r_p"])),
        "norm_r_I": float(jnp.linalg.norm(parts_final["r_I"])),
        "norm_r_x": float(jnp.linalg.norm(parts_final["r_x"])),
        "norm_r_gap": float(jnp.linalg.norm(parts_final["r_gap"])),
        "norm_r_smooth": float(jnp.linalg.norm(parts_final["r_smooth"])),
        "smooth_metric_raw": float(parts_final["smooth_metric"]),
        "interface_gap_mm": [float(v) for v in _to_numpy(parts_final["interface_gap_norms"]).tolist()] if parts_final["interface_gap_norms"].size else [],
        "max_interface_gap_mm": float(jax.device_get(parts_final["max_interface_gap"]) * 1e3),
        "L1_raw_m": float(jax.device_get(parts_final["L1_raw_dim"])),
        "L1_dim_m": float(jax.device_get(parts_final["L1_dim"])),
        "L1_clipped": bool(float(jax.device_get(parts_final["L1_dim"])) > float(jax.device_get(parts_final["L1_raw_dim"])) + 1e-12),
        "max_current_utilization": float(np.max(np.abs(I_star)) / max(float(args.I_max), 1e-12)),
        "residual_blocks": final_block_summary,
    }

    history_payload = {
        "run_mode": str(args.run_mode),
        "stages": stage_outputs,
    }
    (out_dir / "history.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "success": bool(_is_success_stop_reason(str(final_output["stats"]["stop_reason"]))),
                "stop_reason": str(final_output["stats"]["stop_reason"]),
                "n_iter": int(sum(int(s["stats"]["n_iter"]) for s in stage_outputs)),
                "normR": float(final_output["stats"]["normR"]),
                "cost": float(final_output["stats"]["cost"]),
                "stop_meta": final_output["stats"].get("stop_meta", {}),
                "I_A": [float(v) for v in I_star.tolist()],
                "x_m": float(x_star),
                "p_des_m": [float(v) for v in _to_numpy(p_des, shape=(3,)).tolist()],
                "run_mode": str(args.run_mode),
                "stages": [s["stats"] | {"name": s["name"]} for s in stage_outputs],
                "diag": diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    init_tip = extract_tip_p_dim(z_bar=jnp.asarray(y0[:nZ], dtype=jnp.float64), scales=static_fk.scales, M_list=static_fk.M_list)
    init_tip_np = _to_numpy(init_tip, shape=(3,))
    np.savez_compressed(poses_dir / "final.npz", centerline=_centerline_from_z(z_star))

    run_cfg = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": vars(args),
        "runtime": {
            "jax_enable_x64": bool(JAX_ENABLE_X64),
        },
        "mode": {
            "run_mode": str(args.run_mode),
            "p_des": [float(v) for v in _to_numpy(p_des, shape=(3,)).tolist()],
            "target": {
                "p_des_m": [float(v) for v in _to_numpy(p_des, shape=(3,)).tolist()],
                "init_tip_m": [float(v) for v in init_tip_np.tolist()],
            },
        },
        "init": {
            "centerline_npz": "poses/iter_0000.npz",
        },
        "final": {
            "centerline_npz": "poses/final.npz",
        },
        "ik": {
            "w_E": float(args.w_E),
            "sigma_p": float(args.sigma_p),
            "dist_tol_mm": float(args.dist_tol_mm),
            "w_I": float(args.w_I),
            "w_x": float(args.w_x),
            "w_smooth": float(args.w_smooth),
            "w_smooth_baseline": float(args.w_smooth_baseline),
            "w_gap": float(args.w_gap),
            "sigma_gap": float(args.sigma_gap),
            "x_ref": float(args.x_ref),
            "I_scale": float(args.I_scale),
        },
        "solver": {
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "lm_damping": float(args.lm_damping),
            "lam_max": float(args.lam_max),
            "gtol": float(args.gtol),
            "xtol": float(args.xtol),
            "stagnation_patience": int(args.stagnation_patience),
            "stagnation_rtol": float(args.stagnation_rtol),
            "reject_patience": int(args.reject_patience),
            "high_rho_lam_scale": float(args.high_rho_lam_scale),
            "mid_rho_lam_scale": float(args.mid_rho_lam_scale),
            "low_rho_lam_gain": float(args.low_rho_lam_gain),
            "reject_lam_gain": float(args.reject_lam_gain),
            "jac_method": str(args.jac_method),
            "step_norm_clip": float(args.step_norm_clip),
            "backtrack_max": int(args.backtrack_max),
            "backtrack_factor": float(args.backtrack_factor),
            "block_clip_dz": float(args.block_clip_dz),
            "block_clip_dI": float(args.block_clip_dI),
            "block_clip_dx": float(args.block_clip_dx),
            "accept_kappa_E": float(args.accept_kappa_E),
            "accept_eps_E": float(args.accept_eps_E),
            "accept_eps_smooth": float(args.accept_eps_smooth),
        },
        "stages": [
            {
                "name": s["name"],
                **s["config"],
            }
            for s in stage_outputs
        ],
        "initialization": init_summary,
        "diagnostics": {
            "trace_block_topk": int(args.trace_block_topk),
            "trace_block_every": int(args.trace_block_every),
            "smooth_include_rigid": bool(args.smooth_include_rigid),
            "centerline_rigid_samples": int(args.centerline_rigid_samples),
        },
        "trace": {"path": "trace.jsonl", "poses_dir": "poses"},
        "storage": {
            "pose_save_every": int(args.pose_save_every),
            "intermediate_pose_pattern": "poses/iter_*.npz",
        },
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2),
        encoding="utf-8",
    )

    print(f"[Joint-LM] wrote: {out_dir}")


if __name__ == "__main__":
    main()
