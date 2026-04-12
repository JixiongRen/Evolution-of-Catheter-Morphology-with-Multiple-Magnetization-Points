from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .magnetization_aware_current_init import generate_magnetization_aware_current_init


def _vec3_csv(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(s).split(",") if str(x).strip() != ""]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {len(vals)} from: {s}")
    return np.asarray(vals, dtype=np.float64).reshape(3,)


def _csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if str(x).strip() != ""]


def _fmt_csv(vals: Sequence[float]) -> str:
    return ",".join(f"{float(v):.10g}" for v in vals)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto current init + IK solve + plotting pipeline")

    p.add_argument("--p_des", type=str, required=True, help="Target point x,y,z [m]")
    p.add_argument("--p0", type=str, default="0,0,-0.05", help="Base point x,y,z [m]")

    p.add_argument("--actuation_table_pkl", type=str, default="supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl")
    p.add_argument("--I_magnitude", type=float, default=15.0)
    p.add_argument("--I_max", type=float, default=50.0)
    p.add_argument("--min_coil_fraction", type=float, default=0.05)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--current_init_ridge_lambda", type=float, default=1e-6)
    p.add_argument("--I_ref_scale", type=float, default=0.5)
    p.add_argument("--current_init_torque_gain", type=float, default=1.0)

    p.add_argument("--run_mode", type=str, default="v1_staged", choices=["baseline", "v1_staged"])
    p.add_argument("--dist_tol_mm", type=float, default=1.0, help="Stop early when dist_to_target_mm falls below this threshold; <=0 disables")
    p.add_argument("--show_plot", action="store_true", help="Show interactive centerline 3D window after solve")
    p.add_argument("--plot_max_lines", type=int, default=200)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--flex_lengths_tail", type=str, default="0.03,0.03")
    p.add_argument("--rigid_lengths", type=str, default="0.003,0.003,0.003")
    p.add_argument("--M_list", type=str, default="3,2,2")  # 保持较低离散化以加快速度
    p.add_argument("--x_min", type=float, default=0.06)
    p.add_argument("--x_max", type=float, default=0.18)
    p.add_argument("--L_protrude_max", type=float, default=0.18)
    p.add_argument("--L1_min", type=float, default=1e-6)
    p.add_argument("--x_init", type=float, default=0.1)
    p.add_argument("--init_mode", type=str, default="equilibrium", choices=["geometry", "equilibrium"])
    p.add_argument("--init_fk_max_iter", type=int, default=300)  # 降低初始化迭代次数
    p.add_argument("--init_fk_tol", type=float, default=1e-4)  # 放宽初始化容差
    p.add_argument("--init_fk_lm_damping", type=float, default=1e-2)

    p.add_argument("--Q0_wxyz", type=str, default="1,0,0,0")
    p.add_argument("--axis_body", type=str, default="1,0,0")
    p.add_argument("--enable_gravity", action="store_true")
    p.add_argument("--g_world", type=str, default="0,0,-9.81")
    p.add_argument("--enable_magnetics", action="store_true")
    p.add_argument("--m_body_list", type=str, default="0,0,0.005301; 0,0,0.005301; 0,0,0.005301")

    p.add_argument("--w_E", type=float, default=100.0)
    p.add_argument("--sigma_p", type=float, default=0.005)
    p.add_argument("--w_I", type=float, default=1.0)
    p.add_argument("--w_x", type=float, default=0.001)
    p.add_argument("--x_ref", type=float, default=0.1)
    p.add_argument("--I_scale", type=float, default=1.0)
    p.add_argument("--w_smooth", type=float, default=0.2)  # 恢复exp2配置
    p.add_argument("--w_smooth_baseline", type=float, default=0.2)  # 恢复exp2配置
    p.add_argument("--w_gap", type=float, default=2.0)  # 恢复exp2配置
    p.add_argument("--sigma_gap", type=float, default=5e-4)
    p.add_argument("--centerline_rigid_samples", type=int, default=2)

    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--lm_damping", type=float, default=1e-2)
    p.add_argument("--lam_max", type=float, default=1e10)
    p.add_argument("--gtol", type=float, default=1e-12)
    p.add_argument("--xtol", type=float, default=1e-12)
    p.add_argument("--stagnation_patience", type=int, default=80)
    p.add_argument("--stagnation_rtol", type=float, default=1e-6)
    p.add_argument("--reject_patience", type=int, default=20)
    p.add_argument("--high_rho_lam_scale", type=float, default=0.3)
    p.add_argument("--mid_rho_lam_scale", type=float, default=0.5)
    p.add_argument("--low_rho_lam_gain", type=float, default=10.0)
    p.add_argument("--reject_lam_gain", type=float, default=10.0)
    p.add_argument("--jac_method", type=str, default="fwd", choices=["fwd", "rev"])
    p.add_argument("--step_norm_clip", type=float, default=50.0)
    p.add_argument("--backtrack_max", type=int, default=6)
    p.add_argument("--backtrack_factor", type=float, default=0.5)
    p.add_argument("--block_clip_dz", type=float, default=20.0)
    p.add_argument("--block_clip_dI", type=float, default=10.0)
    p.add_argument("--block_clip_dx", type=float, default=0.015)
    p.add_argument("--accept_kappa_E", type=float, default=1.05)  # 恢复exp2配置
    p.add_argument("--accept_eps_E", type=float, default=1e-4)  # 恢复exp2配置
    p.add_argument("--accept_eps_smooth", type=float, default=0.0)  # 恢复exp2配置
    p.add_argument("--stage1_sigma_scale", type=float, default=5.0)
    p.add_argument("--stage1_w_smooth_scale", type=float, default=1.5)
    p.add_argument("--stage1_block_clip_scale", type=float, default=0.5)
    p.add_argument("--stage1_max_iter", type=int, default=0)
    p.add_argument("--trace_block_topk", type=int, default=0)
    p.add_argument("--trace_block_every", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    p_des = _vec3_csv(args.p_des)
    p0 = _vec3_csv(args.p0)
    q0_vals = _csv_floats(args.Q0_wxyz)
    if len(q0_vals) != 4:
        raise ValueError(f"Expected 4 floats for Q0_wxyz, got {len(q0_vals)} from: {args.Q0_wxyz}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    I_init, info = generate_magnetization_aware_current_init(
        p_des=p_des,
        actuation_table_path=args.actuation_table_pkl,
        p0=p0,
        x_init=float(args.x_init),
        Q0_wxyz=q0_vals,
        axis_body=_vec3_csv(args.axis_body),
        m_body_list=str(args.m_body_list),
        rigid_lengths=[float(x.strip()) for x in str(args.rigid_lengths).split(",") if str(x).strip() != ""],
        I_magnitude=float(args.I_magnitude),
        I_max=float(args.I_max),
        min_coil_fraction=float(args.min_coil_fraction),
        top_k_coils=args.top_k,
        ridge_lambda=float(args.current_init_ridge_lambda),
        I_ref_scale=float(args.I_ref_scale),
        torque_gain=float(args.current_init_torque_gain),
    )
    I_ref = np.asarray(info.get("I_ref", I_init.tolist()), dtype=np.float64).reshape(8,)

    current_init_path = out_dir / "auto_current_init.json"
    current_init_path.write_text(
        json.dumps(
            {
                "I_init": [float(v) for v in I_init.tolist()],
                "I_ref": [float(v) for v in I_ref.tolist()],
                "info": info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    i_csv = _fmt_csv(I_init.tolist())
    i_ref_csv = _fmt_csv(I_ref.tolist())

    m_body_list_str = str(args.m_body_list).replace(";", "; ")
    
    run_cmd = [
        sys.executable,
        "-m",
        "single_layer_ik.run_joint_ik",
        f"--run_mode={str(args.run_mode)}",
        f"--p_des={str(args.p_des)}",
        f"--flex_lengths_tail={str(args.flex_lengths_tail)}",
        f"--rigid_lengths={str(args.rigid_lengths)}",
        f"--M_list={str(args.M_list)}",
        f"--x_min={str(args.x_min)}",
        f"--x_max={str(args.x_max)}",
        f"--L_protrude_max={str(args.L_protrude_max)}",
        f"--L1_min={str(args.L1_min)}",
        f"--I_max={str(args.I_max)}",
        f"--I_init={i_csv}",
        f"--I_ref={i_ref_csv}",
        f"--x_init={str(args.x_init)}",
        f"--init_mode={str(args.init_mode)}",
        f"--init_fk_max_iter={str(args.init_fk_max_iter)}",
        f"--init_fk_tol={str(args.init_fk_tol)}",
        f"--init_fk_lm_damping={str(args.init_fk_lm_damping)}",
        f"--p0_dim={str(args.p0)}",
        f"--Q0_wxyz={str(args.Q0_wxyz)}",
        f"--axis_body={str(args.axis_body)}",
        f"--g_world={str(args.g_world)}",
        f"--actuation_table_pkl={str(args.actuation_table_pkl)}",
        f"--m_body_list={m_body_list_str}",
        f"--w_E={str(args.w_E)}",
        f"--sigma_p={str(args.sigma_p)}",
        f"--w_I={str(args.w_I)}",
        f"--w_x={str(args.w_x)}",
        f"--x_ref={str(args.x_ref)}",
        f"--I_scale={str(args.I_scale)}",
        f"--w_smooth={str(args.w_smooth)}",
        f"--w_smooth_baseline={str(args.w_smooth_baseline)}",
        f"--w_gap={str(args.w_gap)}",
        f"--sigma_gap={str(args.sigma_gap)}",
        f"--centerline_rigid_samples={str(args.centerline_rigid_samples)}",
        f"--max_iter={str(args.max_iter)}",
        f"--tol={str(args.tol)}",
        f"--dist_tol_mm={str(args.dist_tol_mm)}",
        f"--lm_damping={str(args.lm_damping)}",
        f"--lam_max={str(args.lam_max)}",
        f"--gtol={str(args.gtol)}",
        f"--xtol={str(args.xtol)}",
        f"--stagnation_patience={str(args.stagnation_patience)}",
        f"--stagnation_rtol={str(args.stagnation_rtol)}",
        f"--reject_patience={str(args.reject_patience)}",
        f"--high_rho_lam_scale={str(args.high_rho_lam_scale)}",
        f"--mid_rho_lam_scale={str(args.mid_rho_lam_scale)}",
        f"--low_rho_lam_gain={str(args.low_rho_lam_gain)}",
        f"--reject_lam_gain={str(args.reject_lam_gain)}",
        f"--jac_method={str(args.jac_method)}",
        f"--step_norm_clip={str(args.step_norm_clip)}",
        f"--backtrack_max={str(args.backtrack_max)}",
        f"--backtrack_factor={str(args.backtrack_factor)}",
        f"--block_clip_dz={str(args.block_clip_dz)}",
        f"--block_clip_dI={str(args.block_clip_dI)}",
        f"--block_clip_dx={str(args.block_clip_dx)}",
        f"--accept_kappa_E={str(args.accept_kappa_E)}",
        f"--accept_eps_E={str(args.accept_eps_E)}",
        f"--accept_eps_smooth={str(args.accept_eps_smooth)}",
        f"--stage1_sigma_scale={str(args.stage1_sigma_scale)}",
        f"--stage1_w_smooth_scale={str(args.stage1_w_smooth_scale)}",
        f"--stage1_block_clip_scale={str(args.stage1_block_clip_scale)}",
        f"--stage1_max_iter={str(args.stage1_max_iter)}",
        f"--trace_block_topk={str(args.trace_block_topk)}",
        f"--trace_block_every={str(args.trace_block_every)}",
        f"--out_dir={str(out_dir)}",
    ]

    if bool(args.enable_gravity):
        run_cmd.append("--enable_gravity")
    if bool(args.enable_magnetics):
        run_cmd.append("--enable_magnetics")

    subprocess.run(run_cmd, check=True)

    plot_cmd = [
        sys.executable,
        "-m",
        "single_layer_ik.plot_joint_ik_run",
        "--run_dir", str(out_dir),
        "--max_lines", str(args.plot_max_lines),
    ]
    if bool(args.show_plot):
        plot_cmd.append("--show")
    subprocess.run(plot_cmd, check=True)

    result_path = out_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print("\n=== Auto IK Summary ===")
        print(f"out_dir: {out_dir}")
        print(f"stop_reason: {result.get('stop_reason')}")
        print(f"pos_err_norm_mm: {result.get('diag', {}).get('pos_err_norm_mm')}")
        print(f"max_interface_gap_mm: {result.get('diag', {}).get('max_interface_gap_mm')}")
        print(f"I_init: {i_csv}")


if __name__ == "__main__":
    main()
