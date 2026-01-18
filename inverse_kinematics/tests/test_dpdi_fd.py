"""Commit C test: dp/dI (LM-adjoint) vs finite differences.

This is a *sanity check* to verify the sensitivity has the correct order of
magnitude and sign.

Because each FD sample requires a full FK solve, keep M_list small and use
a modest number of coils.

Example:
  python tests/test_dpdi_fd.py \
    --flex_lengths=0.03,0.03 --rigid_lengths=0.003,0.003,0.003 \
    --M_list=3,3,3 --L_protrude=0.11 \
    --enable_magnetics --calib_file=/abs/path/calibration.json \
    --m_body_list=0,0,-0.005301;0,0,0.005301;0.005301,0,0 \
    --coil_currents=0,0,0,0,0,20,0,0 \
    --fd_eps=0.5
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
from ..ik_diff import compute_dp_dI_via_lm_adjoint, extract_tip_p_dim

jax.config.update("jax_enable_x64", True)


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()
    p.add_argument("--fd_eps", type=float, default=0.5, help="Finite-difference step [A]")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    engine = ForwardKinematicsEngine(
        flex_lengths_tail=fk_cli._parse_csv_floats(args.flex_lengths),
        rigid_lengths=fk_cli._parse_csv_floats(args.rigid_lengths),
        M_list=fk_cli._parse_csv_ints(args.M_list),
        L1_min=float(args.L1_min),
        flex_d_outer=fk_cli._parse_csv_floats(args.flex_d_outer),
        flex_E=fk_cli._parse_csv_floats(args.flex_E),
        flex_G=fk_cli._parse_csv_floats(args.flex_G),
        flex_rho=fk_cli._parse_csv_floats(args.flex_rho),
        rigid_d_outer=fk_cli._parse_csv_floats(args.rigid_d_outer),
        rigid_rho=fk_cli._parse_csv_floats(args.rigid_rho),
        p0_dim=jnp.asarray([float(x) for x in args.p0.split(",")], dtype=jnp.float64),
        Q0_wxyz=jnp.asarray([float(x) for x in args.Q0.split(",")], dtype=jnp.float64),
        axis_body=jnp.asarray([float(x) for x in args.axis_body.split(",")], dtype=jnp.float64),
        enable_gravity=bool(args.enable_gravity),
        g_world=jnp.asarray([float(x) for x in args.g_world.split(",")], dtype=jnp.float64),
        enable_magnetics=bool(args.enable_magnetics),
        calib_file=args.calib_file,
        actuation_table_pkl=args.actuation_table_pkl,
        m_body_list=fk_cli._parse_list_vec3(args.m_body_list) if args.m_body_list else None,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        lm_damping=float(args.lm_damping),
        jac_method=str(args.jac_method),
    )

    I0 = jnp.asarray(fk_cli._parse_csv_floats(args.coil_currents), dtype=jnp.float64)

    # Base solve
    z_star, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=I0,
        L_protrude=float(args.L_protrude),
        warm_start=False,
        return_stats=True,
    )

    p0 = extract_tip_p_dim(z_star, M_list=params.M_list, scales=params.scales)
    jac = compute_dp_dI_via_lm_adjoint(
        z_star_bar=z_star,
        params=params,
        lm_stats=stats,
        coil_currents=I0,
        ridge=None,
    )
    Jp = jac.J_p_I

    # FD on each coil
    eps = float(args.fd_eps)
    Jp_fd_cols = []
    for j in range(I0.shape[0]):
        ej = jnp.zeros_like(I0).at[j].set(1.0)
        Ip = I0 + eps * ej
        Im = I0 - eps * ej

        zp, pp, _mp, okp = engine.solve(coil_currents=Ip, L_protrude=float(args.L_protrude), warm_start=True)
        zm, pm, _mm, okm = engine.solve(coil_currents=Im, L_protrude=float(args.L_protrude), warm_start=True)

        p_tip_p = extract_tip_p_dim(zp, M_list=pp.M_list, scales=pp.scales)
        p_tip_m = extract_tip_p_dim(zm, M_list=pm.M_list, scales=pm.scales)

        col = (p_tip_p - p_tip_m) / (2.0 * eps)
        Jp_fd_cols.append(col)
        print(f"[C] coil {j}: ok(+/-)=({okp},{okm})")

    Jp_fd = jnp.stack(Jp_fd_cols, axis=1)  # (3, nI)

    diff = float(jnp.linalg.norm(Jp - Jp_fd))
    base = max(float(jnp.linalg.norm(Jp_fd)), 1e-12)
    rel = diff / base

    print("[C] p0:", p0.tolist())
    print("[C] ||J_adj - J_fd||:", diff)
    print("[C] rel:", rel)

    # Loose tolerance: this is just a sanity check.
    assert rel < 0.5, "dp/dI adjoint deviates too much from FD (tune fd_eps / solver tol)"
    print("[C] PASS")


if __name__ == "__main__":
    main()
