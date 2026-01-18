"""Commit A test: explicit residual interface E(z, I).

This test checks that:
  residual_bar_zI(z, I0, params) == residual_bar(z, params_with_I0)

Run example (modify args as needed):
  python tests/test_residual_interface.py \
    --flex_lengths=0.03,0.03 --rigid_lengths=0.003,0.003,0.003 \
    --M_list=5,5,5 --L_protrude=0.11 \
    --enable_magnetics --calib_file=/abs/path/calibration.json \
    --coil_currents=0,0,0,0,0,10,0,0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.equilibrium_solver_nondim import residual_bar, residual_bar_zI
from forward_kinematics_optimized.utils_nondim import make_initial_guess_multi_bar_jax

jax.config.update("jax_enable_x64", True)


def main() -> None:
    args = fk_cli.build_argparser().parse_args()

    flex_lengths = fk_cli._parse_csv_floats(args.flex_lengths)
    rigid_lengths = fk_cli._parse_csv_floats(args.rigid_lengths)
    M_list = fk_cli._parse_csv_ints(args.M_list)

    total_L = float(sum(flex_lengths) + sum(rigid_lengths))
    scales = fk_cli.compute_scales_from_flex(L_ref=total_L, d_outer=float(fk_cli._parse_csv_floats(args.flex_d_outer)[0]),
                                             E=float(fk_cli._parse_csv_floats(args.flex_E)[0]),
                                             G=float(fk_cli._parse_csv_floats(args.flex_G)[0]))

    coil_currents = jnp.asarray(fk_cli._parse_csv_floats(args.coil_currents), dtype=jnp.float64)

    params, meshes = fk_cli.build_solver_params(
        flex_lengths=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        flex_d_outer=fk_cli._parse_csv_floats(args.flex_d_outer),
        flex_E=fk_cli._parse_csv_floats(args.flex_E),
        flex_G=fk_cli._parse_csv_floats(args.flex_G),
        flex_rho=fk_cli._parse_csv_floats(args.flex_rho),
        rigid_d_outer=fk_cli._parse_csv_floats(args.rigid_d_outer),
        rigid_rho=fk_cli._parse_csv_floats(args.rigid_rho),
        p0_dim=jnp.asarray([float(x) for x in args.p0.split(",")], dtype=jnp.float64),
        Q0=jnp.asarray([float(x) for x in args.Q0.split(",")], dtype=jnp.float64),
        axis_body=jnp.asarray([float(x) for x in args.axis_body.split(",")], dtype=jnp.float64),
        enable_gravity=bool(args.enable_gravity),
        g_world=jnp.asarray([float(x) for x in args.g_world.split(",")], dtype=jnp.float64),
        enable_magnetics=bool(args.enable_magnetics),
        calib_file=args.calib_file,
        actuation_table_pkl=args.actuation_table_pkl,
        coil_currents=coil_currents,
        m_body_list=fk_cli._parse_list_vec3(args.m_body_list) if args.m_body_list else None,
        scales=scales,
    )

    # initial guess
    z0, *_ = make_initial_guess_multi_bar_jax(
        flex_segs=list(params.flex),
        meshes=meshes,
        rigid_segs=list(params.rigid),
        scales=scales,
        p0_dim=jnp.asarray([float(x) for x in args.p0.split(",")], dtype=jnp.float64),
        Q0=jnp.asarray([float(x) for x in args.Q0.split(",")], dtype=jnp.float64),
        axis_body=jnp.asarray([float(x) for x in args.axis_body.split(",")], dtype=jnp.float64),
    )

    E1 = residual_bar(z0, params)
    E2 = residual_bar_zI(z0, coil_currents, params)

    diff = float(jnp.linalg.norm(E1 - E2))
    rel = diff / max(float(jnp.linalg.norm(E1)), 1e-12)

    print(f"[A] ||E(params)-E(z,I)|| = {diff:.3e} (rel {rel:.3e})")
    assert diff < 1e-9, "Residual interface mismatch (too large)"
    print("[A] PASS")


if __name__ == "__main__":
    main()
