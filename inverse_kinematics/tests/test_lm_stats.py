"""Commit B test: LM returns stats (including final Jacobian).

This test runs one FK solve with ``return_stats=True`` and prints:
  - LM stop_reason
  - ||E|| and ok_strict
  - shapes of J (dE/dz)

Run example (edit args to your environment):
  python tests/test_lm_stats.py \
    --flex_lengths=0.03,0.03 --rigid_lengths=0.003,0.003,0.003 \
    --M_list=5,5,5 --L_protrude=0.11 \
    --enable_magnetics --calib_file=/abs/path/calibration.json \
    --m_body_list=0,0,-0.005301;0,0,0.005301;0.005301,0,0 \
    --coil_currents=0,0,0,0,0,20,0,0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine
import forward_kinematics_optimized.fk as fk_cli

jax.config.update("jax_enable_x64", True)


def main() -> None:
    args = fk_cli.build_argparser().parse_args()

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

    I = fk_cli._parse_csv_floats(args.coil_currents)
    z_star, params, _meshes, ok, stats = engine.solve_with_stats(
        coil_currents=I,
        L_protrude=float(args.L_protrude),
        warm_start=False,
        return_stats=True,
    )

    print("[B] ok_strict:", bool(ok))
    print("[B] stop_reason:", stats.stop_reason)
    print("[B] ||E||:", stats.normE)
    print("[B] lam:", stats.lam)
    print("[B] J shape (nE,nZ):", tuple(stats.J.shape))

    assert stats.J.ndim == 2
    assert stats.J.shape[1] == z_star.shape[0]
    print("[B] PASS")


if __name__ == "__main__":
    main()
