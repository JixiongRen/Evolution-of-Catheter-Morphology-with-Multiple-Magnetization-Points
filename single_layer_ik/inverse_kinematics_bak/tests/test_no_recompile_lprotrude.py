from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import forward_kinematics_optimized.fk as fk_cli
from forward_kinematics_optimized.fk_engine import ForwardKinematicsEngine


def build_argparser() -> argparse.ArgumentParser:
    p = fk_cli.build_argparser()
    p.add_argument("--n_calls", type=int, default=6)
    p.add_argument("--dx", type=float, default=1e-3)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    flex_lengths = fk_cli._parse_csv_floats(args.flex_lengths)
    rigid_lengths = fk_cli._parse_csv_floats(args.rigid_lengths)
    M_list = fk_cli._parse_csv_ints(args.M_list)

    L1_min = float(args.L1_min)

    L_protrude0 = float(args.L_protrude)
    if not jnp.isfinite(L_protrude0):
        raise ValueError("--L_protrude must be finite")

    L_protrude_max = float(L_protrude0 + abs(float(args.dx)) * max(int(args.n_calls) - 1, 1))

    engine = ForwardKinematicsEngine(
        flex_lengths_tail=flex_lengths,
        rigid_lengths=rigid_lengths,
        M_list=M_list,
        L1_min=L1_min,
        L_protrude_max=L_protrude_max,
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

    I = jnp.asarray(fk_cli._parse_csv_floats(args.coil_currents), dtype=jnp.float64).reshape(-1,)

    print("JAX devices:", jax.devices())
    print("[no_recompile] Expectation: with JAX_LOG_COMPILES=1, jit(residual_bar) should compile only once.")
    print(f"[no_recompile] L_protrude0={L_protrude0:.6g}  dx={float(args.dx):.3g}  n_calls={int(args.n_calls)}")

    for k in range(int(args.n_calls)):
        Lp = float(L_protrude0 + k * float(args.dx))
        z_star, params, _meshes, ok, stats = engine.solve_with_stats(
            coil_currents=I,
            L_protrude=Lp,
            warm_start=True,
            return_stats=True,
        )
        obs = engine.query_sites(z_bar=z_star, scales=params.scales)
        p_tip = obs["tip_p_dim"]
        normE = float(stats.normE) if stats is not None else float("nan")
        print(f"[no_recompile] k={k:02d} L_protrude={Lp:.6g} ok={bool(ok)} ||E||={normE:.3e} tip={p_tip.tolist()}")


if __name__ == "__main__":
    main()
