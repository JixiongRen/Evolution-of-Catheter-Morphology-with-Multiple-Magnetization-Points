from __future__ import annotations
import os
import csv
import time
import argparse
import numpy as np

from pose_modules.segments import RigidSegment, FlexibleSegment
from pose_modules.rod_mesh import RodMesh
from pose_modules.utils import (
    build_k_matrices_for_pdms,
    build_gravity_line_density_for_pdms,
    build_gravity_rigid_for_ndfeb,
    make_initial_guess_multi,
)
from pose_modules.external_wrench import (
    GravityRigid,
    MagneticModel,
    make_external_wrench_density_flexible,
)

from pose_modules.equilibrium_solver import MultiSegmentEquilibriumSolver as BaselineSolver

from pose_modules_nondim.equilibrium_solver_nondim import (
    MultiSegmentEquilibriumSolverNondim as NondimSolver,
)

try:
    from pose_modules_nondim.basics_nondim import (
        NondimScales,
        build_default_scales,
        build_balanced_scales,
    )
except:
    raise ImportError("modules not found")


class SimpleUniformBModel:
    """Minimal uniform-field model used for solver comparisons."""

    def wrench_on_magnet(
        self,
        position_world: np.ndarray,
        R_world_from_body: np.ndarray,
        magnet_params: dict,
        coil_currents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        m_body = np.asarray(magnet_params.get("m_body", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        m_world = R_world_from_body @ m_body
        B_world = np.asarray(coil_currents, dtype=float).reshape(3)
        F_world = np.zeros(3)
        Tau_world = np.cross(m_world, B_world)
        return F_world, Tau_world



def build_case(
    scenario: str = "gravity_only",
    d_outer: float = 1.5e-3,
    Lf_list: list[float] | None = None,
    Lr_list: list[float] | None = None,
    n_intervals: int = 3,
    coil_amp: float = 50.0,
    seed: int | None = None,
):
    """Build a three-flexible, three-rigid benchmark case."""
    if Lf_list is None:
        Lf_list = [0.05, 0.03, 0.01]
    if Lr_list is None:
        Lr_list = [0.03, 0.03, 0.03]

    if seed is not None:
        np.random.seed(seed)

    K_se_pdms, K_bt_pdms = build_k_matrices_for_pdms(d_outer)

    flex_segs: list[FlexibleSegment] = []
    for Lf in Lf_list:
        flex = FlexibleSegment(length=Lf, K_se=K_se_pdms, K_bt=K_bt_pdms)
        flex_segs.append(flex)

    grav_line = build_gravity_line_density_for_pdms(d_outer)
    for flex in flex_segs:
        fext, tauext = make_external_wrench_density_flexible(gravity=grav_line)
        flex.fext_density = fext
        flex.tauext_density = tauext

    meshes: list[RodMesh] = [RodMesh(flex_seg=flex, n_intervals=n_intervals) for flex in flex_segs]

    rigid_segs: list[RigidSegment] = [
        RigidSegment(length=Lr, v_star=np.array([0.0, 0.0, 1.0])) for Lr in Lr_list
    ]

    gravity_rigid_list: list[GravityRigid] = [
        build_gravity_rigid_for_ndfeb(d_outer, Lr) for Lr in Lr_list
    ]

    magnetic_model = None
    magnet_params_list = None
    coil_currents = None

    if scenario == "simple_uniform_B":
        magnetic_model = SimpleUniformBModel()
        m_mag = 5.0e-3
        magnet_params_list = [
            {"m_body": np.array([0.0, 0.0, m_mag])} for _ in rigid_segs
        ]
        direction = np.random.randn(3)
        direction /= (np.linalg.norm(direction) + 1e-12)
        B_mag = float(coil_amp)
        coil_currents = direction * B_mag

    p0_target = np.array([0.0, 0.0, 0.0])
    Q0_target = np.array([1.0, 0.0, 0.0, 0.0])

    z0, x_nodes_list0, k_array_list0, x_rigid_list0 = make_initial_guess_multi(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
    )

    return (
        flex_segs, meshes, rigid_segs,
        gravity_rigid_list,
        magnetic_model, magnet_params_list, coil_currents,
        p0_target, Q0_target,
        z0, x_nodes_list0, k_array_list0, x_rigid_list0,
    )


def compute_scales_from_model(flex_segs: list[FlexibleSegment], rigid_segs: list[RigidSegment]):
    """Derive reference scales from the dimensional model parameters."""
    L_ref = 0.0
    for f, r in zip(flex_segs, rigid_segs):
        L_ref += float(f.length) + float(r.length)

    Kse_ref = 0.0
    Kbt_ref = 0.0
    for flex in flex_segs:
        Kse_ref = max(Kse_ref, float(np.max(np.abs(np.diag(flex.K_se)))))
        Kbt_ref = max(Kbt_ref, float(np.max(np.abs(np.diag(flex.K_bt)))))

    if Kse_ref <= 0:
        Kse_ref = 1.0
    if Kbt_ref <= 0:
        Kbt_ref = 1.0

    return L_ref, Kse_ref, Kbt_ref



def run_once(
    scenario: str,
    scale_type: str,
    n_intervals: int,
    coil_amp: float,
    seed: int,
    tol: float,
    max_iter: int,
):
    (
        flex_segs, meshes, rigid_segs,
        gravity_rigid_list,
        magnetic_model, magnet_params_list, coil_currents,
        p0_target, Q0_target,
        z0_base, x_nodes_list0, k_array_list0, x_rigid_list0,
    ) = build_case(
        scenario=scenario,
        n_intervals=n_intervals,
        coil_amp=coil_amp,
        seed=seed,
    )

    # ---------- Baseline Solver ----------
    baseline_solver = BaselineSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=magnetic_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        max_iter=max_iter,
        tol=tol,
        lm_damping=1e-4,
    )

    history_base = []

    def cb_base(iter_num, z, normE):
        if iter_num % 10 == 0:
            try:
                E = baseline_solver.residual(z)
                J = baseline_solver.jacobian_fd(z, E)
                condJ = np.linalg.cond(J)
            except Exception:
                condJ = np.nan
        else:
            condJ = np.nan
        history_base.append((int(iter_num), float(normE), float(condJ), float(time.perf_counter())))

    t0 = time.perf_counter()
    z_star_base, success_base = baseline_solver.solve(z0_base, callback=cb_base)
    t1 = time.perf_counter()
    time_base = t1 - t0

    # ---------- Nondim Solver ----------
    L_ref, Kse_ref, Kbt_ref = compute_scales_from_model(flex_segs, rigid_segs)
    if scale_type == "balanced":
        scales = build_balanced_scales(L_ref=L_ref, Kse_ref=Kse_ref)
    else:
        scales = build_default_scales(L_ref=L_ref, Kse_ref=Kse_ref, Kbt_ref=Kbt_ref)

    nondim_solver = NondimSolver(
        flex_segs=flex_segs,
        meshes=meshes,
        rigid_segs=rigid_segs,
        p0_target=p0_target,
        Q0_target=Q0_target,
        gravity_rigid_list=gravity_rigid_list,
        magnetic_model=magnetic_model,
        magnet_params_list=magnet_params_list,
        coil_currents=coil_currents,
        max_iter=max_iter,
        tol=tol,
        lm_damping=1e-4,
        nondim_scales=scales,
    )

    z0_nondim = nondim_solver.pack_z(x_nodes_list0, k_array_list0, x_rigid_list0)

    history_nd = []

    def cb_nd(iter_num, z, normE):
        if iter_num % 10 == 0:
            try:
                E = nondim_solver.residual(z)
                J = nondim_solver.jacobian_fd(z, E)
                condJ = np.linalg.cond(J)
            except Exception:
                condJ = np.nan
        else:
            condJ = np.nan
        history_nd.append((int(iter_num), float(normE), float(condJ), float(time.perf_counter())))

    t2 = time.perf_counter()
    z_star_nd, success_nd = nondim_solver.solve(z0_nondim, callback=cb_nd)
    t3 = time.perf_counter()
    time_nd = t3 - t2

    return {
        "baseline": {
            "success": bool(success_base),
            "time": float(time_base),
            "iters": int(history_base[-1][0]) if history_base else 0,
            "history": history_base,
        },
        "nondim": {
            "success": bool(success_nd),
            "time": float(time_nd),
            "iters": int(history_nd[-1][0]) if history_nd else 0,
            "history": history_nd,
        },
    }


def save_results(output_dir: str, tag: str, meta: dict, result: dict):
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(output_dir, f"{tag}_meta.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in meta.items():
            w.writerow([k, v])

    for key in ("baseline", "nondim"):
        hist = result[key]["history"]
        csv_path = os.path.join(output_dir, f"{tag}_{key}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "normE", "condJ", "timestamp"]) 
            for row in hist:
                w.writerow(row)

    sum_path = os.path.join(output_dir, f"{tag}_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solver", "success", "time_sec", "iters"])
        w.writerow(["baseline", result["baseline"]["success"], result["baseline"]["time"], result["baseline"]["iters"]])
        w.writerow(["nondim", result["nondim"]["success"], result["nondim"]["time"], result["nondim"]["iters"]])


# --------------------- CLI ---------------------

def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs nondim (residual scaling) LM solvers.")
    parser.add_argument("--scenario", type=str, default="simple_uniform_B", choices=["gravity_only", "simple_uniform_B"], help="Experiment scenario.")
    parser.add_argument("--scale", type=str, default="balanced", choices=["default", "balanced"], help="Nondimensional scaling scheme.")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of repeated trials.")
    parser.add_argument("--coil_amp", type=float, default=50.0, help="Placeholder field magnitude for the simple uniform-B scenario.")
    parser.add_argument("--n_intervals", type=int, default=3, help="Number of flexible-segment intervals per segment.")
    parser.add_argument("--tol", type=float, default=5e-3, help="LM residual-norm tolerance.")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum LM iterations.")
    parser.add_argument("--out", type=str, default="experiments/results", help="Output directory.")

    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")

    for trial in range(args.n_trials):
        seed = 1000 + trial
        tag = f"{args.scenario}_{args.scale}_trial{trial}_{ts}"

        result = run_once(
            scenario=args.scenario,
            scale_type=args.scale,
            n_intervals=args.n_intervals,
            coil_amp=args.coil_amp,
            seed=seed,
            tol=args.tol,
            max_iter=args.max_iter,
        )

        meta = {
            "scenario": args.scenario,
            "scale": args.scale,
            "n_intervals": args.n_intervals,
            "coil_amp": args.coil_amp,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "seed": seed,
            "timestamp": ts,
        }

        save_results(args.out, tag, meta, result)
        print(f"Saved results to {args.out} with tag {tag}")


if __name__ == "__main__":
    main()
