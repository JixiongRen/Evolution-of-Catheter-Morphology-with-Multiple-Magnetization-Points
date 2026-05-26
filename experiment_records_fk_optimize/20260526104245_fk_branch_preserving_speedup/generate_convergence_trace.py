from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FK_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(FK_OPT_DIR))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.efficiency_optimization.fk_under_60s_driver import _extract_centerline
from equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondimJAX
from run_branch_preserving_speedup import TARGET_CURRENTS, build_engine, get_calib_file, make_initial_state, Candidate


def block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    trace_dir = out_dir / "F2_engine_fused_cholesky_direct"
    trace_dir.mkdir(parents=True, exist_ok=True)
    calib_file, _reason = get_calib_file()
    candidate = Candidate(
        name="trace_direct_cholesky_same_path",
        mode="direct",
        dense_solve_method="cholesky",
    )
    engine = build_engine(candidate, calib_file, max_iter=100000, tol=1e-5)
    target = jnp.asarray(TARGET_CURRENTS, dtype=jnp.float64)
    z0, params = make_initial_state(engine, target)
    snapshots: list[np.ndarray] = []
    iters: list[int] = []
    norms: list[float] = []

    def callback(iter_num: int, z_bar: Any, normE: float) -> None:
        if int(iter_num) == 0 or int(iter_num) % 50 == 0:
            snapshots.append(_extract_centerline(z_bar, params, rigid_sample_count=10))
            iters.append(int(iter_num))
            norms.append(float(normE))

    solver = MultiSegmentEquilibriumSolverNondimJAX(params)
    start = time.perf_counter()
    z_star, ok, stats = solver.solve_lm(
        z0,
        max_iter=100000,
        tol=1e-5,
        lm_damping=1e-1,
        jac_method="fwd",
        callback=callback,
        return_stats=True,
    )
    block(z_star)
    elapsed = time.perf_counter() - start
    final_centerline = _extract_centerline(z_star, params, rigid_sample_count=10)
    if not iters or iters[-1] != int(stats.n_iter):
        snapshots.append(final_centerline)
        iters.append(int(stats.n_iter))
        norms.append(float(stats.normE))

    np.savez_compressed(
        trace_dir / "convergence_trace_every_50.npz",
        iters=np.asarray(iters, dtype=np.int64),
        normE=np.asarray(norms, dtype=np.float64),
        centerlines=np.asarray(snapshots, dtype=np.float64),
        diagnostic_wall_s=float(elapsed),
        diagnostic_ok=bool(ok),
    )

    centerlines = np.asarray(snapshots, dtype=np.float64)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("viridis")
    for idx, line in enumerate(centerlines):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=cmap(idx / max(len(centerlines) - 1, 1)), alpha=0.4, linewidth=1.0)
    tips = centerlines[:, -1, :]
    ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color="tab:red", linewidth=1.8, label="tip path")
    ax.plot(final_centerline[:, 0], final_centerline[:, 1], final_centerline[:, 2], color="black", linewidth=2.4, label="final")
    ax.scatter(final_centerline[0, 0], final_centerline[0, 1], final_centerline[0, 2], color="black", s=40, label="start")
    ax.scatter(final_centerline[-1, 0], final_centerline[-1, 1], final_centerline[-1, 2], color="tab:red", s=45, label="final tip")
    ax.set_title("Production candidate convergence morphology")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(trace_dir / "convergence_centerline_every_50.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.semilogy(iters, norms, marker="o", markersize=2.5, linewidth=1.0)
    ax2.axhline(1e-5, color="tab:red", linestyle="--", linewidth=1.0, label="tol=1e-5")
    ax2.set_xlabel("accepted iteration")
    ax2.set_ylabel("||E||")
    ax2.set_title("Production candidate residual trace")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(trace_dir / "convergence_residual_every_50.png", dpi=180)
    plt.close(fig2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
