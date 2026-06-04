"""Generate representative morphology figures for the P2c10k seed atlas."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
P2C10I_DIR = ROOT / "experiment_records_ik_optimize" / "20260605005323_p2c10i_boundary_candidate_strict_gate"
for path in (ROOT, P2C10I_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import numpy as np

import run_p2c10i_boundary_candidate_strict_gate as p2c10i
import run_p2c10k_stable_seed_sensitivity_atlas as p2c10k

jax.config.update("jax_enable_x64", True)


def main() -> None:
    morphology_rows = []
    for case_name, alpha in [("n1_tip_z_rand15", 0.10), ("n3_mixed_base_i4", 1.00)]:
        target_case = p2c10i._case_by_name(case_name)
        case = p2c10i._scaled_case(target_case, alpha)
        bundle = p2c10i._build_case_functions(case, continuity_mu=1.0, reg_mu=1.0e-4, max_iter=32)
        base = bundle["base"]
        morphology_rows.append(
            {
                "label": f"{case_name}@{alpha:.2f}",
                "centerline": np.asarray(base.out["centerline"], dtype=float).tolist(),
                "tip_p": np.asarray(base.out["tip_p"], dtype=float).reshape(3).tolist(),
                "target_p": np.asarray(bundle["target_p"], dtype=float).reshape(3).tolist(),
            }
        )
    p2c10k._plot_representative_morphology(morphology_rows)
    print("representative_morphology.png")


if __name__ == "__main__":
    main()
