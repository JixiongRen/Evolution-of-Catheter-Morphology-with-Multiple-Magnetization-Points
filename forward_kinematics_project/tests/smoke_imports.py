"""Smoke-test imports and bundled FK resources."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"

for path in (ROOT, FK_OPT_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from fk_engine import ForwardKinematicsEngine  # noqa: E402
from equilibrium_solver_nondim import MultiSegmentEquilibriumSolverNondimJAXCached  # noqa: E402
from supiee_auto_diff.actuation_interpolator_jax import load_actuation_table  # noqa: E402


def main() -> int:
    calib = ROOT / "calib" / "mpem_calibration_file_sp=40_order=1.yaml"
    table = ROOT / "supiee_auto_diff" / "offline_interpolation_data" / "actuation_tables" / "actuation_table.pkl"
    if not calib.exists():
        raise FileNotFoundError(calib)
    if not table.exists():
        raise FileNotFoundError(table)

    loaded = load_actuation_table(table)
    _ = ForwardKinematicsEngine
    _ = MultiSegmentEquilibriumSolverNondimJAXCached

    print(
        {
            "status": "ok",
            "calib": str(calib),
            "actuation_table_shape": tuple(int(v) for v in loaded.A_table.shape),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
