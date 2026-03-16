"""Commit D smoke test: run the GN IK loop for a few iterations.

This test is intentionally lightweight: it does not assert success on a
hard target (because that depends on calibration and magnet setup). Instead it
verifies that:
  - the script runs,
  - it prints per-iteration IK logs,
  - it computes dp/dI without crashing.

Usage:
  python tests/test_ik_smoke.py --help

Example:
  python tests/test_commit_D_ik_smoke.py \
    --p_des=0.05,0.01,0.04 --L_protrude=0.11 \
    --outer_max_iter=3 --line_search_max=2 \
    --enable_magnetics --calib_file=/abs/path/calibration.json \
    --m_body_list=0,0,-0.005301;0,0,0.005301;0.005301,0,0 \
    --coil_currents=0,0,0,0,0,20,0,0
"""

from __future__ import annotations

import sys

from ..ik_position_gn import main


if __name__ == "__main__":
    # Pass through CLI args to the IK script.
    sys.exit(main())
