"""Refine the perturbation boundary for the random-current three-magnet case."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
P2C30_DIR = ROOT / "experiment_records_ik_optimize" / "20260606194500_p2c30_rand20_multi_eps_recheck"
if str(P2C30_DIR) not in sys.path:
    sys.path.insert(0, str(P2C30_DIR))

import run_p2c30_rand20_multi_eps_recheck as p2c30


def main() -> None:
    p2c30.OUT_DIR = Path(__file__).resolve().parent
    sys.argv = [
        sys.argv[0],
        "--case",
        "n3_mixed_rand20",
        "--target_alpha",
        "0.4",
        "--eps_values",
        "0.0125,0.015,0.0175,0.02",
    ]
    p2c30.main()


if __name__ == "__main__":
    main()
