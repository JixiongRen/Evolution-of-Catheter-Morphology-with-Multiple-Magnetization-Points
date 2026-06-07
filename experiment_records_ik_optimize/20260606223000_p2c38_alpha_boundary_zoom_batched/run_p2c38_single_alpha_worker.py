"""Run one P2c38 alpha point in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mcrs_mpl_cache")

ROOT = Path(__file__).resolve().parents[2]
P2C37_DIR = ROOT / "experiment_records_ik_optimize" / "20260606220500_p2c37_alpha_transition_sensitivity_map"
if str(P2C37_DIR) not in sys.path:
    sys.path.insert(0, str(P2C37_DIR))

import run_p2c37_alpha_transition_sensitivity_map as p2c37


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eps", default="0.0125")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    p2c37.OUT_DIR = out_dir
    sys.argv = [
        sys.argv[0],
        "--alphas",
        f"{args.alpha:.6f}",
        "--eps_values",
        args.eps,
    ]
    p2c37.main()


if __name__ == "__main__":
    main()
