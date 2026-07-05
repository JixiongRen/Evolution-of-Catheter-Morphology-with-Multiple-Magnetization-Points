#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260705113000_p2c195_n2_gpu_cached_steady_replay/run_p2c195_n2_gpu_cached_steady_replay.py
