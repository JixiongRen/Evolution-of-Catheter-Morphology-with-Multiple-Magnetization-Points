#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629170000_p2c129_n1_explicit_rigid_load_residual/run_p2c129_n1_explicit_rigid_load_residual.py \
  > experiment_records_ik_optimize/20260629170000_p2c129_n1_explicit_rigid_load_residual/stdout.log \
  2> experiment_records_ik_optimize/20260629170000_p2c129_n1_explicit_rigid_load_residual/stderr.log
