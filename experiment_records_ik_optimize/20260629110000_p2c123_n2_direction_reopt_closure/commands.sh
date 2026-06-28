#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629110000_p2c123_n2_direction_reopt_closure/run_p2c123_n2_direction_reopt_closure.py \
  > experiment_records_ik_optimize/20260629110000_p2c123_n2_direction_reopt_closure/stdout.log \
  2> experiment_records_ik_optimize/20260629110000_p2c123_n2_direction_reopt_closure/stderr.log
