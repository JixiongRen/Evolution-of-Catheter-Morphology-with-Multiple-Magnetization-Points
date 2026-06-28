#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629073000_p2c118_n2_iterative_outer_step_prototype/run_p2c118_n2_iterative_outer_step_prototype.py \
  > experiment_records_ik_optimize/20260629073000_p2c118_n2_iterative_outer_step_prototype/stdout.log \
  2> experiment_records_ik_optimize/20260629073000_p2c118_n2_iterative_outer_step_prototype/stderr.log
