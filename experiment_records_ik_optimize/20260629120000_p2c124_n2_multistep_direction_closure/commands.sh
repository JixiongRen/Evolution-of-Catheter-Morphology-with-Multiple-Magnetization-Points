#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629120000_p2c124_n2_multistep_direction_closure/run_p2c124_n2_multistep_direction_closure.py \
  > experiment_records_ik_optimize/20260629120000_p2c124_n2_multistep_direction_closure/stdout.log \
  2> experiment_records_ik_optimize/20260629120000_p2c124_n2_multistep_direction_closure/stderr.log
