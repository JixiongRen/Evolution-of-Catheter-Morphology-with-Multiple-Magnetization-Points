#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy/run_p2c119_n2_diagnostic_outer_ik_policy.py \
  > experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy/stdout.log \
  2> experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy/stderr.log
