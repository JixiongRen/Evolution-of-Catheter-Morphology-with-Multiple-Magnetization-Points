#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629230000_p2c132_n1_mid_branch_consistency/run_p2c132_n1_mid_branch_consistency.py \
  > experiment_records_ik_optimize/20260629230000_p2c132_n1_mid_branch_consistency/stdout.log \
  2> experiment_records_ik_optimize/20260629230000_p2c132_n1_mid_branch_consistency/stderr.log
