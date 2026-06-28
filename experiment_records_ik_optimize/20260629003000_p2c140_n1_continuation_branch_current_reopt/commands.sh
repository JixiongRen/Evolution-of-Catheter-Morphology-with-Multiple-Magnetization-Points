#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260629003000_p2c140_n1_continuation_branch_current_reopt/run_p2c140_n1_continuation_branch_current_reopt.py \
  > experiment_records_ik_optimize/20260629003000_p2c140_n1_continuation_branch_current_reopt/stdout.log \
  2> experiment_records_ik_optimize/20260629003000_p2c140_n1_continuation_branch_current_reopt/stderr.log
