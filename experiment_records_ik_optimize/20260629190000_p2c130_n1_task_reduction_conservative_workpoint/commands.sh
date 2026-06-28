#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629190000_p2c130_n1_task_reduction_conservative_workpoint/run_p2c130_n1_task_reduction_conservative_workpoint.py \
  > experiment_records_ik_optimize/20260629190000_p2c130_n1_task_reduction_conservative_workpoint/stdout.log \
  2> experiment_records_ik_optimize/20260629190000_p2c130_n1_task_reduction_conservative_workpoint/stderr.log
