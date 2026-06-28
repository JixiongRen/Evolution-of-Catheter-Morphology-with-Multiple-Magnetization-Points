#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629100000_p2c122_n2_workspace_direction_scan/run_p2c122_n2_workspace_direction_scan.py \
  > experiment_records_ik_optimize/20260629100000_p2c122_n2_workspace_direction_scan/stdout.log \
  2> experiment_records_ik_optimize/20260629100000_p2c122_n2_workspace_direction_scan/stderr.log
