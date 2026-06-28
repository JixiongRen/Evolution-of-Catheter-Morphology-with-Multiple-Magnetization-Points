#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629090000_p2c121_ik_sensitivity_milestone_summary/run_p2c121_ik_sensitivity_milestone_summary.py \
  > experiment_records_ik_optimize/20260629090000_p2c121_ik_sensitivity_milestone_summary/stdout.log \
  2> experiment_records_ik_optimize/20260629090000_p2c121_ik_sensitivity_milestone_summary/stderr.log
