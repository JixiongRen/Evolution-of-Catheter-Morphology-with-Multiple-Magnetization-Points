#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260604110544_p2c7_shooting_sensitivity_audit/run_p2c7_sensitivity_audit.py \
  "$@" \
  > experiment_records_ik_optimize/20260604110544_p2c7_shooting_sensitivity_audit/stdout.log \
  2> experiment_records_ik_optimize/20260604110544_p2c7_shooting_sensitivity_audit/stderr.log

