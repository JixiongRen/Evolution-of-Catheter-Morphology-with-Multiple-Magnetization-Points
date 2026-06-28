#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629130000_p2c125_n2_directional_engineering_policy/run_p2c125_n2_directional_engineering_policy.py \
  > experiment_records_ik_optimize/20260629130000_p2c125_n2_directional_engineering_policy/stdout.log \
  2> experiment_records_ik_optimize/20260629130000_p2c125_n2_directional_engineering_policy/stderr.log
