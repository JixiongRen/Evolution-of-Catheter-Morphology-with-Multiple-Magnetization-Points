#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629140000_p2c126_n1_failure_mechanism_repair_map/run_p2c126_n1_failure_mechanism_repair_map.py \
  > experiment_records_ik_optimize/20260629140000_p2c126_n1_failure_mechanism_repair_map/stdout.log \
  2> experiment_records_ik_optimize/20260629140000_p2c126_n1_failure_mechanism_repair_map/stderr.log
