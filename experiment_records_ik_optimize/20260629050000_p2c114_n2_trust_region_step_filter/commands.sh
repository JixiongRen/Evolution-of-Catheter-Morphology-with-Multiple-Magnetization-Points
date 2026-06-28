#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629050000_p2c114_n2_trust_region_step_filter/run_p2c114_n2_trust_region_step_filter.py \
  > experiment_records_ik_optimize/20260629050000_p2c114_n2_trust_region_step_filter/stdout.log \
  2> experiment_records_ik_optimize/20260629050000_p2c114_n2_trust_region_step_filter/stderr.log
