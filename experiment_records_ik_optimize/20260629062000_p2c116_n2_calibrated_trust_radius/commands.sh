#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629062000_p2c116_n2_calibrated_trust_radius/run_p2c116_n2_calibrated_trust_radius.py \
  > experiment_records_ik_optimize/20260629062000_p2c116_n2_calibrated_trust_radius/stdout.log \
  2> experiment_records_ik_optimize/20260629062000_p2c116_n2_calibrated_trust_radius/stderr.log
