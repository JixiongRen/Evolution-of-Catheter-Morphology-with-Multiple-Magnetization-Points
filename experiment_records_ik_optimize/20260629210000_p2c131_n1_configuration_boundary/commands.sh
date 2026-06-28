#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629210000_p2c131_n1_configuration_boundary/run_p2c131_n1_configuration_boundary.py \
  > experiment_records_ik_optimize/20260629210000_p2c131_n1_configuration_boundary/stdout.log \
  2> experiment_records_ik_optimize/20260629210000_p2c131_n1_configuration_boundary/stderr.log
