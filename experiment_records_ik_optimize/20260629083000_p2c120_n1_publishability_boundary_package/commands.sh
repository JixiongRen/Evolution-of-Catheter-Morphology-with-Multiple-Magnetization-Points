#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/run_p2c120_n1_publishability_boundary_package.py \
  > experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/stdout.log \
  2> experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/stderr.log
