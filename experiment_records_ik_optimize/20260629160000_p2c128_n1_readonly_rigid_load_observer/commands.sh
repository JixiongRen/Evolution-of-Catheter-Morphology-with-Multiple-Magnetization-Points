#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629160000_p2c128_n1_readonly_rigid_load_observer/run_p2c128_n1_readonly_rigid_load_observer.py \
  > experiment_records_ik_optimize/20260629160000_p2c128_n1_readonly_rigid_load_observer/stdout.log \
  2> experiment_records_ik_optimize/20260629160000_p2c128_n1_readonly_rigid_load_observer/stderr.log
