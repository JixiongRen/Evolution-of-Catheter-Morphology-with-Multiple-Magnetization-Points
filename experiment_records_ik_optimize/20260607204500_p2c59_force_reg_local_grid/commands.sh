#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607204500_p2c59_force_reg_local_grid/run_p2c59_force_reg_local_grid.py \
  > experiment_records_ik_optimize/20260607204500_p2c59_force_reg_local_grid/stdout.log \
  2> experiment_records_ik_optimize/20260607204500_p2c59_force_reg_local_grid/stderr.log
