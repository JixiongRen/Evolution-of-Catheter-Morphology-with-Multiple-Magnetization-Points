#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629150000_p2c127_n1_rigid_load_repair_interface_design/run_p2c127_n1_rigid_load_repair_interface_design.py \
  > experiment_records_ik_optimize/20260629150000_p2c127_n1_rigid_load_repair_interface_design/stdout.log \
  2> experiment_records_ik_optimize/20260629150000_p2c127_n1_rigid_load_repair_interface_design/stderr.log
