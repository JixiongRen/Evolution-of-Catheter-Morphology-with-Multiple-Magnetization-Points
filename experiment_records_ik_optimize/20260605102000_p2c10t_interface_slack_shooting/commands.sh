#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605102000_p2c10t_interface_slack_shooting/run_p2c10t_interface_slack_shooting.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00 \
  --variants force,torque,wrench \
  --max_iter 120
