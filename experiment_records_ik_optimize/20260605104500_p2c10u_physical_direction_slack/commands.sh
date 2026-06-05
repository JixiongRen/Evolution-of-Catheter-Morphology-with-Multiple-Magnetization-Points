#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605104500_p2c10u_physical_direction_slack/run_p2c10u_physical_direction_slack.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00 \
  --variants force_axis,torque_axis,wrench_axis \
  --max_iter 120
