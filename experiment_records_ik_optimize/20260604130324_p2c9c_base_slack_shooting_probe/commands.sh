#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604130324_p2c9c_base_slack_shooting_probe/run_p2c9c_base_slack_shooting_probe.py \
  --eps 0.02,0.05,0.10 \
  --slack_mu 1e-6,1e-4,1e-2,1e0 \
  --base_reg_mu 1.0e-4 \
  --max_iter 80 \
  --random_count 1
