#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605124500_p2c14_slack_suppression_counterfactual/run_p2c14_slack_suppression_counterfactual.py \
  --alphas 0.40,0.45 \
  --slack_scales 0.05,0.2,1,5,20 \
  --position_scale 60 \
  --max_iter 180
