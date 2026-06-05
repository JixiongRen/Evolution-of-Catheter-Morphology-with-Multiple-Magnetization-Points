#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260605125315_p2c15_slack_suppressed_sensitivity_audit/run_p2c15_slack_suppressed_sensitivity_audit.py \
  --alpha 0.40 \
  --eps_values 0.005,0.01,0.02 \
  --position_scale 60 \
  --boundary_scale 0.5 \
  --slack_scale 20 \
  --steps_per_flex 4 \
  --max_iter 240
