#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260605145500_p2c17_refined_boundary_path_dependence/run_p2c17_refined_boundary_path_dependence.py \
  --lower_alphas 0.395,0.390,0.385 \
  --upper_alphas 0.435,0.440,0.445,0.450 \
  --eps_alpha 0.01 \
  --position_scale 60 \
  --boundary_scale 0.5 \
  --slack_scale 20 \
  --steps_per_flex 4 \
  --max_iter 240
