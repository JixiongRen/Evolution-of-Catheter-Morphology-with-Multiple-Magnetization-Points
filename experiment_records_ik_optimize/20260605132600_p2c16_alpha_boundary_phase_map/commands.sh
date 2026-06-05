#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260605132600_p2c16_alpha_boundary_phase_map/run_p2c16_alpha_boundary_phase_map.py \
  --alphas 0.35,0.375,0.40,0.425,0.45 \
  --eps_alpha 0.01 \
  --position_scale 60 \
  --boundary_scale 0.5 \
  --slack_scale 20 \
  --steps_per_flex 4 \
  --max_iter 240
