#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606103000_p2c19_ultrafine_lower_alpha_breakpoint/run_p2c19_ultrafine_lower_alpha_breakpoint.py \
  --alphas 0.3758,0.3756,0.3754,0.3752,0.3750 \
  --eps_alpha 0.01 \
  --position_scale 60 \
  --boundary_scale 0.5 \
  --slack_scale 20 \
  --steps_per_flex 4 \
  --max_iter 240
