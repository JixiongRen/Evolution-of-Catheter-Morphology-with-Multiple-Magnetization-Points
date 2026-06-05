#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260605153000_p2c18_lower_alpha_breakpoint_load_audit/run_p2c18_lower_alpha_breakpoint_load_audit.py \
  --alphas 0.384,0.382,0.380,0.378,0.376 \
  --eps_alpha 0.01 \
  --position_scale 60 \
  --boundary_scale 0.5 \
  --slack_scale 20 \
  --steps_per_flex 4 \
  --max_iter 240
