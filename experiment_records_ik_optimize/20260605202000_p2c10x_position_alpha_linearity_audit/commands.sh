#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605202000_p2c10x_position_alpha_linearity_audit/run_p2c10x_position_alpha_linearity_audit.py \
  --case n3_mixed_rand20 \
  --alphas 0.30,0.35,0.40,0.45,0.50 \
  --eps_alpha 0.01 \
  --max_iter 100
