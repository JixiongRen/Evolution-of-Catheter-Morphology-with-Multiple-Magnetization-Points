#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605211000_p2c10y_position_refinement_step_audit/run_p2c10y_position_refinement_step_audit.py \
  --case n3_mixed_rand20 \
  --alphas 0.40,0.45 \
  --eps_values 0.005,0.01,0.02 \
  --max_iters 100,250,500
