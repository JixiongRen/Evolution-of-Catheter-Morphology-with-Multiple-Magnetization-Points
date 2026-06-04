#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605012012_p2c10j_residual_block_condition_attribution/run_p2c10j_residual_block_condition_attribution.py \
  --candidates n2_z_pair_rand12:0.25,n3_mixed_rand20:0.25 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 32
