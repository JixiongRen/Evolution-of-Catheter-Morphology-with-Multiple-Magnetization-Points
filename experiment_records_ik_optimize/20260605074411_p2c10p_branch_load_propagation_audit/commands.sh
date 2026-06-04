#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605074411_p2c10p_branch_load_propagation_audit/run_p2c10p_branch_load_propagation_audit.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 32
