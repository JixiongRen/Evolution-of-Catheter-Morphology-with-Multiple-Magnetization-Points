#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260604222000_p2c10c_rigid_load_coupling_counterfactual/run_p2c10c_rigid_load_coupling_counterfactual.py \
  --base_case n3_mixed_base_i4 \
  --target_case n3_mixed_rand20 \
  --alphas 0.20,0.25 \
  --betas 0.0,0.005,0.01,0.02 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24 \
  --steps_per_flex 4
