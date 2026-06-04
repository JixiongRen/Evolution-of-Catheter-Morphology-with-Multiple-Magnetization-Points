#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260604164000_p2c9j_target_current_continuation/run_p2c9j_target_current_continuation.py \
  --base_case n3_mixed_base_i4 \
  --target_case n3_mixed_rand20 \
  --alphas 0,0.2,0.4,0.6,0.8,1.0 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24
