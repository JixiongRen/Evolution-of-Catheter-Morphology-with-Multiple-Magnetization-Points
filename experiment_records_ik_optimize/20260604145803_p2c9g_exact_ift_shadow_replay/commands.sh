#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260604145803_p2c9g_exact_ift_shadow_replay/run_p2c9g_exact_ift_shadow_replay.py \
  --cases n3_mixed_base_i4,n2_z_pair_rand12,n1_tip_z_rand15,n3_mixed_rand20 \
  --step 0.05 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24 \
  --random_count 1
