#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605040948_p2c10n_position_medium_current_scan/run_p2c10n_position_medium_current_scan.py \
  --cases n1_tip_z_rand15,n2_z_pair_rand12,n3_mixed_rand20,n3_mixed_base_i4 \
  --alphas 0.25,0.50,0.75,1.00 \
  --step 0.05 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 32 \
  --random_count 1
