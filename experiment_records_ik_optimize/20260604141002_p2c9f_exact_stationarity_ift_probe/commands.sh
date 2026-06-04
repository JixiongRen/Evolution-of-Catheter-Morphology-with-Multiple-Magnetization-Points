#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604141002_p2c9f_exact_stationarity_ift_probe/run_p2c9f_exact_stationarity_ift_probe.py \
  --cases n3_mixed_base_i4,n2_z_pair_rand12,n1_tip_z_rand15,n3_mixed_rand20 \
  --eps 0.02,0.05,0.10 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24 \
  --random_count 1
