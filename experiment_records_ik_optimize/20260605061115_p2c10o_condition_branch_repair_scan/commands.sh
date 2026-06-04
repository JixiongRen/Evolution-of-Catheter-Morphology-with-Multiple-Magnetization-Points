#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605061115_p2c10o_condition_branch_repair_scan/run_p2c10o_condition_branch_repair_scan.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00 \
  --continuity_mus 1.0,3.0 \
  --reg_mus 1e-5,1e-4,1e-3 \
  --step 0.05 \
  --max_iter 32 \
  --random_count 1
