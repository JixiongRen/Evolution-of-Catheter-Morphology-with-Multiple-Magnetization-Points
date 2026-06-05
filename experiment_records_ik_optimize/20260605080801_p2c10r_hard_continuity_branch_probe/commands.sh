#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605080801_p2c10r_hard_continuity_branch_probe/run_p2c10r_hard_continuity_branch_probe.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_base_i4:1.00 \
  --anchor_mu 1e-6 \
  --constraint_scale 1.0 \
  --max_iter 80
