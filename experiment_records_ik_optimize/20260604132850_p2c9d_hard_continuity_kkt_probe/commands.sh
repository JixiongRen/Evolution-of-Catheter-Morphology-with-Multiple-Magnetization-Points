#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604132850_p2c9d_hard_continuity_kkt_probe/run_p2c9d_hard_continuity_kkt_probe.py \
  --cases n3_mixed_base_i4,n3_mixed_rand20 \
  --eps 0.02,0.05,0.10 \
  --anchor_mu 1.0e-6 \
  --constraint_scale 1.0 \
  --max_iter 80
