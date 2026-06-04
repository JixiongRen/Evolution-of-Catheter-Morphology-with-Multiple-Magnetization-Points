#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260604154000_p2c9h_n3_rand20_baseline_recovery/run_p2c9h_n3_rand20_baseline_recovery.py \
  --case n3_mixed_rand20 \
  --neighbor n3_mixed_base_i4 \
  --scan_iter 16 \
  --refine_iter 48 \
  --top_k 3
