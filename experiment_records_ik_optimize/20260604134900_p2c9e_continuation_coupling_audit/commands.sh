#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604134900_p2c9e_continuation_coupling_audit/run_p2c9e_continuation_coupling_audit.py \
  --cases n3_mixed_base_i4,n3_mixed_rand20 \
  --continuity_mu 0,1e-4,1e-2,1e0,1e2 \
  --reg_mu 1.0e-4 \
  --max_iter 80
