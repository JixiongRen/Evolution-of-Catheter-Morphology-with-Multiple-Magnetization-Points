#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260604215800_p2c10b_break_cause_audit/run_p2c10b_break_cause_audit.py \
  --base_case n3_mixed_base_i4 \
  --target_case n3_mixed_rand20 \
  --alphas 0.20,0.25 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24 \
  --steps_per_flex 4
