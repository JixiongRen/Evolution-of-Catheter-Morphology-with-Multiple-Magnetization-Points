#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604114419_p2c8_continuity_branch_sensitivity/run_p2c8_continuity_branch_sensitivity.py \
  --quick \
  --eps 0.05 \
  --modes reg_only,cont_0.01,cont_0.1,cont_1.0 \
  --max_iter 24

# Final focused rerun used for the committed summary.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604114419_p2c8_continuity_branch_sensitivity/run_p2c8_continuity_branch_sensitivity.py \
  --quick \
  --eps 0.05 \
  --modes cont_0.1,cont_1.0 \
  --max_iter 24 \
  --random_count 1
