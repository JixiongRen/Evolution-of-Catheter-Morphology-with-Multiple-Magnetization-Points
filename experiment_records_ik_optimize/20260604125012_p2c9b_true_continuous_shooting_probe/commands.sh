#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604125012_p2c9b_true_continuous_shooting_probe/run_p2c9b_true_continuous_shooting_probe.py \
  --eps 0.05 \
  --reg_mu 1.0e-4 \
  --max_iter 80 \
  --random_count 1
