#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604122310_p2c9_strict_sensitivity_gate/run_p2c9_strict_sensitivity_gate.py \
  --eps 0.02,0.05,0.10 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 24 \
  --random_count 2

conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260604122310_p2c9_strict_sensitivity_gate/run_p2c9_baseline_recovery.py
