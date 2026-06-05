#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605121500_p2c12_position_weight_compatibility_scan/run_p2c12_position_weight_compatibility_scan.py \
  --alphas 0.40,0.45 \
  --position_scales 15,30,60,120,240 \
  --max_iter 180
