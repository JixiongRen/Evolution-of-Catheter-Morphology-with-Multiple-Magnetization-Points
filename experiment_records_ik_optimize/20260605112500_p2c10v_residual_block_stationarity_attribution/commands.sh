#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605112500_p2c10v_residual_block_stationarity_attribution/run_p2c10v_residual_block_stationarity_attribution.py
