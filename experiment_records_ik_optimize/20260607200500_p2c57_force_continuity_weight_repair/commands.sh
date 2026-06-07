#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/run_p2c57_force_continuity_weight_repair.py \
  > experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/stdout.log \
  2> experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/stderr.log
