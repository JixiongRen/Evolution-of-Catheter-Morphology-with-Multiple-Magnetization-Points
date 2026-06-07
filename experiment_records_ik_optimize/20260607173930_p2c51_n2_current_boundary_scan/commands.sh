#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607173930_p2c51_n2_current_boundary_scan/run_p2c51_n2_current_boundary_scan.py \
  > experiment_records_ik_optimize/20260607173930_p2c51_n2_current_boundary_scan/stdout.log \
  2> experiment_records_ik_optimize/20260607173930_p2c51_n2_current_boundary_scan/stderr.log
