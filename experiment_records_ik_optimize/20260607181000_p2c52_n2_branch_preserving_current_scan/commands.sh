#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607181000_p2c52_n2_branch_preserving_current_scan/run_p2c52_n2_branch_preserving_current_scan.py \
  > experiment_records_ik_optimize/20260607181000_p2c52_n2_branch_preserving_current_scan/stdout.log \
  2> experiment_records_ik_optimize/20260607181000_p2c52_n2_branch_preserving_current_scan/stderr.log
