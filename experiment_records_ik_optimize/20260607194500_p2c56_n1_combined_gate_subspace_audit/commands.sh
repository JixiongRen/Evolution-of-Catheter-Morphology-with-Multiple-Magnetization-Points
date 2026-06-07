#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607194500_p2c56_n1_combined_gate_subspace_audit/run_p2c56_n1_combined_gate_subspace_audit.py \
  > experiment_records_ik_optimize/20260607194500_p2c56_n1_combined_gate_subspace_audit/stdout.log \
  2> experiment_records_ik_optimize/20260607194500_p2c56_n1_combined_gate_subspace_audit/stderr.log
