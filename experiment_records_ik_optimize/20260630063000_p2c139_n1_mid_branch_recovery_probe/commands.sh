#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630063000_p2c139_n1_mid_branch_recovery_probe/run_p2c139_n1_mid_branch_recovery_probe.py \
  > experiment_records_ik_optimize/20260630063000_p2c139_n1_mid_branch_recovery_probe/stdout.log \
  2> experiment_records_ik_optimize/20260630063000_p2c139_n1_mid_branch_recovery_probe/stderr.log
