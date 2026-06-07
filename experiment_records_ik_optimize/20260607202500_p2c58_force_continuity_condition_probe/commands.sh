#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607202500_p2c58_force_continuity_condition_probe/run_p2c58_force_continuity_condition_probe.py \
  > experiment_records_ik_optimize/20260607202500_p2c58_force_continuity_condition_probe/stdout.log \
  2> experiment_records_ik_optimize/20260607202500_p2c58_force_continuity_condition_probe/stderr.log
