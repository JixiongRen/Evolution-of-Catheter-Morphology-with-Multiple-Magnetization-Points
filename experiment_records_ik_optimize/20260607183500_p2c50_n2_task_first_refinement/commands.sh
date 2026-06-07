#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607183500_p2c50_n2_task_first_refinement/run_p2c50_n2_task_first_refinement.py
