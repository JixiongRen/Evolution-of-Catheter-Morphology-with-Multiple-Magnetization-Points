#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607181500_p2c49_n2_explicit_continuity_continuation/run_p2c49_n2_explicit_continuity_continuation.py
