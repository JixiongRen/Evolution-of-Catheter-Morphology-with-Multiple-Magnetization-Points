#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630180000_p2c162_n2_rolling_two_step_prototype/run_p2c162_n2_rolling_two_step_prototype.py
