#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260705001000_p2c170_n2_ill_conditioned_direction_audit/run_p2c170_n2_ill_conditioned_direction_audit.py
