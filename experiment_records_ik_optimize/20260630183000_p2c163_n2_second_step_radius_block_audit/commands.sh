#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630183000_p2c163_n2_second_step_radius_block_audit/run_p2c163_n2_second_step_radius_block_audit.py
