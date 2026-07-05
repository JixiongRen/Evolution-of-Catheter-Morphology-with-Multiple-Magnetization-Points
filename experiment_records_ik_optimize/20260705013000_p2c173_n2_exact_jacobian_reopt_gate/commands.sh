#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260705013000_p2c173_n2_exact_jacobian_reopt_gate/run_p2c173_n2_exact_jacobian_reopt_gate.py
