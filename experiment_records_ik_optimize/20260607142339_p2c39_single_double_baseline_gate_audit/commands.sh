#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/matplotlib_p2c39
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260607142339_p2c39_single_double_baseline_gate_audit/run_p2c39_single_double_baseline_gate_audit.py \
  > experiment_records_ik_optimize/20260607142339_p2c39_single_double_baseline_gate_audit/stdout_stderr.log \
  2>&1
