#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606170041_p2c25_scaled_stationarity_and_interface_audit/run_p2c25_scaled_stationarity_and_interface_audit.py \
  > experiment_records_ik_optimize/20260606170041_p2c25_scaled_stationarity_and_interface_audit/stdout_stderr.log \
  2>&1
