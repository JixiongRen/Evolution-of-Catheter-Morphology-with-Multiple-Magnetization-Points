#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606172200_p2c27_stationarity_refinement_gate/run_p2c27_stationarity_refinement_gate.py \
  > experiment_records_ik_optimize/20260606172200_p2c27_stationarity_refinement_gate/stdout_stderr.log \
  2>&1
