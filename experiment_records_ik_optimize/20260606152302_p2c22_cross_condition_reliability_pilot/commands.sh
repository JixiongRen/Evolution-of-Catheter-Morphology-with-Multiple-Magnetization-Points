#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606152302_p2c22_cross_condition_reliability_pilot/run_p2c22_cross_condition_reliability_pilot.py \
  > experiment_records_ik_optimize/20260606152302_p2c22_cross_condition_reliability_pilot/stdout_stderr.log \
  2>&1
