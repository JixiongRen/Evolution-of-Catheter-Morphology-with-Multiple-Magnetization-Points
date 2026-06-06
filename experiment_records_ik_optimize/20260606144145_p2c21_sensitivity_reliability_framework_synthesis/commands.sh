#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606144145_p2c21_sensitivity_reliability_framework_synthesis/run_p2c21_sensitivity_reliability_framework_synthesis.py \
  > experiment_records_ik_optimize/20260606144145_p2c21_sensitivity_reliability_framework_synthesis/stdout_stderr.log \
  2>&1
