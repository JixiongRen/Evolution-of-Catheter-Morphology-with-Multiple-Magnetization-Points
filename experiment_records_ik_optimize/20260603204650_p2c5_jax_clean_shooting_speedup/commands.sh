#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260603204650_p2c5_jax_clean_shooting_speedup/run_jax_clean_shooting.py \
  > experiment_records_ik_optimize/20260603204650_p2c5_jax_clean_shooting_speedup/stdout.log \
  2> experiment_records_ik_optimize/20260603204650_p2c5_jax_clean_shooting_speedup/stderr.log
