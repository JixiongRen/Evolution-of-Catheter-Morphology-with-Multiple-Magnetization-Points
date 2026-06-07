#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606194500_p2c30_rand20_multi_eps_recheck/run_p2c30_rand20_multi_eps_recheck.py \
  > experiment_records_ik_optimize/20260606194500_p2c30_rand20_multi_eps_recheck/stdout_stderr.log \
  2>&1
