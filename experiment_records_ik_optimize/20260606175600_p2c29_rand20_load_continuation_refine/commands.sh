#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/mcrs_mpl_cache

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606175600_p2c29_rand20_load_continuation_refine/run_p2c29_rand20_load_continuation_refine.py \
  > experiment_records_ik_optimize/20260606175600_p2c29_rand20_load_continuation_refine/stdout_stderr.log \
  2>&1
