#!/usr/bin/env bash
set -euo pipefail

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --catheters A1 \
  --target-indices 1 \
  --outer-max-iter 1 \
  --per-point-timeout-s 900 \
  --force-formal-20 \
  > experiment_records_ik_optimize/20260713194050_p0_production_traceability_smoke/stdout.log \
  2> experiment_records_ik_optimize/20260713194050_p0_production_traceability_smoke/stderr.log
