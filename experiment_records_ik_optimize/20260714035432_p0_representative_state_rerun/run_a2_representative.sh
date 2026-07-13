#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A2 \
  --target-indices 5,17,19 \
  --outer-max-iter 50 \
  --per-point-timeout-s 2700 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --sensitivity-mode residual_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9

