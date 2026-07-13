#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --catheters A1 \
  --target-indices 1 \
  --force-formal-20 \
  --outer-max-iter 2 \
  --per-point-timeout-s 900 \
  --sensitivity-mode residual_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9 \
  --sensitivity-spectral-filter hard \
  --sensitivity-spectral-damping 0 \
  --x-sensitivity-method implicit \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-reliable-task-rcond 1e-3 \
  --p2-min-full-task-rank 2 \
  --p2-disable-Ix-jacobian-below-dx 0.001
