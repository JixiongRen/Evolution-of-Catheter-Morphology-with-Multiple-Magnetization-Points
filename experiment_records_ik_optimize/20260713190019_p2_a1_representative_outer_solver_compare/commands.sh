#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

PY=/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python
SCRIPT=DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
OUT=experiment_records_ik_optimize/20260713190019_p2_a1_representative_outer_solver_compare

COMMON=(
  --disable-gravity
  --eps-p-mm 5
  --force-formal-20
  --catheters A1
  --target-indices 1,12,13
  --outer-max-iter 15
  --per-point-timeout-s 1800
  --x-sensitivity-method implicit
  --sensitivity-mode residual_svd
  --sensitivity-solve-mode svd
  --sensitivity-svd-rcond 1e-9
  --sensitivity-spectral-filter hard
  --sensitivity-spectral-damping 0.0
)

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
"${PY}" "${SCRIPT}" \
  "${COMMON[@]}" \
  > "${OUT}/baseline_stdout.log" \
  2> "${OUT}/baseline_stderr.log"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
"${PY}" "${SCRIPT}" \
  "${COMMON[@]}" \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-reliable-task-rcond 1e-3 \
  --p2-min-full-task-rank 2 \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-disable-Ix-jacobian-below-dx 0.001 \
  > "${OUT}/p2_stdout.log" \
  2> "${OUT}/p2_stderr.log"
