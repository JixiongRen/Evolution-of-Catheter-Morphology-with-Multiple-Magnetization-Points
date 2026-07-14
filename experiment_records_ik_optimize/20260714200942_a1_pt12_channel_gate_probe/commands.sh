#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="experiment_records_ik_optimize/20260714200942_a1_pt12_channel_gate_probe"
source "$EXP_DIR/pt13_seed.env"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 12 \
  --per-point-timeout-s 1800 \
  --outer-max-iter 25 \
  --initial-I-A="$INITIAL_I_A" \
  --initial-x-m "$INITIAL_X_M" \
  --initial-branch-state-npz "$INITIAL_BRANCH_STATE_NPZ" \
  --x-sensitivity-method implicit \
  --sensitivity-mode gn_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9 \
  --sensitivity-spectral-filter hard \
  --sensitivity-spectral-damping 0 \
  --ridge-adjoint 1e-10 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-reliable-task-rcond 0.001 \
  --p2-min-full-task-rank 2 \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-disable-Ix-jacobian-below-dx 0.001 \
  --channel-gate-file "$EXP_DIR/a1_pt12_channel_gate.csv" \
  --channel-gate-missing-policy block_all \
  --channel-gate-stage refine
