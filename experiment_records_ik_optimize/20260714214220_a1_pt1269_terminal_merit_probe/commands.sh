#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="experiment_records_ik_optimize/20260714214220_a1_pt1269_terminal_merit_probe"
source "$EXP_DIR/pt1275_seed.env"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_angle_continuation_experiment.py \
  --out-dir "$EXP_DIR/run_data" \
  --catheter A1 \
  --start-index 12.75 \
  --end-index 12.50 \
  --segments 4 \
  --eps-p-mm 5 \
  --outer-max-iter 25 \
  --per-point-timeout-s 1200 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --initial-I-A="$INITIAL_I_A" \
  --initial-x-m "$INITIAL_X_M" \
  --initial-branch-state-npz "$INITIAL_BRANCH_STATE_NPZ" \
  --p2-terminal-only-step \
  --no-p2-reliable-subspace \
  --p2-hold-lam-on-terminal-accept \
  --p2-stall-rescue-max 2 \
  --p2-stall-rescue-lam-factor 0.01 \
  --stop-on-failure
