#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="experiment_records_ik_optimize/20260714201739_a1_pt13_to_pt12_angle_continuation"
source "$EXP_DIR/pt13_seed.env"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_angle_continuation_experiment.py \
  --out-dir "$EXP_DIR/run_data" \
  --catheter A1 \
  --start-index 13 \
  --end-index 12 \
  --segments 4 \
  --eps-p-mm 5 \
  --outer-max-iter 25 \
  --per-point-timeout-s 1200 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --initial-I-A="$INITIAL_I_A" \
  --initial-x-m "$INITIAL_X_M" \
  --initial-branch-state-npz "$INITIAL_BRANCH_STATE_NPZ" \
  --stop-on-failure
