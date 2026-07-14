#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
RUNNER="$REPO/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py"
OUT_DIR="$REPO/DELIVER_TO_EXP/formal_experiments"
EXP_DIR="$REPO/experiment_records_ik_optimize/20260714073700_p2_a1_three_point_controlled_validation"
GATE="$REPO/experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/audit_output/a1_representative_channel_gate.csv"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

cd "$REPO"

echo "controlled_rerun_start=$(date --iso-8601=seconds)" | tee -a "$EXP_DIR/run_status.txt"
echo "controlled_rerun_git_commit=$(git rev-parse HEAD)" | tee -a "$EXP_DIR/run_status.txt"
"$PY" "$RUNNER" \
  --disable-gravity \
  --eps-p-mm 5 \
  --catheters A1 \
  --target-indices 1,12,13 \
  --outer-max-iter 50 \
  --per-point-timeout-s 2700 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --x-sensitivity-method implicit \
  --sensitivity-mode gn_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9 \
  --sensitivity-spectral-filter hard \
  --sensitivity-spectral-damping 0 \
  --ridge-adjoint 1e-10 \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-reliable-task-rcond 0.001 \
  --p2-min-full-task-rank 2 \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-disable-Ix-jacobian-below-dx 0.001 \
  --channel-gate-file "$GATE" \
  --channel-gate-missing-policy block_all \
  --channel-gate-allow-limited \
  --force-formal-20
CONTROLLED_RERUN_ROOT="$(find "$OUT_DIR" -maxdepth 1 -type d -name '*A1_A2_5mm_supervised_IK_no_gravity' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
echo "$CONTROLLED_RERUN_ROOT" > "$EXP_DIR/controlled_rerun_formal_run_root.txt"
echo "controlled_rerun_root=$CONTROLLED_RERUN_ROOT" | tee -a "$EXP_DIR/run_status.txt"
echo "controlled_rerun_end=$(date --iso-8601=seconds)" | tee -a "$EXP_DIR/run_status.txt"
