#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="experiment_records_ik_optimize/20260714195941_a1_pt12_sensitivity_audit"
POINT_DIR="DELIVER_TO_EXP/formal_experiments/20260714192447_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_12"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  "$POINT_DIR" \
  --out-dir "$EXP_DIR/audit" \
  --selector last_accept \
  --current-steps-a 0.25,0.125 \
  --x-steps-mm 0.5,0.25 \
  --current-channels auto \
  --auto-current-channels 2 \
  --sensitivity-modes solver_response,gn_svd,stationarity_exact \
  --solve-mode svd \
  --ridge 1e-10 \
  --svd-rcond 1e-9 \
  --spectral-filter hard \
  --spectral-damping 0 \
  --roundtrip-segments 3 \
  --terminal-gate-mm 0.25 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.10 \
  --health-gate warn \
  --point-timeout-s 1800
