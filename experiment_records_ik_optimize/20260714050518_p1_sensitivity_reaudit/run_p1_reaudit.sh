#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  DELIVER_TO_EXP/formal_experiments/20260714035640_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression \
  DELIVER_TO_EXP/formal_experiments/20260714041425_A1_A2_5mm_supervised_IK_no_gravity/A2/small_regression \
  --out-dir experiment_records_ik_optimize/20260714050518_p1_sensitivity_reaudit/audit_output \
  --selector best_accept \
  --current-channels 0,1,2,3,4,5,6,7 \
  --current-steps-a 0.25,0.125,0.0625 \
  --x-steps-mm 0.5,0.25,0.125 \
  --sensitivity-modes solver_response,gn_svd,stationarity_exact \
  --solve-mode svd \
  --ridge 1e-10 \
  --svd-rcond 1e-9 \
  --spectral-filter hard \
  --spectral-damping 0.0 \
  --roundtrip-segments 1 \
  --terminal-gate-mm 0.25 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.25

