#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714142837_A1_A2_5mm_supervised_IK_no_gravity/A1/smoke/pt_13 \
  --out-dir /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/experiment_records_ik_optimize/20260714144638_p1_a1_pt13_local_sensitivity_audit/audit_output \
  --selector best_accept \
  --sensitivity-modes solver_response,gn_svd,stationarity_exact \
  --solve-mode svd \
  --svd-rcond 1e-9 \
  --ridge 1e-10 \
  --current-steps-a 0.25,0.125,0.0625 \
  --x-steps-mm 0.5,0.25,0.125 \
  --current-channels auto \
  --auto-current-channels 3 \
  --roundtrip-segments 2 \
  --terminal-gate-mm 0.25 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.25 \
  --health-gate warn \
  --point-timeout-s 2400
