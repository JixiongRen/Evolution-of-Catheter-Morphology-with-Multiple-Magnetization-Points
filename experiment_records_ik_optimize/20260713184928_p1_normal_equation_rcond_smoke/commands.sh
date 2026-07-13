#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19 \
  --out-dir experiment_records_ik_optimize/20260713184928_p1_normal_equation_rcond_smoke \
  --selector last_accept \
  --current-channels 5 \
  --current-steps-a 0.125 \
  --x-steps-mm 0.5 \
  --sensitivity-modes solver_response,gn_svd,stationarity_exact,residual_svd \
  --solve-mode svd \
  --svd-rcond 1e-9 \
  --spectral-filter hard \
  --spectral-damping 0.0 \
  --roundtrip-segments 1 \
  --terminal-gate-mm 0.25 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.25
