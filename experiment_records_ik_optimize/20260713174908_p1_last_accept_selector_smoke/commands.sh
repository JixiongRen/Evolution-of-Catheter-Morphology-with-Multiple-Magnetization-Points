#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
  --out-dir experiment_records_ik_optimize/20260713174908_p1_last_accept_selector_smoke \
  --selector last_accept \
  --current-steps-a 0.0625 \
  --x-steps-mm "" \
  --current-channels 3 \
  --sensitivity-modes residual_svd \
  --solve-mode svd \
  --svd-rcond 1e-9 \
  --spectral-filter hard \
  --spectral-damping 0.0 \
  --roundtrip-segments 1 \
  --terminal-gate-mm 0.25 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.25
