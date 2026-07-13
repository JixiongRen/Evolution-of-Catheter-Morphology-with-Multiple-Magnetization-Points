#!/usr/bin/env bash
set -euo pipefail

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  DELIVER_TO_EXP/formal_experiments/20260713191352_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13 \
  --out-dir experiment_records_ik_optimize/20260713224415_a1_pt13_residual_svd_fine_reaudit/audit \
  --selector best_accept \
  --current-steps-a 0.0625,0.03125 \
  --x-steps-mm 0.125,0.0625 \
  --current-channels 4,5,6 \
  --sensitivity-modes residual_svd \
  --solve-mode svd \
  --ridge 1e-10 \
  --svd-rcond 1e-9 \
  --spectral-filter hard \
  --spectral-damping 0.0 \
  --roundtrip-segments 8 \
  --terminal-gate-mm 0.1 \
  --centerline-gate-mm 0.5 \
  --weak-displacement-mm 0.05
