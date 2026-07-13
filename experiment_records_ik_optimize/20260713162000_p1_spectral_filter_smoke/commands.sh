#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
--out-dir experiment_records_ik_optimize/20260713162000_p1_spectral_filter_smoke \
--selector best_accept \
--current-channels 0 \
--current-steps-a 0.25 \
--x-steps-mm "" \
--sensitivity-modes residual_svd \
--solve-mode svd \
--svd-rcond 1e-8 \
--spectral-filter tikhonov \
--spectral-damping 0.0 \
--max-points 1
