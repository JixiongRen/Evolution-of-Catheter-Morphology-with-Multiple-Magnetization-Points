#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_12 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_05 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_17 \
DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19 \
--out-dir experiment_records_ik_optimize/20260713165000_p1_six_point_residual_svd_gate \
--selector best_accept \
--current-channels auto \
--auto-current-channels 2 \
--current-steps-a 0.25,0.125,0.0625 \
--x-steps-mm 0.5,0.25,0.125 \
--sensitivity-modes residual_svd \
--solve-mode svd \
--svd-rcond 1e-9 \
--spectral-filter hard \
--spectral-damping 0.0 \
--max-points 6
