#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 13 \
  --outer-max-iter 20 \
  --per-point-timeout-s 1800 \
  --step-I-max 0.125 \
  --step-x-max 0.002 \
  --initial-I-A=-6.0430248096,-6.88902396133,-0.796161564456,1.40175091772,2.24072763687,-1.89509877415,5.19731017528,9.82618724372 \
  --initial-x-m 0.13179801344 \
  --x-sensitivity-method implicit \
  --sensitivity-mode residual_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9 \
  --sensitivity-spectral-filter hard \
  --sensitivity-spectral-damping 0 \
  --ridge-adjoint 0 \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-reliable-task-rcond 1e-3 \
  --p2-min-full-task-rank 2 \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-disable-Ix-jacobian-below-dx 0.001 \
  --channel-gate-file experiment_records_ik_optimize/20260713225202_a1_three_point_gated_small_regression/channel_gate_a1_three_point.csv \
  --channel-gate-missing-policy block_all
