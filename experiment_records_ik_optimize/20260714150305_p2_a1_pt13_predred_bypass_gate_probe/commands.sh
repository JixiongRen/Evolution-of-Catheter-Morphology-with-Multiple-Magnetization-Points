#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 13 \
  --per-point-timeout-s 1800 \
  --outer-max-iter 25 \
  --initial-I-A=2.3471809420634786,1.762028118912526,-0.17097792041998483,-0.3790754115817867,0.0298130825658432,0.36836739879629105,-1.506297366433078,-2.444014832550359 \
  --initial-x-m=0.10088380625805202 \
  --initial-branch-state-npz=/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714142837_A1_A2_5mm_supervised_IK_no_gravity/A1/smoke/pt_13/branch_states/states/eval_000005_trial_iter_0302.npz \
  --x-sensitivity-method implicit \
  --sensitivity-mode gn_svd \
  --sensitivity-solve-mode svd \
  --sensitivity-svd-rcond 1e-9 \
  --sensitivity-spectral-filter hard \
  --sensitivity-spectral-damping 0 \
  --ridge-adjoint 1e-10 \
  --step-I-max 5 \
  --step-x-max 0.02 \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-reliable-task-rcond 0.001 \
  --p2-min-full-task-rank 2 \
  --p2-max-backtracks 3 \
  --p2-backtrack-shrink 0.5 \
  --p2-disable-Ix-jacobian-below-dx 0.001 \
  --channel-gate-file /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/experiment_records_ik_optimize/20260714150305_p2_a1_pt13_predred_bypass_gate_probe/a1_pt13_local_gate.csv \
  --channel-gate-missing-policy block_all \
  --channel-gate-stage refine \
  --channel-gate-allow-limited
