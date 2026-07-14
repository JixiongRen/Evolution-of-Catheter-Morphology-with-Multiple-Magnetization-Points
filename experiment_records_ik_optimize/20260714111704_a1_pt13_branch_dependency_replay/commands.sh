#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

NEAR_I="-6.043024850460061,-6.889024022822658,-0.796161555180463,1.4017510098979562,2.240727667826086,-1.895098819886132,5.197310238294989,9.82618735440505"
NEAR_X="0.13179801339136246"
NEAR_BRANCH="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260714092428_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 13 \
  --outer-max-iter 0 \
  --per-point-timeout-s 600 \
  --initial-I-A="${NEAR_I}" \
  --initial-x-m "${NEAR_X}" \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 1

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 13 \
  --outer-max-iter 0 \
  --per-point-timeout-s 600 \
  --initial-I-A="${NEAR_I}" \
  --initial-x-m "${NEAR_X}" \
  --initial-branch-state-npz "${NEAR_BRANCH}" \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 1

