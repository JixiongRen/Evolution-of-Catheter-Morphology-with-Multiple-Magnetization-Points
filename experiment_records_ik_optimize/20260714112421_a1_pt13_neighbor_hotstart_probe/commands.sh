#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

PT01_I="0.7376872701709714,-4.947495433964284,-1.1431887099933202,4.1258558095441416,1.5868114245211706,-3.9922855991738446,-0.6520868323262559,4.31645655646366"
PT01_X="0.13209663485598064"

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 13 \
  --outer-max-iter 50 \
  --per-point-timeout-s 1200 \
  --initial-I-A="${PT01_I}" \
  --initial-x-m "${PT01_X}" \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 3

