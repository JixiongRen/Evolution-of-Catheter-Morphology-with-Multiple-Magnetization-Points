#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py

git diff --check -- \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  --disable-gravity \
  --eps-p-mm 5 \
  --force-formal-20 \
  --catheters A1 \
  --target-indices 1 \
  --outer-max-iter 0 \
  --per-point-timeout-s 600 \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 1

