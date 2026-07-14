#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  -m py_compile inverse_kinematics/ik_multisite_lm_opt.py

git diff -- inverse_kinematics/ik_multisite_lm_opt.py
