#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Syntax check for the experiment-local diagnostic script.
python -m py_compile \
  experiment_records_ik_optimize/20260601093012_p2b_shooting_sensitivity/run_p2b_shooting_sensitivity.py

# Observation residual interface regression.
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_observation_interface

# FK residual parameter interface regression.
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_residual_interface \
  --flex_lengths=0.044,0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --M_list=1,1,1 \
  --L_protrude=0.11 \
  --coil_currents=0,0,0,0,0,0,0,0

# P2b non-production shooting sensitivity audit.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601093012_p2b_shooting_sensitivity/run_p2b_shooting_sensitivity.py \
  --mode full \
  --M_list 1,1,1 \
  --max_iter 160 \
  --tol 1e-5 \
  --jac_eps 0.02 \
  --audit_eps 0.02,0.05 \
  > experiment_records_ik_optimize/20260601093012_p2b_shooting_sensitivity/full_stdout.log \
  2> experiment_records_ik_optimize/20260601093012_p2b_shooting_sensitivity/full_stderr.log
