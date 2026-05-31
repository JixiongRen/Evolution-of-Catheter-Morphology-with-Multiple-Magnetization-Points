#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Syntax check.
python -m py_compile \
  inverse_kinematics/ik_diff.py \
  experiment_records_ik_optimize/20260531221959_p2_task_sensitivity_audit/run_p2_task_sensitivity_audit.py

# Observation residual interface regression.
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_observation_interface

# Residual parameter interface regression.
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_residual_interface \
  --flex_lengths=0.044,0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --M_list=1,1,1 \
  --L_protrude=0.11 \
  --coil_currents=0,0,0,0,0,0,0,0

# P2 finite-difference audit.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260531221959_p2_task_sensitivity_audit/run_p2_task_sensitivity_audit.py \
  --mode quick \
  --max_iter 120 \
  --tol 1e-5 \
  --tol_good 1e-3 \
  > experiment_records_ik_optimize/20260531221959_p2_task_sensitivity_audit/quick_stdout.log \
  2> experiment_records_ik_optimize/20260531221959_p2_task_sensitivity_audit/quick_stderr.log
