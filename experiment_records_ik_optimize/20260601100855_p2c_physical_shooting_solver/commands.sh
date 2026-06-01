#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Syntax check.
python -m py_compile \
  experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/run_p2c_physical_shooting_solver.py

# Interface smoke tests.
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_observation_interface
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_residual_interface \
  --flex_lengths=0.044,0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --M_list=1,1,1 \
  --L_protrude=0.11 \
  --coil_currents=0,0,0,0,0,0,0,0

# Base physical shooting replay probe with informed proximal wrench.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/run_p2c_physical_shooting_solver.py \
  --mode quick \
  --M_list 1,1,1 \
  --max_iter 120 \
  --tol 1e-5 \
  --jac_eps 0.02 \
  --audit_eps 0.02,0.05 \
  --shooting_max_nfev 4 \
  --steps_per_flex 8 \
  --active_current_indices 5 \
  --residual_types position,axis \
  --base_only \
  --use_oracle_proximal_guess \
  > experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/base_stdout.log \
  2> experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/base_stderr.log

# Lightweight finite-difference audit in one active-current subspace.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/run_p2c_physical_shooting_solver.py \
  --mode quick \
  --M_list 1,1,1 \
  --max_iter 120 \
  --tol 1e-5 \
  --jac_eps 0.02 \
  --audit_eps 0.02,0.05 \
  --shooting_max_nfev 4 \
  --steps_per_flex 4 \
  --active_current_indices 5 \
  --residual_types position,axis \
  --fd_reuse_base_y \
  --use_oracle_proximal_guess \
  > experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/quick_stdout.log \
  2> experiment_records_ik_optimize/20260601100855_p2c_physical_shooting_solver/quick_stderr.log
