#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# First pass: physical wrench candidates c0-c6.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_p2c1_convergence.py \
  --max_iter 120 \
  --tol 1e-5 \
  --steps_per_flex 4 \
  --candidate_max_nfev 12 \
  > experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_stdout.log \
  2> experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_stderr.log

# Preserve the first pass before testing geometric twist modes.
cp experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/summary.json \
  experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/summary_first7.json
cp experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/summary.csv \
  experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/summary_first7.csv

# Second pass: non-production segment-twist geometry modes c7-c8.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_p2c1_convergence.py \
  --max_iter 120 \
  --tol 1e-5 \
  --steps_per_flex 4 \
  --candidate_max_nfev 16 \
  --candidate_names c7_segment_twist_centerline,c8_segment_twist_soft_l1 \
  > experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_twist_stdout.log \
  2> experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_twist_stderr.log

# Final merge uses endpoint-first scoring after the tip/terminal-pose priority clarification.

# Lightweight fixed-state finite-difference audit for the best physical candidate.
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/run_p2c1_fd_audit.py \
  > experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/fd_audit_stdout.log \
  2> experiment_records_ik_optimize/20260601110758_p2c1_physical_shooting_convergence/fd_audit_stderr.log
