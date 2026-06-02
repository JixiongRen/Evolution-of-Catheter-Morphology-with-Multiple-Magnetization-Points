#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Core benchmark: legacy c3 vs terminal-pose c3. FD audit is skipped here
# because the complete audit is expensive and is run by the second command.
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/run_p2c2_terminal_pose.py \
  --skip_fd_audit \
  > experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/run_skip_fd_stdout.log \
  2> experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/run_skip_fd_stderr.log

# Re-optimized finite-difference audit for the terminal-pose candidate.
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/run_p2c2_fd_reopt.py \
  > experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/fd_reopt_small_stdout.log \
  2> experiment_records_ik_optimize/20260602103750_p2c2_terminal_pose_constraint/fd_reopt_small_stderr.log
