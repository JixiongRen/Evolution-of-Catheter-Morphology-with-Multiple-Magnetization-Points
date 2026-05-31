#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda run -n evolution_of_mcrs python inverse_kinematics/tests/test_observation_interface.py
conda run -n evolution_of_mcrs python -m inverse_kinematics.tests.test_residual_interface \
  --flex_lengths=0.044,0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --M_list=1,1,1 \
  --L_protrude=0.11 \
  --coil_currents=0,0,0,0,0,0,0,0
conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/run_observation_probe.py \
  > experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/observation_probe_stdout.log \
  2> experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/observation_probe_stderr.log

conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/run_task_sensitivity_probe.py \
  > experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/task_sensitivity_probe_stdout.log \
  2> experiment_records_ik_optimize/20260531204120_ik_pose_observation_probe/task_sensitivity_probe_stderr.log
