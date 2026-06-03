#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/run_multipoint_magnet_pose_probe.py \
  > experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/stdout.log \
  2> experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/stderr.log

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/run_neighbor_warmstart_probe.py \
  > experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/neighbor_stdout.log \
  2> experiment_records_ik_optimize/20260603222805_p2c6_multipoint_magnet_pose_probe/neighbor_stderr.log
