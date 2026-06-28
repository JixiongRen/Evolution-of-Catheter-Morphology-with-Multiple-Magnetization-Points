#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630023000_p2c134_n2_trusted_branch_basin_radius/run_p2c134_n2_trusted_branch_basin_radius.py \
  > experiment_records_ik_optimize/20260630023000_p2c134_n2_trusted_branch_basin_radius/stdout.log \
  2> experiment_records_ik_optimize/20260630023000_p2c134_n2_trusted_branch_basin_radius/stderr.log
