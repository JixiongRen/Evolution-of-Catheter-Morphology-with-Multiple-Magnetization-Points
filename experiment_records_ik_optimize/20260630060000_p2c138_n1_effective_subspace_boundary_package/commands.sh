#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630060000_p2c138_n1_effective_subspace_boundary_package/run_p2c138_n1_effective_subspace_boundary_package.py \
  > experiment_records_ik_optimize/20260630060000_p2c138_n1_effective_subspace_boundary_package/stdout.log \
  2> experiment_records_ik_optimize/20260630060000_p2c138_n1_effective_subspace_boundary_package/stderr.log
