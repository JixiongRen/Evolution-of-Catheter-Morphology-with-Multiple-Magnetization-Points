#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630100000_p2c147_n2_principal_radius_bracket/run_p2c147_n2_principal_radius_bracket.py \
  > experiment_records_ik_optimize/20260630100000_p2c147_n2_principal_radius_bracket/stdout.log \
  2> experiment_records_ik_optimize/20260630100000_p2c147_n2_principal_radius_bracket/stderr.log
