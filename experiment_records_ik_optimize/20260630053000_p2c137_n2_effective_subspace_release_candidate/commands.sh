#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630053000_p2c137_n2_effective_subspace_release_candidate/run_p2c137_n2_effective_subspace_release_candidate.py \
  > experiment_records_ik_optimize/20260630053000_p2c137_n2_effective_subspace_release_candidate/stdout.log \
  2> experiment_records_ik_optimize/20260630053000_p2c137_n2_effective_subspace_release_candidate/stderr.log
