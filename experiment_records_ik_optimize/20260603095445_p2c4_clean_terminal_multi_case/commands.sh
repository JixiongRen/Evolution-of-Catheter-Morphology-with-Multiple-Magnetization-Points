#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260603095445_p2c4_clean_terminal_multi_case/run_clean_multi_case.py \
  > experiment_records_ik_optimize/20260603095445_p2c4_clean_terminal_multi_case/run_stdout.log \
  2> experiment_records_ik_optimize/20260603095445_p2c4_clean_terminal_multi_case/run_stderr.log
