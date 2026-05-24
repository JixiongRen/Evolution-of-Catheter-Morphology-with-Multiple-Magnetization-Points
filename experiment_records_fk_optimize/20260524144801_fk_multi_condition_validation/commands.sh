#!/usr/bin/env bash
set -euo pipefail

BASE="experiment_records_fk_optimize/20260524144801_fk_multi_condition_validation"

XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs \
  python "$BASE/plot_pre_post_fk.py" \
  > "$BASE/pre_post_plot_stdout.log" \
  2> "$BASE/pre_post_plot_stderr.log"

XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/time -v \
  -o "$BASE/multi_condition_time.txt" \
  conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$BASE/run_multi_condition_fk.py" \
  > "$BASE/multi_condition_stdout.log" \
  2> "$BASE/multi_condition_stderr.log"
