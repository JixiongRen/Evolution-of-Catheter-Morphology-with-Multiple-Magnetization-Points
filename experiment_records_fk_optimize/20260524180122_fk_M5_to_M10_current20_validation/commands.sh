#!/usr/bin/env bash
set -euo pipefail

BASE="experiment_records_fk_optimize/20260524180122_fk_M5_to_M10_current20_validation"

XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/time -v \
  -o "$BASE/batch_time.txt" \
  conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$BASE/run_fk_M5_to_M10_validation.py" \
  > "$BASE/batch_stdout.log" \
  2> "$BASE/batch_stderr.log"
