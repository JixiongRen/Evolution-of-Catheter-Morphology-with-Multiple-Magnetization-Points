#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$ROOT/experiment_records_fk_optimize/20260526220659_fk_magnet_current_m555_validation"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

/usr/bin/time -v "$PY" "$OUT/run_fk_magnet_current_m555_validation.py" \
  > "$OUT/stdout.log" \
  2> "$OUT/stderr_time.log"
