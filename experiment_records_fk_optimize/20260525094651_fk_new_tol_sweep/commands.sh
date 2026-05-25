#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
OUT_DIR="$ROOT/experiment_records_fk_optimize/20260525094651_fk_new_tol_sweep"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"

cd "$ROOT"
mkdir -p "$OUT_DIR"

env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MPLBACKEND=Agg \
  PYTHONPATH="$ROOT" \
  /usr/bin/time -f 'WALL_SECONDS=%e
MAX_RSS_KB=%M' -o "$OUT_DIR/time.txt" \
  "$PY" "$OUT_DIR/run_tol_sweep.py" \
  --out_dir "$OUT_DIR" \
  > "$OUT_DIR/stdout.log" \
  2> "$OUT_DIR/stderr.log"
