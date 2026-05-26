#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
OUT_DIR="$ROOT/experiment_records_fk_optimize/20260526104245_fk_branch_preserving_speedup"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"

cd "$ROOT"
mkdir -p "$OUT_DIR"

env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MPLBACKEND=Agg \
  PYTHONPATH="$ROOT" \
  /usr/bin/time -f 'WALL_SECONDS=%e
MAX_RSS_KB=%M' -o "$OUT_DIR/time.txt" \
  "$PY" "$OUT_DIR/run_branch_preserving_speedup.py" \
  --out_dir "$OUT_DIR" \
  > "$OUT_DIR/stdout.log" \
  2> "$OUT_DIR/stderr.log"

"$PY" "$OUT_DIR/generate_convergence_trace.py" \
  > "$OUT_DIR/convergence_trace_stdout.log" \
  2> "$OUT_DIR/convergence_trace_stderr.log"
