#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
PY=/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python
RUNNER="$ROOT/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py"
OUT="$ROOT/experiment_records_ik_optimize/20260714092026_p1_a1_three_point_causal_matrix"
GATE="$ROOT/experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/audit_output/a1_representative_channel_gate.csv"
STATUS="$OUT/status.csv"

mkdir -p "$OUT/logs"
cd "$ROOT"

printf "case,exit_code,summary,report,log\n" > "$STATUS"

COMMON=(
  --disable-gravity
  --eps-p-mm 5
  --catheters A1
  --target-indices 1,12,13
  --outer-max-iter 50
  --per-point-timeout-s 2700
  --step-I-max 5
  --step-x-max 0.02
  --x-sensitivity-method implicit
  --sensitivity-mode gn_svd
  --sensitivity-solve-mode svd
  --sensitivity-svd-rcond 1e-9
  --sensitivity-spectral-filter hard
  --sensitivity-spectral-damping 0
  --ridge-adjoint 1e-10
  --p2-trust-region
  --p2-terminal-error-first
  --p2-reliable-subspace
  --p2-reliable-task-rcond 0.001
  --p2-min-full-task-rank 2
  --p2-max-backtracks 3
  --p2-backtrack-shrink 0.5
  --p2-disable-Ix-jacobian-below-dx 0.001
  --force-formal-20
)

run_case() {
  local name="$1"
  shift
  local log="$OUT/logs/${name}.log"
  echo "[RUN] $name" | tee -a "$OUT/progress.log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda "$PY" "$RUNNER" "${COMMON[@]}" "$@" >"$log" 2>&1
  local code=$?
  local summary
  local report
  summary=$(grep -E '^summary:' "$log" | tail -1 | sed 's/^summary:[[:space:]]*//' || true)
  report=$(grep -E '^report:' "$log" | tail -1 | sed 's/^report:[[:space:]]*//' || true)
  printf "%s,%s,%s,%s,%s\n" "$name" "$code" "$summary" "$report" "$log" >> "$STATUS"
  echo "[DONE] $name exit=$code summary=$summary" | tee -a "$OUT/progress.log"
  return 0
}

run_case p2_no_gate
run_case p2_refine_gate \
  --channel-gate-file "$GATE" \
  --channel-gate-missing-policy block_all \
  --channel-gate-stage refine
run_case p2_basin_gate \
  --channel-gate-file "$GATE" \
  --channel-gate-missing-policy block_all \
  --channel-gate-stage basin
run_case p2_two_stage_gate \
  --channel-gate-file "$GATE" \
  --channel-gate-missing-policy block_all \
  --two-stage-gate

"$PY" "$OUT/summarize.py" "$STATUS" > "$OUT/matrix_summary.json"

