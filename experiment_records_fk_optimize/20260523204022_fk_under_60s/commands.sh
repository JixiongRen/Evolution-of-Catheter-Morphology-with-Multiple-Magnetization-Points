#!/usr/bin/env bash
set -euo pipefail

BASE="experiment_records_fk_optimize/20260523204022_fk_under_60s"

# Smoke test: import, fixed-case metadata, no callback/no plot path.
OUT="$BASE/smoke_direct_warmup_10iter"
/usr/bin/time -v -o "$OUT/time.txt" conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$OUT" --case_name smoke_direct_warmup_10iter --mode direct \
  --max_iter 10 --tol 1e-5 --scale_mode optimized --scale_include_external --warmup \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"

# Main continuation probes.
OUT="$BASE/fk_current_continuation_probe_a"
/usr/bin/time -v -o "$OUT/time.txt" conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$OUT" --case_name fk_current_continuation_probe_a --mode continuation \
  --classification "speed-only result" --scale_mode optimized --scale_include_external --warmup \
  --current_factors 0,0.2,0.4,0.6,0.8,1.0 --stage_max_iter 120 --stage_tol 1e-3 \
  --final_max_iter 800 --final_tol 1e-5 \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"

# Best strict max-iteration geometry probe under the 60s steady budget.
OUT="$BASE/fk_short_continuation_final1150"
/usr/bin/time -v -o "$OUT/time.txt" conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$OUT" --case_name fk_short_continuation_final1150 --mode continuation \
  --classification "physically comparable result" --scale_mode optimized --scale_include_external --warmup \
  --lm_damping 0.3 --current_factors 0,1.0 --stage_max_iter 60 --stage_tol 1e-3 \
  --final_max_iter 1150 --final_tol 1e-5 \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"

# Final candidate with a loose residual gate and mandatory centerline comparison.
OUT="$BASE/fk_under_60s_final_candidate_tol15e4"
/usr/bin/time -v -o "$OUT/time.txt" conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$OUT" --case_name fk_under_60s_final_candidate_tol15e4 --mode continuation \
  --classification "production-candidate result" --scale_mode optimized --scale_include_external --warmup \
  --lm_damping 0.3 --current_factors 0,1.0 --stage_max_iter 60 --stage_tol 1e-3 \
  --final_max_iter 1150 --final_tol 1.5e-4 \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
