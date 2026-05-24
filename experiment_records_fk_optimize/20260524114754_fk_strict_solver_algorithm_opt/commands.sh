#!/usr/bin/env bash
set -euo pipefail

BASE="experiment_records_fk_optimize/20260524114754_fk_strict_solver_algorithm_opt"

# Representative strict baseline.
mkdir -p "$BASE/fk_driver_lm03_solve_strict_baseline"
XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/time -v \
  -o "$BASE/fk_driver_lm03_solve_strict_baseline/time.txt" \
  conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$BASE/fk_driver_lm03_solve_strict_baseline" \
  --case_name fk_driver_lm03_solve_strict_baseline \
  --mode continuation \
  --classification "strict baseline" \
  --scale_mode optimized \
  --scale_include_external \
  --warmup \
  --tol 1e-5 \
  --lm_damping 0.3 \
  --damping_mode identity \
  --dense_solve_method solve \
  --current_factors 0,1.0 \
  --stage_max_iter 20 \
  --stage_tol 1e-3 \
  --final_max_iter 2500 \
  --final_tol 1e-5 \
  > "$BASE/fk_driver_lm03_solve_strict_baseline/stdout.log" \
  2> "$BASE/fk_driver_lm03_solve_strict_baseline/stderr.log"

# Final production-candidate result.
mkdir -p "$BASE/fk_pos_lm01_shortstage_final1450_production"
XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/time -v \
  -o "$BASE/fk_pos_lm01_shortstage_final1450_production/time.txt" \
  conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$BASE/fk_pos_lm01_shortstage_final1450_production" \
  --case_name fk_pos_lm01_shortstage_final1450_production \
  --mode continuation \
  --classification "production-candidate result" \
  --scale_mode optimized \
  --scale_include_external \
  --warmup \
  --tol 1e-5 \
  --lm_damping 0.1 \
  --damping_mode identity \
  --dense_solve_method pos \
  --current_factors 0,1.0 \
  --stage_max_iter 20 \
  --stage_tol 1e-3 \
  --final_max_iter 1450 \
  --final_tol 1e-5 \
  > "$BASE/fk_pos_lm01_shortstage_final1450_production/stdout.log" \
  2> "$BASE/fk_pos_lm01_shortstage_final1450_production/stderr.log"

# Parameter sweep executed in one Python process to reuse JAX compilation.
XLA_PYTHON_CLIENT_PREALLOCATE=false /usr/bin/time -v \
  -o "$BASE/single_process_sweep_time.txt" \
  conda run -n evolution_of_mcrs env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$BASE/run_single_process_sweep.py" \
  > "$BASE/single_process_sweep_stdout.log" \
  2> "$BASE/single_process_sweep_stderr.log"

# Aggregate summaries and geometry comparisons.
conda run -n evolution_of_mcrs python "$BASE/aggregate_results.py"
