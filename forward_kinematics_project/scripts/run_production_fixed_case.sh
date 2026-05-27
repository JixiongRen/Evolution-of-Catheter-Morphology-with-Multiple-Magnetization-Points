#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/outputs/production_fixed_case"

mkdir -p "$OUT"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PY" experiments/efficiency_optimization/fk_under_60s_driver.py \
  --out_dir "$OUT" \
  --case_name production_fixed_case \
  --mode direct \
  --classification "production-candidate result" \
  --coil_currents "0,0,0,0,0,50,0,0" \
  --m_body_list "0,0,-0.005301; 0,0,0.005301; 0.005301,0,0" \
  --M_list "5,5,5" \
  --max_iter 100000 \
  --tol 1e-5 \
  --lm_damping 0.1 \
  --jac_method fwd \
  --linear_solver dense_fused_cholesky \
  --dense_solve_method cholesky \
  --scale_mode optimized \
  --scale_include_external \
  --warmup
