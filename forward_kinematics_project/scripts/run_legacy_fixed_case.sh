#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/outputs/legacy_fixed_case"

mkdir -p "$OUT"
cd "$ROOT"

export PYTHONPATH="$ROOT:$ROOT/forward_kinematics_legacy:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PY" forward_kinematics_legacy/fk.py \
  --flex_lengths=0.03,0.03 \
  --rigid_lengths=0.003,0.003,0.003 \
  --L_protrude=0.11 \
  --M_list=5,5,5 \
  --enable_gravity \
  --enable_magnetics \
  --calib_file="$ROOT/calib/mpem_calibration_file_sp=40_order=1.yaml" \
  --actuation_table_pkl="$ROOT/supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl" \
  --coil_currents=0,0,0,0,0,50,0,0 \
  --m_body_list="0,0,-0.005301; 0,0,0.005301; 0.005301,0,0" \
  --p0=0,0,0 \
  --Q0=0.70710678,0,0.70710678,0 \
  --axis_body=0,0,1 \
  --plot \
  > "$OUT/stdout.log" \
  2> "$OUT/stderr.log"
