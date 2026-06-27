#!/usr/bin/env bash
set -euo pipefail
REPO="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$REPO/experiment_records_fk_optimize/20260615_fk_legacy_scripts_gravity_tol1e7"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
COMMON=(
  --config="$REPO/experiment_records_fk_optimize/20260613_forward_kinematics_fk_legacy_gravity_single_coil/config_gravity_only.json"
  --flex_lengths=0.041,0.03,0.03
  --rigid_lengths=0.003,0.003,0.003
  --M_list=5,5,5
  --p0=0,0,0
  --Q0=0.70710678,0,0.70710678,0
  --axis_body=0,0,1
  --enable_gravity
  --g_world=0,0,-9.81
  --max_iter=20000
  --tol=1e-7
  --lm_damping=0.1
  --jac_method=fwd
)
echo "[run] fk.py $(date -Is)"
"$PY" "$REPO/forward_kinematics/fk.py" \
  "${COMMON[@]}" \
  --callback_every=20 \
  --callback_save_every=20 \
  --callback_dir="$OUT/fk_py/callback_snaps" \
  --final_npz="$OUT/fk_py/final_solution.npz" \
  > "$OUT/fk_py/console.log" 2>&1
echo "[done] fk.py $(date -Is)"
echo "[run] fk_example.py $(date -Is)"
"$PY" "$REPO/forward_kinematics/fk_example.py" \
  "${COMMON[@]}" \
  > "$OUT/fk_example_py/console.log" 2>&1
echo "[done] fk_example.py $(date -Is)"
