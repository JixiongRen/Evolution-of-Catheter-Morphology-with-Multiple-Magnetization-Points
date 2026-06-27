#!/usr/bin/env bash
set -euo pipefail
REPO="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$REPO/experiment_records_fk_optimize/20260615_fk_single_tip_axial_coils1_4_15A_tol1e6"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib
COMMON=(
  --config="$REPO/experiment_records_fk_optimize/20260613_forward_kinematics_fk_legacy_gravity_single_coil/config_single_tip_axial_coil6_20A.json"
  --flex_lengths=0.041,0.03,0.03
  --rigid_lengths=0.003,0.003,0.003
  --M_list=5,5,5
  --p0=0,0,0
  --Q0=0.70710678,0,0.70710678,0
  --axis_body=0,0,1
  --enable_gravity
  --g_world=0,0,-9.81
  --max_iter=20000
  --tol=1e-6
  --lm_damping=0.1
  --jac_method=fwd
  --enable_magnetics
  --calib_file="$REPO/calib/mpem_calibration_file_sp=40_order=1.yaml"
  --actuation_table_pkl="$REPO/supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl"
  --m_body_list="0,0,0; 0,0,0; 0,0,0.005301"
  --callback_every=20
  --callback_save_every=20
)
run_case() {
  local name="$1"
  local currents="$2"
  local case_dir="$OUT/$name"
  mkdir -p "$case_dir/callback_snaps"
  printf '{\n  "case_name": "%s",\n  "coil_currents": "%s",\n  "tol": 1e-6,\n  "max_iter": 20000,\n  "m_body_list": "0,0,0; 0,0,0; 0,0,0.005301"\n}\n' "$name" "$currents" > "$case_dir/case_config.json"
  echo "[run] $name $(date -Is) currents=$currents"
  "$PY" "$REPO/forward_kinematics/fk.py" \
    "${COMMON[@]}" \
    --coil_currents="$currents" \
    --callback_dir="$case_dir/callback_snaps" \
    --final_npz="$case_dir/final_solution.npz" \
    > "$case_dir/console.log" 2>&1
  echo "[done] $name $(date -Is)"
}
run_case coil1_15A "15,0,0,0,0,0,0,0"
run_case coil2_15A "0,15,0,0,0,0,0,0"
run_case coil3_15A "0,0,15,0,0,0,0,0"
run_case coil4_15A "0,0,0,15,0,0,0,0"
