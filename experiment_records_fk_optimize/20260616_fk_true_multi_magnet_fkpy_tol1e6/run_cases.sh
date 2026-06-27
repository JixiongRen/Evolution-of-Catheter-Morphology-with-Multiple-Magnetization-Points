#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$REPO/experiment_records_fk_optimize/20260616_fk_true_multi_magnet_fkpy_tol1e6"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib

COMMON=(
  --p0=0,0,0
  --Q0=0.70710678,0,0.70710678,0
  --axis_body=0,0,1
  --enable_gravity
  --g_world=0,0,-9.81
  --enable_magnetics
  --calib_file="$REPO/calib/mpem_calibration_file_sp=40_order=1.yaml"
  --actuation_table_pkl="$REPO/supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl"
  --max_iter=12000
  --tol=1e-6
  --lm_damping=0.1
  --jac_method=fwd
  --callback_every=20
  --callback_save_every=20
)

run_case() {
  local name="$1"
  local label="$2"
  local flex_lengths="$3"
  local rigid_lengths="$4"
  local m_list="$5"
  local currents="$6"
  local m_body_list="$7"
  local total_length="$8"
  local case_dir="$OUT/$name"
  mkdir -p "$case_dir/callback_snaps"
  cat > "$case_dir/case_config.json" <<JSON
{
  "case_name": "$name",
  "label": "$label",
  "source_script": "forward_kinematics/fk.py",
  "flex_lengths": "$flex_lengths",
  "rigid_lengths": "$rigid_lengths",
  "M_list": "$m_list",
  "total_length_m": $total_length,
  "coil_currents": "$currents",
  "m_body_list": "$m_body_list",
  "tol": 1e-6,
  "max_iter": 12000,
  "callback_every": 20,
  "physical_note": "No inactive zero-moment magnetic segment is included; segment count equals magnetic segment count."
}
JSON
  echo "[run] $name $(date -Is) currents=$currents"
  "$PY" "$REPO/forward_kinematics/fk.py" \
    "${COMMON[@]}" \
    --flex_lengths="$flex_lengths" \
    --rigid_lengths="$rigid_lengths" \
    --M_list="$m_list" \
    --coil_currents="$currents" \
    --m_body_list="$m_body_list" \
    --callback_dir="$case_dir/callback_snaps" \
    --final_npz="$case_dir/final_solution.npz" \
    > "$case_dir/console.log" 2>&1
  echo "[done] $name $(date -Is)"
}

mkdir -p "$OUT"

run_case \
  "n2_z_pair_rand_a" \
  "真实双磁体段-z向配对" \
  "0.041,0.03" \
  "0.003,0.003" \
  "5,5" \
  "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4" \
  "0,0,-0.005301;0,0,0.005301" \
  "0.077"

run_case \
  "n2_xz_pair_rand_b" \
  "真实双磁体段-x/z组合" \
  "0.041,0.03" \
  "0.003,0.003" \
  "5,5" \
  "-19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2" \
  "0.005301,0,0;0,0,0.005301" \
  "0.077"

run_case \
  "n3_xyz_rand_a" \
  "真实三磁体段-xyz组合" \
  "0.041,0.03,0.03" \
  "0.003,0.003,0.003" \
  "5,5,5" \
  "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4" \
  "0.005301,0,0;0,0.005301,0;0,0,0.005301" \
  "0.11"
