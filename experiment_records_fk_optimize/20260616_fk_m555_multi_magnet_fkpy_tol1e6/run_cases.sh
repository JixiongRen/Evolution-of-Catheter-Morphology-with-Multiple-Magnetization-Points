#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$REPO/experiment_records_fk_optimize/20260616_fk_m555_multi_magnet_fkpy_tol1e6"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib

COMMON=(
  --flex_lengths=0.041,0.03,0.03
  --rigid_lengths=0.003,0.003,0.003
  --M_list=5,5,5
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
  local currents="$2"
  local m_body_list="$3"
  local label="$4"
  local case_dir="$OUT/$name"
  mkdir -p "$case_dir/callback_snaps"
  cat > "$case_dir/case_config.json" <<JSON
{
  "case_name": "$name",
  "label": "$label",
  "source_script": "forward_kinematics/fk.py",
  "flex_lengths": "0.041,0.03,0.03",
  "rigid_lengths": "0.003,0.003,0.003",
  "M_list": "5,5,5",
  "total_length_m": 0.11,
  "coil_currents": "$currents",
  "m_body_list": "$m_body_list",
  "tol": 1e-6,
  "max_iter": 12000,
  "callback_every": 20
}
JSON
  echo "[run] $name $(date -Is) currents=$currents"
  "$PY" "$REPO/forward_kinematics/fk.py" \
    "${COMMON[@]}" \
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
  "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4" \
  "0,0,-0.005301;0,0,0.005301;0,0,0" \
  "双磁化点-z向配对"

run_case \
  "n2_xz_pair_rand_b" \
  "-19.3,7.1,-4.8,16.6,-10.9,2.3,13.7,-8.2" \
  "0.005301,0,0;0,0,0;0,0,0.005301" \
  "双磁化点-x/z组合"

run_case \
  "n3_xyz_rand_a" \
  "11.2,-6.7,18.4,-15.1,3.6,9.8,-12.5,5.4" \
  "0.005301,0,0;0,0.005301,0;0,0,0.005301" \
  "三磁化点-xyz组合"
