#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
OUT="$ROOT/experiment_records_fk_optimize/20260613_forward_kinematics_fk_legacy_gravity_single_coil"

export PYTHONPATH="$ROOT:$ROOT/forward_kinematics:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

COMMON_ARGS=(
  --flex_lengths=0.041,0.03,0.03
  --rigid_lengths=0.003,0.003,0.003
  --M_list=5,5,5
  --p0=0,0,0
  --Q0=0.70710678,0,0.70710678,0
  --axis_body=0,0,1
  --enable_gravity
  --g_world=0,0,-9.81
  --max_iter=6000
  --tol=1e-5
  --lm_damping=0.1
  --jac_method=fwd
)

CALIB="$ROOT/calib/mpem_calibration_file_sp=40_order=1.yaml"
TABLE="$ROOT/supiee_auto_diff/offline_interpolation_data/actuation_tables/actuation_table.pkl"
SINGLE_COIL_ARGS=(
  --enable_magnetics
  --calib_file="$CALIB"
  --actuation_table_pkl="$TABLE"
  --coil_currents=0,0,0,0,0,20,0,0
  "--m_body_list=0,0,0; 0,0,0; 0,0,0.005301"
)

run_case() {
  local script_name="$1"
  local case_name="$2"
  local config_name="$3"
  shift 3

  local case_dir="$OUT/$case_name"
  mkdir -p "$case_dir"
  cd "$case_dir"

  {
    echo "# $case_name"
    echo "script=$script_name"
    printf "%q " "$PY" "$ROOT/forward_kinematics/$script_name" --config="$OUT/$config_name" "${COMMON_ARGS[@]}" "$@"
    echo
  } > command.txt

  "$PY" "$ROOT/forward_kinematics/$script_name" \
    --config="$OUT/$config_name" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > stdout.log \
    2> stderr.log
}

run_case fk_example.py gravity_only config_gravity_only.json
run_case fk.py gravity_only_with_callback config_gravity_only.json \
  --callback_every=20 \
  --callback_save_every=20 \
  --callback_dir=callback_snaps \
  --final_npz=final_solution.npz

run_case fk_example.py single_tip_axial_coil6_20A config_single_tip_axial_coil6_20A.json "${SINGLE_COIL_ARGS[@]}"
run_case fk.py single_tip_axial_coil6_20A_with_callback config_single_tip_axial_coil6_20A.json \
  "${SINGLE_COIL_ARGS[@]}" \
  --callback_every=20 \
  --callback_save_every=20 \
  --callback_dir=callback_snaps \
  --final_npz=final_solution.npz
