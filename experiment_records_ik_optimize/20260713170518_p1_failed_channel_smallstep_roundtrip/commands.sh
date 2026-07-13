#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

ROOT="experiment_records_ik_optimize/20260713170518_p1_failed_channel_smallstep_roundtrip"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
AUDIT="DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py"
COMMON_ARGS=(
  --selector best_accept
  --current-steps-a 0.0625,0.03125,0.015625
  --x-steps-mm ""
  --sensitivity-modes residual_svd
  --solve-mode svd
  --ridge 1e-10
  --svd-rcond 1e-9
  --spectral-filter hard
  --spectral-damping 0
  --terminal-gate-mm 0.25
  --centerline-gate-mm 0.5
  --weak-displacement-mm 0.25
)
COMMON_ARGS_RCOND1E10=(
  --selector best_accept
  --current-steps-a 0.0625,0.03125,0.015625
  --x-steps-mm ""
  --sensitivity-modes residual_svd
  --solve-mode svd
  --ridge 1e-10
  --svd-rcond 1e-10
  --spectral-filter hard
  --spectral-damping 0
  --terminal-gate-mm 0.25
  --centerline-gate-mm 0.5
  --weak-displacement-mm 0.25
)

run_audit() {
  local name="$1"
  local path="$2"
  local channels="$3"
  local segments="$4"
  mkdir -p "${ROOT}/${name}"
  "${PY}" "${AUDIT}" "${path}" \
    --out-dir "${ROOT}/${name}" \
    --current-channels "${channels}" \
    --roundtrip-segments "${segments}" \
    "${COMMON_ARGS[@]}" \
    > "${ROOT}/${name}/stdout.log" 2> "${ROOT}/${name}/stderr.log"
}

run_audit "seg1_A1_pt13_ch3_ch4" \
  "DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13" \
  "3,4" 1
run_audit "seg1_A2_pt05_ch1_ch3" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_05" \
  "1,3" 1
run_audit "seg1_A2_pt17_ch4" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_17" \
  "4" 1
run_audit "seg1_A2_pt19_ch5" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19" \
  "5" 1

run_audit "seg4_A1_pt13_ch3_ch4" \
  "DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13" \
  "3,4" 4
run_audit "seg4_A2_pt05_ch1_ch3" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_05" \
  "1,3" 4
run_audit "seg4_A2_pt17_ch4" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_17" \
  "4" 4
run_audit "seg4_A2_pt19_ch5" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19" \
  "5" 4

mkdir -p "${ROOT}/seg4_A2_pt19_ch5_rcond1e-10"
"${PY}" "${AUDIT}" \
  "DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19" \
  --out-dir "${ROOT}/seg4_A2_pt19_ch5_rcond1e-10" \
  --current-channels "5" \
  --roundtrip-segments 4 \
  "${COMMON_ARGS_RCOND1E10[@]}" \
  > "${ROOT}/seg4_A2_pt19_ch5_rcond1e-10/stdout.log" \
  2> "${ROOT}/seg4_A2_pt19_ch5_rcond1e-10/stderr.log"
