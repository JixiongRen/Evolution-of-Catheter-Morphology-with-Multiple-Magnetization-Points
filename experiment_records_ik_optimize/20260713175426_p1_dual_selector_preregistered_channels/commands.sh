#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

PY=/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python
SCRIPT=DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py
OUT=experiment_records_ik_optimize/20260713175426_p1_dual_selector_preregistered_channels
A1_SRC=DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt
A2_SRC=DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt

run_audit() {
  local label="$1"
  shift
  mkdir -p "${OUT}/${label}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_PLATFORMS=cuda \
  "${PY}" "${SCRIPT}" "$@" \
    > "${OUT}/${label}/stdout.log" \
    2> "${OUT}/${label}/stderr.log"
}

COMMON=(
  --current-steps-a 0.25,0.125,0.0625
  --x-steps-mm 0.5,0.25,0.125
  --sensitivity-modes solver_response,gn_svd,stationarity_exact,residual_svd
  --solve-mode svd
  --svd-rcond 1e-9
  --spectral-filter hard
  --spectral-damping 0.0
  --roundtrip-segments 1
  --terminal-gate-mm 0.25
  --centerline-gate-mm 0.5
  --weak-displacement-mm 0.25
)

for selector in last_accept best_accept; do
  run_audit "${selector}_a1_pts01_12_13_ch3_4" \
    "${A1_SRC}/pt_01" "${A1_SRC}/pt_12" "${A1_SRC}/pt_13" \
    --out-dir "${OUT}/${selector}_a1_pts01_12_13_ch3_4" \
    --selector "${selector}" \
    --current-channels 3,4 \
    "${COMMON[@]}"

  run_audit "${selector}_a2_pt05_ch1_3" \
    "${A2_SRC}/pt_05" \
    --out-dir "${OUT}/${selector}_a2_pt05_ch1_3" \
    --selector "${selector}" \
    --current-channels 1,3 \
    "${COMMON[@]}"

  run_audit "${selector}_a2_pt17_ch4" \
    "${A2_SRC}/pt_17" \
    --out-dir "${OUT}/${selector}_a2_pt17_ch4" \
    --selector "${selector}" \
    --current-channels 4 \
    "${COMMON[@]}"

  run_audit "${selector}_a2_pt19_ch5" \
    "${A2_SRC}/pt_19" \
    --out-dir "${OUT}/${selector}_a2_pt19_ch5" \
    --selector "${selector}" \
    --current-channels 5 \
    "${COMMON[@]}"
done
