#!/usr/bin/env bash
set -uo pipefail

ROOT="experiment_records_ik_optimize/20260713203859_p1_clean_failed_channel_reaudit"
PY="/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python"
SCRIPT="DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py"
COMMON_ENV="XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"
COMMON_ARGS="--current-steps-a 0.0625,0.03125,0.015625 --x-steps-mm 0.5,0.25,0.125 --solve-mode svd --ridge 0 --svd-rcond 1e-9 --spectral-filter hard --spectral-damping 0 --terminal-gate-mm 0.25 --centerline-gate-mm 0.5 --weak-displacement-mm 0.25"
STATUS_FILE="${ROOT}/batch_status.tsv"

mkdir -p "${ROOT}"
printf "label\tselector\tsegments\texit_code\tstarted\tfinished\n" > "${STATUS_FILE}"

run_case() {
  local label="$1"
  local point_dir="$2"
  local channels="$3"
  local selector="$4"
  local segments="$5"
  local out_dir="${ROOT}/${label}_${selector}_seg${segments}"
  local started
  local finished
  local exit_code
  started="$(date '+%Y-%m-%d %H:%M:%S')"
  mkdir -p "${out_dir}"
  eval "${COMMON_ENV} ${PY} ${SCRIPT} ${point_dir} --out-dir ${out_dir} --selector ${selector} --current-channels ${channels} --roundtrip-segments ${segments} ${COMMON_ARGS}" \
    > "${out_dir}/stdout.log" \
    2> "${out_dir}/stderr.log"
  exit_code=$?
  finished="$(date '+%Y-%m-%d %H:%M:%S')"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${selector}" "${segments}" "${exit_code}" "${started}" "${finished}" >> "${STATUS_FILE}"
}

A1_PT13="DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_13"
A2_PT05="DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_05"
A2_PT17="DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_17"
A2_PT19="DELIVER_TO_EXP/formal_experiments/20260713141233_A1_A2_5mm_supervised_IK_no_gravity/A2/formal_20pt/pt_19"

for selector in last_accept best_accept; do
  for segments in 1 4; do
    run_case "A1_pt13_ch3_4" "${A1_PT13}" "3,4" "${selector}" "${segments}"
    run_case "A2_pt05_ch1_3" "${A2_PT05}" "1,3" "${selector}" "${segments}"
    run_case "A2_pt17_ch4" "${A2_PT17}" "4" "${selector}" "${segments}"
    run_case "A2_pt19_ch5" "${A2_PT19}" "5" "${selector}" "${segments}"
  done
done
