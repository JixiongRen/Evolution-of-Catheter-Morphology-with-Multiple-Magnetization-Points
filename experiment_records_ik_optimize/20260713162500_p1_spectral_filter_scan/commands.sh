#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
POINTS=(
  DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01
  DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_12
)
run_case() {
  local name="$1"
  local filter="$2"
  local damping="$3"
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  "${POINTS[@]}" \
  --out-dir "experiment_records_ik_optimize/20260713162500_p1_spectral_filter_scan/${name}" \
  --selector best_accept \
  --current-channels 0,4 \
  --current-steps-a 0.25 \
  --x-steps-mm "" \
  --sensitivity-modes residual_svd \
  --solve-mode svd \
  --svd-rcond 1e-8 \
  --spectral-filter "${filter}" \
  --spectral-damping "${damping}" \
  --max-points 2 \
  > "experiment_records_ik_optimize/20260713162500_p1_spectral_filter_scan/${name}_stdout.log" \
  2> "experiment_records_ik_optimize/20260713162500_p1_spectral_filter_scan/${name}_stderr.log"
}
run_case hard_rcond1e-8 hard 0.0
run_case hybrid_lam1e-6 hybrid 1e-6
run_case tikhonov_lam1e-6 tikhonov 1e-6
