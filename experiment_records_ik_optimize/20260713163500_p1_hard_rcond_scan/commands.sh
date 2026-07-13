#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
POINTS=(
  DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_01
  DELIVER_TO_EXP/formal_experiments/20260713135237_A1_A2_5mm_supervised_IK_no_gravity/A1/formal_20pt/pt_12
)
run_case() {
  local name="$1"
  local rcond="$2"
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  "${POINTS[@]}" \
  --out-dir "experiment_records_ik_optimize/20260713163500_p1_hard_rcond_scan/${name}" \
  --selector best_accept \
  --current-channels 0,4 \
  --current-steps-a 0.25 \
  --x-steps-mm "" \
  --sensitivity-modes residual_svd \
  --solve-mode svd \
  --svd-rcond "${rcond}" \
  --spectral-filter hard \
  --spectral-damping 0.0 \
  --max-points 2 \
  > "experiment_records_ik_optimize/20260713163500_p1_hard_rcond_scan/${name}_stdout.log" \
  2> "experiment_records_ik_optimize/20260713163500_p1_hard_rcond_scan/${name}_stderr.log"
}
run_case rcond1e-6 1e-6
run_case rcond1e-9 1e-9
run_case rcond1e-10 1e-10
