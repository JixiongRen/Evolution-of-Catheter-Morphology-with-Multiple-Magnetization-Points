#!/usr/bin/env bash
set -euo pipefail

OUT="experiment_records_ik_optimize/20260714004300_p1_pt13_reload_ablation"
STATE="DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"

I="$(/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
import numpy as np
path = "DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"
data = np.load(path)
print(",".join(f"{float(v):.12g}" for v in data["I_A"].reshape(-1)))
PY
)"

X="$(/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
import numpy as np
path = "DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"
data = np.load(path)
print(f"{float(data['x_m'].reshape(-1)[0]):.12g}")
PY
)"

run_case() {
  local name="$1"
  shift
  echo "[RUN] $name" | tee -a "$OUT/stdout.log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
    --disable-gravity \
    --eps-p-mm 5 \
    --force-formal-20 \
    --catheters A1 \
    --target-indices 13 \
    --outer-max-iter 0 \
    --per-point-timeout-s 900 \
    --step-I-max 0.125 \
    --step-x-max 0.002 \
    --p2-trust-region \
    --p2-terminal-error-first \
    --p2-reliable-subspace \
    --p2-max-backtracks 3 \
    "$@" 2>&1 | tee -a "$OUT/stdout.log"
}

run_case cold_start
run_case control_only --initial-I-A="$I" --initial-x-m "$X"
run_case state_only --initial-branch-state-npz "$STATE"
run_case control_and_state --initial-I-A="$I" --initial-x-m "$X" --initial-branch-state-npz "$STATE"

