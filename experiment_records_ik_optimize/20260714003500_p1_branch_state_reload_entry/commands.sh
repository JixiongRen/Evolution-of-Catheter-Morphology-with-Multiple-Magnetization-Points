#!/usr/bin/env bash
set -euo pipefail

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
  --initial-I-A="$I" \
  --initial-x-m "$X" \
  --initial-branch-state-npz "$STATE" \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 3

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
-m inverse_kinematics.verify_branch_states \
DELIVER_TO_EXP/formal_experiments/20260714003850_A1_A2_5mm_supervised_IK_no_gravity/A1/smoke/pt_13 \
  --out-json experiment_records_ik_optimize/20260714003500_p1_branch_state_reload_entry/branch_reload_verify.json \
  --out-csv experiment_records_ik_optimize/20260714003500_p1_branch_state_reload_entry/branch_reload_verify.csv

