#!/usr/bin/env bash
set -euo pipefail

OUT="experiment_records_ik_optimize/20260714005200_p1_pt13_same_branch_gated_refine"
STATE="DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity/A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"
GATE="experiment_records_ik_optimize/20260713225202_a1_three_point_gated_small_regression/channel_gate_a1_three_point.csv"

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
  --outer-max-iter 20 \
  --per-point-timeout-s 1800 \
  --step-I-max 0.125 \
  --step-x-max 0.002 \
  --initial-I-A="$I" \
  --initial-x-m "$X" \
  --initial-branch-state-npz "$STATE" \
  --channel-gate-file "$GATE" \
  --channel-gate-missing-policy block_all \
  --p2-trust-region \
  --p2-terminal-error-first \
  --p2-reliable-subspace \
  --p2-max-backtracks 3 2>&1 | tee "$OUT/stdout.log"

XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
-m inverse_kinematics.verify_branch_states \
DELIVER_TO_EXP/formal_experiments/20260714005016_A1_A2_5mm_supervised_IK_no_gravity/A1/smoke/pt_13 \
  --out-json "$OUT/branch_reload_verify.json" \
  --out-csv "$OUT/branch_reload_verify.csv"

