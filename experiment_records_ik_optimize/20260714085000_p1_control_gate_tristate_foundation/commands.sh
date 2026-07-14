#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  -m py_compile \
  inverse_kinematics/ik_control_gate.py \
  inverse_kinematics/tests/test_control_gate.py \
  inverse_kinematics/ik_multisite_lm_opt.py

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from inverse_kinematics.ik_control_gate import load_control_gate

path = "experiment_records_ik_optimize/20260714070559_p1b_representative_sensitivity_audit/audit_output/a1_representative_channel_gate.csv"
for key in ["A1_pt01", "A1_pt12", "A1_pt13"]:
    gate = load_control_gate(path, key=key, n_coils=8, allow_limited=True)
    print(key, "refine=", gate.as_dict()["allowed_channels"], "basin=", gate.as_dict()["basin_channels"])
PY
