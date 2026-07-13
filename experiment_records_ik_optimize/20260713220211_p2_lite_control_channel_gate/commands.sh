#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  inverse_kinematics/ik_control_gate.py \
  inverse_kinematics/ik_multisite_lm_opt.py \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  inverse_kinematics/tests/test_control_gate.py

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from inverse_kinematics.ik_control_gate import load_control_gate

with TemporaryDirectory() as d:
    table = Path(d) / "gate.csv"
    table.write_text("\n".join([
        "run,kind,channel,classification",
        "A1_pt13_ch3_4_best_accept_seg1,I,3,同分支误差超限",
        "A1_pt13_ch3_4_best_accept_seg1,I,4,同分支误差超限",
        "A1_pt13_ch3_4_best_accept_seg1,x,-1,可生产使用",
    ]), encoding="utf-8")
    gate = load_control_gate(table, key="A1_pt13", n_coils=8)
    assert gate.allowed_mask == (False, False, False, False, False, False, False, False, True)

with TemporaryDirectory() as d:
    table = Path(d) / "gate.csv"
    table.write_text("\n".join([
        "run,kind,channel,classification",
        "A2_pt17_ch4_last_accept_seg1,I,4,可有限使用",
    ]), encoding="utf-8")
    assert load_control_gate(table, key="A2_pt17", n_coils=8, allow_limited=True).allowed_mask[4]
    assert not load_control_gate(table, key="A2_pt17", n_coils=8, allow_limited=False).allowed_mask[4]

assert load_control_gate(None, key=None, n_coils=8).allowed_mask == (True, True, True, True, True, True, True, True, True)
print("control gate direct assertions passed")
PY
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --help >/tmp/formal_help_gate.txt
rg -n "channel-gate" /tmp/formal_help_gate.txt

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m inverse_kinematics.ik_multisite_lm_opt --help >/tmp/ik_help_gate.txt
rg -n "channel_gate" /tmp/ik_help_gate.txt

XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from inverse_kinematics.ik_control_gate import load_control_gate

path = "experiment_records_ik_optimize/20260713203859_p1_clean_failed_channel_reaudit/channel_gate_classification.csv"
for key in ["A1_pt13", "A2_pt17", "A2_pt05", "A1_pt01"]:
    gate = load_control_gate(path, key=key, n_coils=8, missing_policy="block_all")
    print(key, gate.as_dict()["allowed_channels"], gate.as_dict()["blocked_channels"], gate.matched_rows)
PY
