#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from pathlib import Path
import tempfile
from DELIVER_TO_EXP.run_a1_axial_circle_formal_experiment import _experiment_type, _write_report, _write_output_index
assert _experiment_type(True, 1) == "smoke"
assert _experiment_type(True, 3) == "small_regression"
assert _experiment_type(True, 20) == "formal_20pt"
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    metadata = {
        "eps_p_m": 0.005, "experiment_type": "smoke", "target_count": 1,
        "requested_target_indices": [13], "target_center_m": [0.06,0,0],
        "target_radius_m": 0.1, "git_commit": "test", "enable_gravity": False,
        "x_search_max_m": 0.203, "sensitivity_config": {}, "p2_config": {},
        "per_point_timeout_s": 1200.0, "force_formal_20": True, "catheter_keys": ["A1"]
    }
    row = {
        "target_index": 13, "phase": "smoke", "err_true_mm": 100.0,
        "err_flex_mm": 99.0, "success_eps": False, "success_3mm": False,
        "success_5mm": False, "success_10mm": False, "cls": "GOOD",
        "outer_iter_count": 3, "wall_s": 1.0, "stop_reason": "max_outer_iter"
    }
    results = {"A1": {"gate_success_count_eps": 0, "gate_total": 0, "decision": "smoke_forced", "formal_rows": [row], "diagnostic_rows": [], "gate_rows": [], "overview_plot": None}}
    report = root / "report.md"
    _write_report(report, root, metadata, results)
    text = report.read_text()
    assert "烟测报告" in text
    assert "不得作为正式 20 点统计" in text
    _write_output_index(root, metadata, results)
print("experiment_type_semantics_ok")
PY
