#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py")
spec = importlib.util.spec_from_file_location("formal_runner", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rows = [
    {
        "err_true_mm": 4.0,
        "success_eps": True,
        "success_3mm": False,
        "success_5mm": True,
        "success_10mm": True,
        "wall_s": 1.0,
        "algorithm_statistics_included": False,
    },
    {
        "err_true_mm": 6.0,
        "success_eps": False,
        "success_3mm": False,
        "success_5mm": False,
        "success_10mm": True,
        "wall_s": 2.0,
        "algorithm_statistics_included": True,
    },
]
stats = mod._summarize_rows(rows)
assert stats["count"] == 2
assert stats["algorithm_count"] == 1
assert stats["success_count_eps"] == 0
assert abs(stats["err_true_mm_mean"] - 6.0) < 1e-9
print("summary_filter_ok")
PY

