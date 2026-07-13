#!/usr/bin/env bash
set -euo pipefail

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --help \
  | rg -n "step-I|max|channel-gate|target-indices|force-formal"

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python - <<'PY'
from DELIVER_TO_EXP.run_a1_axial_circle_formal_experiment import _circle_targets, _select_target_positions, _experiment_type, _forced_phase_name
all_targets = _circle_targets(n_points=20, center_m=(0.06, 0, 0), radius_m=0.1)
subset = [all_targets[i] for i in [1, 12, 13]]
print('subset_indices', [t['index'] for t in subset])
print('gate_positions_subset', _select_target_positions(subset, [0,5,10,15]))
print('diag_positions_subset', _select_target_positions(subset, [0,2,5,7,10,12,15,17]))
print('full_gate_positions', _select_target_positions(all_targets, [0,5,10,15]))
print('experiment_type_3_forced', _experiment_type(True, 3))
print('phase_3_forced', _forced_phase_name(_experiment_type(True, 3)))
PY
