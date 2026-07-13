#!/usr/bin/env bash
set -euo pipefail

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python -m py_compile \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py \
  inverse_kinematics/ik_multisite_lm_opt.py

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --help | rg "sensitivity|ridge"

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  -m inverse_kinematics.ik_multisite_lm_opt --help | rg "sensitivity|ridge"

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  DELIVER_TO_EXP/run_a1_a2_sensitivity_reaudit.py --help | rg "sensitivity|spectral|svd|ridge|solve"
