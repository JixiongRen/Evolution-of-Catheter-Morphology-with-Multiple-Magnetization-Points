#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --disable-gravity --eps-p-mm 5 --catheters A2 --target-indices 5,9,10,17,19 --outer-max-iter 50 --per-point-timeout-s 2700 --force-formal-20
