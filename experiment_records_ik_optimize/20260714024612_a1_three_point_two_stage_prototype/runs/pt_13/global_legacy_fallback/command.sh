#!/usr/bin/env bash
set -euo pipefail
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --disable-gravity --eps-p-mm 5 --force-formal-20 --catheters A1 --target-indices 13 --outer-max-iter 50 --per-point-timeout-s 1800 --step-I-max 5 --step-x-max 0.02
