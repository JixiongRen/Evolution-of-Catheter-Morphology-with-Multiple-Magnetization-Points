#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --disable-gravity --eps-p-mm 5 --catheters A1,A2 --target-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --outer-max-iter 50 --per-point-timeout-s 2700 --force-formal-20
