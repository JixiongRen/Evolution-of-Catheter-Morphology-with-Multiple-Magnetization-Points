#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py --disable-gravity --eps-p-mm 5 --catheters A2 --target-indices 6,7,8 --outer-max-iter 50 --per-point-timeout-s 7200 --initial-I-A 4.71132582358,1.55504678688,-2.78249017799,-2.8248382608,4.21302426588,-0.438647087905,-4.60115238743,-3.78825265379 --initial-x-m 0.130351858822 --force-formal-20
