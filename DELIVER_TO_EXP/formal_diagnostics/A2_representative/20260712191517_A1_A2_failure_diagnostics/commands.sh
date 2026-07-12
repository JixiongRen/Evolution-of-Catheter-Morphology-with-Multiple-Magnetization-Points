#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/run_a1_a2_failure_diagnostics.py --source-run /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_experiments/20260712183547_A1_A2_5mm_supervised_IK_no_gravity --out-root /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/DELIVER_TO_EXP/formal_diagnostics/A2_representative
