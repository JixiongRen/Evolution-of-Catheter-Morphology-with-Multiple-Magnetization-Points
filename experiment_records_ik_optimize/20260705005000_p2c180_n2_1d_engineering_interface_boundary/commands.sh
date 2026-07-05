#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260705005000_p2c180_n2_1d_engineering_interface_boundary/run_p2c180_n2_1d_engineering_interface_boundary.py
