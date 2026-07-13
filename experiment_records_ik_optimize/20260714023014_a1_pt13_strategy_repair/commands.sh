#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points/experiment_records_ik_optimize/20260714023014_a1_pt13_strategy_repair/run_a1_pt13_three_arm_ablation.py
