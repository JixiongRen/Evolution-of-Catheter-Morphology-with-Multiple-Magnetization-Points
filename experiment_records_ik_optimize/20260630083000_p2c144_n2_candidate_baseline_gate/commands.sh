#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

# Plan-only mode.
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/run_p2c144_n2_candidate_baseline_gate.py

# Expensive execution mode. Expected runtime: roughly 30-45 minutes.
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/run_p2c144_n2_candidate_baseline_gate.py --run \
  > experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/stdout.log \
  2> experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/stderr.log

XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/run_p2c144_n2_candidate_baseline_gate.py --postprocess
