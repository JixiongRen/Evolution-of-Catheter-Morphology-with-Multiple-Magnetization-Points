#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
  python experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract/run_p2c117_n2_trust_radius_interface_contract.py \
  > experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract/stdout.log \
  2> experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract/stderr.log
