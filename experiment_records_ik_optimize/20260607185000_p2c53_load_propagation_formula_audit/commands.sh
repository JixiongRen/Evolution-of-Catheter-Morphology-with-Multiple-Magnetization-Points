#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260607185000_p2c53_load_propagation_formula_audit/run_p2c53_load_propagation_formula_audit.py \
  > experiment_records_ik_optimize/20260607185000_p2c53_load_propagation_formula_audit/stdout.log \
  2> experiment_records_ik_optimize/20260607185000_p2c53_load_propagation_formula_audit/stderr.log
