#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260606204500_p2c33_strict_alpha_protocol_audit/run_p2c33_strict_alpha_protocol_audit.py \
  > experiment_records_ik_optimize/20260606204500_p2c33_strict_alpha_protocol_audit/stdout_stderr.log \
  2>&1
