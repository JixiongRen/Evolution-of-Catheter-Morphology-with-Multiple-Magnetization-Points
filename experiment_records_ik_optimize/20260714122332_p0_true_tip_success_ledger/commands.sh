#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/audit_supervised_ik_success_ledger.py \
  --out-dir experiment_records_ik_optimize/20260714122332_p0_true_tip_success_ledger
