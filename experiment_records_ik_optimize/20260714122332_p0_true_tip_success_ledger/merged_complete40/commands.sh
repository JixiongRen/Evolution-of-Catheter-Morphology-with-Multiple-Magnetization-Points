#!/usr/bin/env bash
set -euo pipefail
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python DELIVER_TO_EXP/audit_supervised_ik_success_ledger.py \
  --summary-csv DELIVER_TO_EXP/formal_experiments/20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40/summary.csv \
  --out-dir experiment_records_ik_optimize/20260714122332_p0_true_tip_success_ledger/merged_complete40
