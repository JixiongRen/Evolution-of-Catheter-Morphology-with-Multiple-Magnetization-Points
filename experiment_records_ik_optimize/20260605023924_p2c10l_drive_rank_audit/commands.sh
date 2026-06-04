#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605023924_p2c10l_drive_rank_audit/run_p2c10l_drive_rank_audit.py \
  --candidates n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_rand20:0.10,n3_mixed_base_i4:1.00 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 32 \
  --rank_rel_tol 1.0e-3 \
  --rank_abs_tol 1.0e-8 \
  --min_rank 6 \
  --min_sigma 1.0e-8

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605023924_p2c10l_drive_rank_audit/postprocess_drive_rank_summary.py
