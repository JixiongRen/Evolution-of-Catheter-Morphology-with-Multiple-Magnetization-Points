#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points

XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs python \
  experiment_records_ik_optimize/20260605032052_p2c10m_task_reduction_gate/run_p2c10m_task_reduction_gate.py \
  --candidates n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10,n3_mixed_rand20:0.10,n3_mixed_base_i4:1.00 \
  --modes position3,position_axis,full_pose6 \
  --step 0.05 \
  --continuity_mu 1.0 \
  --reg_mu 1.0e-4 \
  --max_iter 32 \
  --random_count 1
