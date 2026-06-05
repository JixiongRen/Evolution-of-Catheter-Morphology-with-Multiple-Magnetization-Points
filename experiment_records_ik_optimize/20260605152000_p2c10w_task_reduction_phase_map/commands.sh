#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run -n evolution_of_mcrs \
python experiment_records_ik_optimize/20260605152000_p2c10w_task_reduction_phase_map/run_p2c10w_task_reduction_phase_map.py \
  --candidates n3_mixed_rand20:0.25,n3_mixed_rand20:0.40,n3_mixed_base_i4:1.00 \
  --tasks position,axis,position_axis,full_pose \
  --max_iter 100
