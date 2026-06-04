#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260605004600_p2c10g_feasible_manifold_distance_map/run_p2c10g_feasible_manifold_distance_map.py \
  --base_case n3_mixed_base_i4 \
  --target_case n3_mixed_rand20 \
  --alpha_start 0.20 \
  --alpha_stop 0.25 \
  --alpha_step 0.01 \
  --continuity_scale 1000 \
  --boundary_scale 1.0 \
  --reg_mu 1.0e-6 \
  --max_iter 120 \
  --steps_per_flex 4
