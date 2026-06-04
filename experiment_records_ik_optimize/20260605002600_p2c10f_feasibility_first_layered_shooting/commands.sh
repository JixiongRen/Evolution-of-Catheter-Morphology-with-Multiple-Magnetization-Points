#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260605002600_p2c10f_feasibility_first_layered_shooting/run_p2c10f_feasibility_first_layered_shooting.py \
  --base_case n3_mixed_base_i4 \
  --target_case n3_mixed_rand20 \
  --alphas 0.20,0.25 \
  --task_scales 0,0.1,0.3,1.0 \
  --continuity_scale 1000 \
  --boundary_scale 1.0 \
  --reg_mu 1.0e-6 \
  --max_iter 80 \
  --steps_per_flex 4
