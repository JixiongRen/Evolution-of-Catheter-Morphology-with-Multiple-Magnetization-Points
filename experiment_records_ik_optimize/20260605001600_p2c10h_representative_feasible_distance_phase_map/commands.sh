#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python experiment_records_ik_optimize/20260605001600_p2c10h_representative_feasible_distance_phase_map/run_p2c10h_representative_feasible_distance_phase_map.py \
  --cases n1_tip_z_rand15,n2_z_pair_rand12,n3_mixed_rand20 \
  --alphas 0.25,0.50,0.75,1.00 \
  --continuity_scale 1000 \
  --boundary_scale 1.0 \
  --reg_mu 1.0e-6 \
  --max_iter 120 \
  --steps_per_flex 4
