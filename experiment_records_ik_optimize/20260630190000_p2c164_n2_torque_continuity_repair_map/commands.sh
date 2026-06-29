#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260630190000_p2c164_n2_torque_continuity_repair_map/run_p2c164_n2_torque_continuity_repair_map.py
