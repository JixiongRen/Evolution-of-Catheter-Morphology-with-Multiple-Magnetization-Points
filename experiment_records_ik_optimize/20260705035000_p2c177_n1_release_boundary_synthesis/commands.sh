#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \
  experiment_records_ik_optimize/20260705035000_p2c177_n1_release_boundary_synthesis/run_p2c177_n1_release_boundary_synthesis.py
