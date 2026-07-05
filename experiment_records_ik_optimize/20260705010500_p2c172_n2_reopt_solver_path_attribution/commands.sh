#!/usr/bin/env bash
set -euo pipefail
python -m json.tool experiment_records_ik_optimize/20260705010500_p2c172_n2_reopt_solver_path_attribution/summary.json >/tmp/p2c172_summary_checked.json
