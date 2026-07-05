#!/usr/bin/env bash
set -euo pipefail

# P2c166 is a protocol-only checkpoint. It does not run a long numerical solve.
python - <<'PY'
import json
from pathlib import Path

path = Path("experiment_records_ik_optimize/20260704220000_p2c166_n2_moment_reference_audit_protocol/summary.json")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["status"] == "moment_reference_audit_protocol_ready"
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
