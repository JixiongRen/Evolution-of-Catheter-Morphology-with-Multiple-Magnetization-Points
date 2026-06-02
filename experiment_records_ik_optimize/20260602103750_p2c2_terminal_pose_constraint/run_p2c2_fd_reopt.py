"""Run a small re-optimized FD audit for the P2c2 terminal-pose candidate."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

import jax
import numpy as np

import run_p2c2_terminal_pose as p2c2

jax.config.update("jax_enable_x64", True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    engine = p2c2.p2c1.p2c._build_engine((1, 1, 1), max_iter=120, tol=1e-5)
    I0 = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=float)
    base_oracle = p2c2.p2c1.p2c._solve_oracle(engine, I0, z0=None)
    y = np.load(OUT_DIR / "M111" / "pose_c3_state_y.npy")
    base_result = p2c2._solve_pose_candidate(
        p2c2.PoseCandidateSpec("c3_pose_terminal_v1", max_nfev=1),
        base_oracle,
        engine,
        steps_per_flex=4,
        y0=y,
    )
    base_result.y = y
    payload = p2c2._run_reoptimized_fd_audit(
        engine=engine,
        base_oracle=base_oracle,
        base_result=base_result,
        spec=p2c2.PoseCandidateSpec("c3_pose_terminal_v1", max_nfev=8),
        steps_per_flex=4,
        audit_nfev=8,
        eps_values=[0.02],
    )
    (OUT_DIR / "fd_reopt_small.json").write_text(json.dumps(p2c2._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "fd_reopt_small.csv", payload["rows"])
    print(json.dumps(p2c2._jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
