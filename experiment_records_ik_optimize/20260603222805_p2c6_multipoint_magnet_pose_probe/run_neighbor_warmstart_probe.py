"""Run limited neighbor warm-start probes for P2c6 failed zero-y0 cases."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_multipoint_magnet_pose_probe as p2c6


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    rows: list[dict[str, Any]] = []
    cases = {case.name: case for case in p2c6.p2c4._cases()}
    target_case = cases["n3_mixed_rand20"]
    source_case = "n3_mixed_base_i4"
    bundles = [
        p2c6.BundleSpec("p1_position", "multipoint-position", True, False, False),
        p2c6.BundleSpec("p2_position_axis", "multipoint-axis", True, True, False),
        p2c6.BundleSpec("p3_previous_full_pose", "optional-full-pose-stress", True, False, True),
    ]
    engine = p2c6.p2c4._build_engine(target_case, max_iter=120, tol=1e-5)
    oracle = p2c6.p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(target_case.currents, dtype=float), z0=None)
    params = oracle["params"]
    targets = p2c6._targets_from_oracle(oracle, engine)
    active = p2c6._active_magnet_mask(target_case)
    previous = p2c6._previous_active_mask(active)
    for bundle in bundles:
        source_y = RUN_DIR / source_case / bundle.name / "zero_y0" / "state_y.npy"
        if not source_y.exists():
            continue
        y0 = np.load(source_y)
        masks = {
            "position": active if bundle.use_positions else np.zeros_like(active, dtype=bool),
            "axis": previous if bundle.use_previous_axis else np.zeros_like(active, dtype=bool),
            "so3": previous if bundle.use_previous_so3 else np.zeros_like(active, dtype=bool),
        }
        residual_jit, jacobian_jit, output_jit = p2c6._make_functions(params, targets, masks)
        compile_profile = p2c6._time_compile(residual_jit, jacobian_jit, output_jit, y0, p2c6._stages(bundle)[-1])
        result = p2c6._lm_solve(
            y0=y0,
            residual_jit=residual_jit,
            jacobian_jit=jacobian_jit,
            output_jit=output_jit,
            targets=targets,
            masks=masks,
            bundle=bundle,
        )
        metrics = p2c6._metrics(result.out, targets, masks)
        row = p2c6._row_from_result(
            case=target_case,
            bundle=bundle,
            init_mode="neighbor_base_zero_y0",
            result=result,
            compile_profile=compile_profile,
            metrics=metrics,
            masks=masks,
            oracle_wall_s=float(oracle["wall_s"]),
        )
        rows.append(row)
        sub_dir = RUN_DIR / "neighbor_warmstart" / target_case.name / bundle.name
        sub_dir.mkdir(parents=True, exist_ok=True)
        np.save(sub_dir / "state_y.npy", result.y)
        np.save(sub_dir / "centerline.npy", result.out["centerline"])
        (sub_dir / "summary.json").write_text(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), encoding="utf-8")
        (sub_dir / "history.json").write_text(json.dumps(_jsonable(result.history), ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"source_case": source_case, "target_case": target_case.name, "rows": rows}
    (RUN_DIR / "neighbor_warmstart_summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(RUN_DIR / "neighbor_warmstart_summary.csv", rows)
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
