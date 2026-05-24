"""Aggregate fixed-case FK optimization outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
L_REF = 0.11


def main() -> int:
    rows = []
    for case_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        summary_path = case_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_name": case_dir.name,
                "ok": summary.get("ok"),
                "classification": summary.get("classification"),
                "wall_s": summary.get("wall_s"),
                "first_call_compile_s": summary.get("first_call_compile_s"),
                "steady_solve_s": summary.get("steady_solve_s"),
                "n_iter": summary.get("n_iter"),
                "final_normE": summary.get("final_normE"),
                "max_abs_E": summary.get("max_abs_E"),
                "centerline_end_m": json.dumps(summary.get("centerline_end_m")),
                "stop_reason": summary.get("stop_reason"),
            }
        )

    if rows:
        fields = list(rows[0].keys())
        with (ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ref = np.load(ROOT / "fk_driver_lm03_solve_strict_baseline" / "centerline.npy")
    comparisons = []
    for case_name in [
        "fk_pos_lm01_shortstage_final1450_production",
        "fk_pos_lm01_shortstage_final1450",
        "fk_pos_lm01_final1450",
        "fk_driver_lm03_solve_strict_baseline",
    ]:
        cand = np.load(ROOT / case_name / "centerline.npy")
        n = min(len(ref), len(cand))
        delta = np.linalg.norm(ref[:n] - cand[:n], axis=1)
        comparisons.append(
            {
                "case_name": case_name,
                "reference": "fk_driver_lm03_solve_strict_baseline",
                "point_count": int(n),
                "max_point_error_m": float(delta.max()),
                "rms_point_error_m": float(np.sqrt(np.mean(delta * delta))),
                "end_error_m": float(delta[-1]),
                "max_rel_to_L_ref": float(delta.max() / L_REF),
            }
        )

    (ROOT / "candidate_vs_driver_baseline.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (ROOT / "candidate_vs_driver_baseline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"wrote {len(rows)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
