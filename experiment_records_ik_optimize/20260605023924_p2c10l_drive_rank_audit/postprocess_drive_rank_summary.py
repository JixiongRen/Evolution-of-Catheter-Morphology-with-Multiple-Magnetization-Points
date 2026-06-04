"""Postprocess P2c10l summary after revising effective rank classification."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

import run_p2c10l_drive_rank_audit as audit


def main() -> None:
    summary_path = OUT_DIR / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    for row in rows:
        row["failure_class"] = audit._classify(row, min_rank=6, min_sigma=1.0e-8)
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["failure_class"]] = class_counts.get(row["failure_class"], 0) + 1
    data["class_counts"] = class_counts
    data["metadata"]["min_rank"] = 6
    data["metadata"]["postprocess_note"] = (
        "Failure classes use total effective task sensitivity rank for the full six-dimensional terminal pose."
    )
    summary_path.write_text(json.dumps(audit._jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
    audit._write_csv(OUT_DIR / "summary.csv", rows)
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if row["failure_class"] != "驱动秩可接受":
                event = {
                    "ts": time.time(),
                    "case": row["scaled_case"],
                    "failure_class": row["failure_class"],
                    "evidence": row,
                }
                f.write(json.dumps(audit._jsonable(event), ensure_ascii=False) + "\n")
    audit._plot_spectrum(rows)
    audit._plot_coil_gain(rows)
    audit._plot_gate_vs_drive(rows)
    audit._plot_time_cost(rows, data["metadata"]["total_wall_s"])
    print(class_counts)


if __name__ == "__main__":
    main()
