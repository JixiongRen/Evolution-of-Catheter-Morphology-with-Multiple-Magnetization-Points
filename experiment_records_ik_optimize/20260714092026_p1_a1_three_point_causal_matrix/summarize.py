"""Summarize A1 three-point causal-matrix formal-run outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def main() -> None:
    status_path = Path(sys.argv[1])
    rows = list(csv.DictReader(status_path.open("r", encoding="utf-8")))
    cases: list[dict[str, Any]] = []
    for row in rows:
        summary = _read_json(row.get("summary", ""))
        case_rows: list[dict[str, Any]] = []
        for item in summary.get("results", {}).values():
            case_rows.extend(item.get("formal_rows") or item.get("diagnostic_rows") or [])
        cases.append(
            {
                "case": row.get("case"),
                "exit_code": int(row.get("exit_code") or 0),
                "summary": row.get("summary"),
                "report": row.get("report"),
                "log": row.get("log"),
                "rows": [
                    {
                        "target_index": item.get("target_index"),
                        "err_true_mm": _safe_float(item.get("err_true_mm")),
                        "success_eps": bool(item.get("success_eps")),
                        "stop_reason": item.get("stop_reason"),
                        "evidence_role": item.get("evidence_role"),
                        "algorithm_statistics_included": bool(item.get("algorithm_statistics_included")),
                        "control_gate_stage": item.get("control_gate_stage"),
                        "wall_s": _safe_float(item.get("wall_s")),
                    }
                    for item in case_rows
                ],
            }
        )
    print(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

