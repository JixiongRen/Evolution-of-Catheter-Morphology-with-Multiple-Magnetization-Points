#!/usr/bin/env python3
"""Build a reusable branch-state seed registry from formal A1 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any, default: float = float("inf")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(root: Path, catheter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob(f"*/{catheter}/all_points_summary.csv")):
        run_root = csv_path.parents[1]
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row["source_csv"] = str(csv_path)
                row["source_run_root"] = str(run_root)
                rows.append(row)
    return rows


def _is_candidate(row: dict[str, Any], max_error_mm: float) -> bool:
    branch_path = Path(str(row.get("branch_npz_path", "")))
    if not branch_path.exists():
        return False
    err = _parse_float(row.get("err_true_mm"))
    if err > max_error_mm:
        return False
    if str(row.get("cls", "")).upper() != "GOOD":
        return False
    if str(row.get("runtime_platform", "")).lower() not in {"gpu", "cuda"}:
        return False
    if _parse_float(row.get("normE")) > 1.1e-5:
        return False
    return True


def _candidate_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "catheter_type": row.get("catheter_type", "A1"),
        "target_index": int(float(row.get("target_index", -1))),
        "err_true_mm": _parse_float(row.get("err_true_mm")),
        "success_5mm": _parse_bool(row.get("success_5mm")),
        "success_10mm": _parse_bool(row.get("success_10mm")),
        "x_m": _parse_float(row.get("x_m"), default=float("nan")),
        "max_abs_current_A": _parse_float(row.get("max_abs_current_A"), default=float("nan")),
        "cls": row.get("cls", ""),
        "normE": _parse_float(row.get("normE"), default=float("nan")),
        "branch_npz_path": row.get("branch_npz_path", ""),
        "z_star_hash": row.get("z_star_hash", ""),
        "centerline_hash": row.get("centerline_hash", ""),
        "source_run_root": row.get("source_run_root", ""),
        "source_csv": row.get("source_csv", ""),
        "phase": row.get("phase", ""),
        "warm_start_source": row.get("warm_start_source", ""),
        "evidence_role_zh": row.get("evidence_role_zh", ""),
    }


def build_registry(root: Path, catheter: str, max_error_mm: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_rows(root, catheter)
    under_error = [row for row in rows if _parse_float(row.get("err_true_mm")) <= max_error_mm]
    missing_branch = [
        row
        for row in under_error
        if not str(row.get("branch_npz_path", "")).strip() or not Path(str(row.get("branch_npz_path", ""))).exists()
    ]
    missing_runtime = [
        row
        for row in under_error
        if str(row.get("runtime_platform", "")).lower() not in {"gpu", "cuda"}
    ]
    weak_or_bad_fk = [
        row
        for row in under_error
        if str(row.get("cls", "")).upper() != "GOOD" or _parse_float(row.get("normE")) > 1.1e-5
    ]
    candidates = [_candidate_record(row) for row in rows if _is_candidate(row, max_error_mm)]
    candidates.sort(key=lambda r: (r["target_index"], r["err_true_mm"]))

    best_by_target: dict[int, dict[str, Any]] = {}
    for row in candidates:
        target = int(row["target_index"])
        if target not in best_by_target:
            best_by_target[target] = row

    best_rows = [best_by_target[k] for k in sorted(best_by_target)]
    summary = {
        "catheter": catheter,
        "scan_root": str(root),
        "source_rows": len(rows),
        "rows_with_error_le_threshold": len(under_error),
        "rows_missing_reusable_branch_state": len(missing_branch),
        "rows_missing_gpu_runtime_evidence": len(missing_runtime),
        "rows_with_weak_or_bad_fk": len(weak_or_bad_fk),
        "candidate_rows": len(candidates),
        "covered_targets": len(best_rows),
        "success_5mm_targets": sum(1 for row in best_rows if row["success_5mm"]),
        "success_10mm_targets": sum(1 for row in best_rows if row["success_10mm"]),
        "max_error_mm": max_error_mm,
        "target_indices": [row["target_index"] for row in best_rows],
    }
    return best_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "catheter_type",
        "target_index",
        "err_true_mm",
        "success_5mm",
        "success_10mm",
        "x_m",
        "max_abs_current_A",
        "cls",
        "normE",
        "branch_npz_path",
        "z_star_hash",
        "centerline_hash",
        "source_run_root",
        "source_csv",
        "phase",
        "warm_start_source",
        "evidence_role_zh",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("DELIVER_TO_EXP/formal_experiments"))
    parser.add_argument("--catheter", default="A1")
    parser.add_argument("--max-error-mm", type=float, default=10.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = build_registry(args.root, args.catheter, args.max_error_mm)
    _write_csv(args.out_csv, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
