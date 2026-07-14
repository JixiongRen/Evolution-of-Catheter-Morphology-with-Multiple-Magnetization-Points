#!/usr/bin/env python3
"""Build an A1 control-only seed registry from historical formal outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_vector(value: Any) -> list[float]:
    """Parse a vector saved in JSON, Python, or CSV notation."""
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (TypeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [float(v) for v in parsed]
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _detail_for_row(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = str(row.get("run_dir", "")).strip()
    if not run_dir:
        return {}
    point_dir = Path(run_dir)
    for name in ("formal_point_summary.json", "result.json"):
        path = point_dir / name
        if path.exists():
            data = _read_json(path)
            if data:
                return data
    return {}


def _control_from(row: dict[str, Any], detail: dict[str, Any]) -> tuple[list[float], float]:
    I_A = _parse_vector(row.get("I_A"))
    if len(I_A) != 8:
        I_A = _parse_vector(detail.get("I_A") or detail.get("I"))
    x_m = _parse_float(row.get("x_m"))
    if math.isnan(x_m):
        x_m = _parse_float(detail.get("x_m", detail.get("x")))
    return I_A, x_m


def _iter_rows(root: Path, catheter: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob(f"*/{catheter}/all_points_summary.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                item = dict(row)
                item["source_csv"] = str(csv_path)
                item["source_run_root"] = str(csv_path.parents[1])
                rows.append(item)
    return rows


def build_registry(
    *,
    root: Path,
    catheter: str,
    max_error_mm: float,
    target_indices: set[int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    skipped_missing_control = 0
    for row in _iter_rows(root, catheter):
        target_index = int(float(row.get("target_index", -1)))
        if target_indices is not None and target_index not in target_indices:
            continue
        err = _parse_float(row.get("err_true_mm"))
        if math.isnan(err) or err > max_error_mm:
            continue
        detail = _detail_for_row(row)
        I_A, x_m = _control_from(row, detail)
        if len(I_A) != 8 or math.isnan(x_m):
            skipped_missing_control += 1
            continue
        branch_path = str(row.get("branch_npz_path") or detail.get("branch_npz_path") or "").strip()
        branch_exists = bool(branch_path and Path(branch_path).expanduser().exists())
        seed_type = "branch_state" if branch_exists else "control_only"
        candidates.append(
            {
                "catheter_type": row.get("catheter_type", catheter),
                "target_index": target_index,
                "err_true_mm": err,
                "success_5mm": err <= 5.0,
                "success_10mm": err <= 10.0,
                "I_A_json": json.dumps(I_A),
                "x_m": x_m,
                "max_abs_current_A": max(abs(float(v)) for v in I_A),
                "cls": row.get("cls") or detail.get("cls", ""),
                "normE": _parse_float(row.get("normE") or detail.get("normE")),
                "branch_npz_path": branch_path if branch_exists else "",
                "seed_type": seed_type,
                "z_star_hash": row.get("z_star_hash", detail.get("z_star_hash", "")),
                "centerline_hash": row.get("centerline_hash", detail.get("centerline_hash", "")),
                "source_run_root": row.get("source_run_root", ""),
                "source_csv": row.get("source_csv", ""),
                "phase": row.get("phase", ""),
                "warm_start_source": row.get("warm_start_source", ""),
                "evidence_role_zh": "历史控制量种子" if seed_type == "control_only" else "历史分支状态种子",
            }
        )

    candidates.sort(key=lambda item: (item["target_index"], item["err_true_mm"]))
    best_by_target: dict[int, dict[str, Any]] = {}
    for item in candidates:
        best_by_target.setdefault(int(item["target_index"]), item)
    best_rows = [best_by_target[k] for k in sorted(best_by_target)]
    summary = {
        "catheter": catheter,
        "scan_root": str(root),
        "max_error_mm": max_error_mm,
        "target_indices_requested": sorted(target_indices) if target_indices is not None else None,
        "candidate_rows": len(candidates),
        "covered_targets": len(best_rows),
        "control_only_targets": sum(1 for item in best_rows if item["seed_type"] == "control_only"),
        "branch_state_targets": sum(1 for item in best_rows if item["seed_type"] == "branch_state"),
        "skipped_missing_control": skipped_missing_control,
        "target_indices": [item["target_index"] for item in best_rows],
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
        "I_A_json",
        "x_m",
        "max_abs_current_A",
        "cls",
        "normE",
        "branch_npz_path",
        "seed_type",
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


def _parse_target_indices(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("DELIVER_TO_EXP/formal_experiments"))
    parser.add_argument("--catheter", default="A1")
    parser.add_argument("--max-error-mm", type=float, default=10.0)
    parser.add_argument("--target-indices", type=str, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = build_registry(
        root=args.root,
        catheter=args.catheter,
        max_error_mm=args.max_error_mm,
        target_indices=_parse_target_indices(args.target_indices),
    )
    _write_csv(args.out_csv, rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
