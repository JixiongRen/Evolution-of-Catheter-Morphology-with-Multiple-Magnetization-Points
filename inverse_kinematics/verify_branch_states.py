"""Verify saved branch-state snapshots without re-solving FK."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_DIR = Path(__file__).resolve().parent
_FK_DIR = _REPO_ROOT / "forward_kinematics_optimized"
for _path in (_REPO_ROOT, _MODULE_DIR, _FK_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from forward_kinematics_optimized.nondim import NondimScales
from ik_artifacts import extract_centerline_dim
from ik_diff import extract_terminal_p_dim


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_one_state(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    npz_path = root / row["npz_path"]
    data = np.load(npz_path)
    z_star = data["z_star_bar"]
    scale_values = data["scales"]
    scales = NondimScales(L_ref=float(scale_values[0]), F_ref=float(scale_values[1]), M_ref=float(scale_values[2]))
    m_list = [int(v) for v in row["M_list"]]
    terminal = np.asarray(extract_terminal_p_dim(z_star, M_list=m_list, scales=scales), dtype=float).reshape(3)
    centerline = np.asarray(extract_centerline_dim(z_star, M_list=m_list, scales=scales), dtype=float)
    terminal_saved = np.asarray(data["terminal_true_m"], dtype=float).reshape(3)
    centerline_saved = np.asarray(data["centerline_m"], dtype=float)
    if centerline.shape == centerline_saved.shape:
        centerline_max_mm = float(np.max(np.linalg.norm(centerline - centerline_saved, axis=1)) * 1000.0)
    else:
        centerline_max_mm = float("inf")
    return {
        "eval_id": row["eval_id"],
        "phase": row["phase"],
        "iter": int(row["iter"]),
        "npz_path": row["npz_path"],
        "terminal_reload_error_mm": float(np.linalg.norm(terminal - terminal_saved) * 1000.0),
        "centerline_reload_max_error_mm": centerline_max_mm,
        "centerline_shape_match": bool(centerline.shape == centerline_saved.shape),
    }


def verify_branch_state_dir(
    point_dir: Path,
    *,
    terminal_gate_mm: float = 0.1,
    centerline_gate_mm: float = 0.5,
) -> tuple[list[dict[str, Any]], bool]:
    """Verify all saved branch states in one point directory."""
    root = point_dir / "branch_states"
    rows = _load_manifest(root / "branch_state_manifest.jsonl")
    out_rows = [_verify_one_state(root, row) for row in rows]
    ok = all(
        row["terminal_reload_error_mm"] <= terminal_gate_mm
        and row["centerline_reload_max_error_mm"] <= centerline_gate_mm
        and row["centerline_shape_match"]
        for row in out_rows
    )
    return out_rows, bool(ok)


def _iter_point_dirs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = path.resolve()
        if (path / "branch_states" / "branch_state_manifest.jsonl").exists():
            yield path
            continue
        for child in sorted(path.rglob("branch_state_manifest.jsonl")):
            yield child.parents[1]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Point directories or parent experiment directories.")
    parser.add_argument("--terminal-gate-mm", type=float, default=0.1)
    parser.add_argument("--centerline-gate-mm", type=float, default=0.5)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    all_rows: list[dict[str, Any]] = []
    all_ok = True
    for point_dir in _iter_point_dirs(Path(p) for p in args.paths):
        rows, ok = verify_branch_state_dir(
            point_dir,
            terminal_gate_mm=float(args.terminal_gate_mm),
            centerline_gate_mm=float(args.centerline_gate_mm),
        )
        all_ok = all_ok and ok
        for row in rows:
            row["point_dir"] = str(point_dir)
            row["ok"] = bool(ok)
        all_rows.extend(rows)
        print(f"{point_dir}: states={len(rows)} reload_ok={ok}")
    if args.out_csv:
        path = Path(args.out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "point_dir",
            "eval_id",
            "phase",
            "iter",
            "terminal_reload_error_mm",
            "centerline_reload_max_error_mm",
            "centerline_shape_match",
            "ok",
            "npz_path",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in fields} for row in all_rows)
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ok": all_ok, "rows": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
