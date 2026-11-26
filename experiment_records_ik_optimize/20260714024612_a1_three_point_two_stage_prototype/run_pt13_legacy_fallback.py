"""Run pt13 legacy global-basin fallback and same-branch x refinement."""

from __future__ import annotations

import csv
import json
import os
import shutil
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
FORMAL_SCRIPT = REPO / "DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py"
OUT_ROOT = REPO / "DELIVER_TO_EXP/formal_experiments"
POINT = 13


def _formal_run_dirs_since(start_time: float) -> list[Path]:
    candidates: list[Path] = []
    for path in OUT_ROOT.glob("*_A1_A2_5mm_supervised_IK_no_gravity"):
        summary = path / "summary.json"
        if summary.exists() and summary.stat().st_mtime >= start_time - 1.0:
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _point_dir(run_root: Path) -> Path:
    path = run_root / "A1/smoke/pt_13"
    if path.exists():
        return path
    matches = list((run_root / "A1").glob("*/pt_13"))
    if not matches:
        raise FileNotFoundError(f"Cannot find pt_13 under {run_root}")
    return matches[0]


def _load_summary(run_root: Path) -> dict[str, Any]:
    return json.loads((_point_dir(run_root) / "formal_point_summary.json").read_text(encoding="utf-8"))


def _write_gate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "point_key": "A1_pt13",
            "kind": "x",
            "channel": "-1",
            "decision": "有限使用候选",
            "limited_safe_step_abs_max": "0.000125",
            "notes": "Empirical x micro-search, not production sensitivity approval",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["point_key", "kind", "channel", "decision", "limited_safe_step_abs_max", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _run(stage: str, extra_args: list[str], timeout_s: float) -> tuple[Path, dict[str, Any], float, int]:
    stage_dir = EXP_DIR / "runs" / "pt_13" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(FORMAL_SCRIPT),
        "--disable-gravity",
        "--eps-p-mm",
        "5",
        "--force-formal-20",
        "--catheters",
        "A1",
        "--target-indices",
        "13",
        "--outer-max-iter",
        "50",
        "--per-point-timeout-s",
        f"{timeout_s:.12g}",
        *extra_args,
    ]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda"}
    (stage_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shlex.quote(v) for v in cmd) + "\n",
        encoding="utf-8",
    )
    start = time.time()
    with (stage_dir / "stdout.log").open("w", encoding="utf-8") as out, (stage_dir / "stderr.log").open("w", encoding="utf-8") as err:
        completed = subprocess.run(cmd, cwd=REPO, env=env, stdout=out, stderr=err, text=True, check=False)
    wall_s = float(time.time() - start)
    dirs = _formal_run_dirs_since(start)
    if not dirs:
        raise RuntimeError(f"No formal output found for {stage}")
    run_root = dirs[-1]
    summary = _load_summary(run_root)
    shutil.copytree(_point_dir(run_root), stage_dir / "formal_point_snapshot", dirs_exist_ok=True)
    return run_root, summary, wall_s, int(completed.returncode)


def _row(stage: str, run_root: Path, summary: dict[str, Any], wall_s: float, return_code: int) -> dict[str, Any]:
    return {
        "point": 13,
        "stage": stage,
        "return_code": int(return_code),
        "wall_s": float(wall_s),
        "formal_run_root": str(run_root.resolve()),
        "err_true_mm": summary.get("err_true_mm"),
        "best_err_true_mm": summary.get("best_err_true_mm"),
        "success_5mm": summary.get("success_5mm"),
        "stop_reason": summary.get("stop_reason"),
        "accept_count": summary.get("accept_count"),
        "reject_count": summary.get("reject_count"),
        "outer_iter_count": summary.get("outer_iter_count"),
        "runtime_platform": summary.get("runtime_platform"),
        "runtime_device_kind": summary.get("runtime_device_kind"),
        "branch_npz_path": summary.get("branch_npz_path"),
        "z_star_hash": summary.get("z_star_hash"),
        "centerline_hash": summary.get("centerline_hash"),
        "I_A": summary.get("I_A"),
        "x_m": summary.get("x_m"),
    }


def _append_rows(rows: list[dict[str, Any]]) -> None:
    csv_path = EXP_DIR / "two_stage_summary.csv"
    existing: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    fieldnames = [
        "point",
        "stage",
        "return_code",
        "wall_s",
        "err_true_mm",
        "best_err_true_mm",
        "success_5mm",
        "stop_reason",
        "accept_count",
        "reject_count",
        "outer_iter_count",
        "runtime_platform",
        "runtime_device_kind",
        "formal_run_root",
        "branch_npz_path",
        "z_star_hash",
        "centerline_hash",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing + rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path = EXP_DIR / "two_stage_summary.json"
    payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {"rows": []}
    payload["rows"] = payload.get("rows", []) + rows
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    gate = EXP_DIR / "gates" / "pt13_x_micro.csv"
    _write_gate(gate)
    rows: list[dict[str, Any]] = []
    run_root, summary, wall_s, rc = _run(
        "global_legacy_fallback",
        ["--step-I-max", "5", "--step-x-max", "0.02"],
        timeout_s=1800.0,
    )
    rows.append(_row("global_legacy_fallback", run_root, summary, wall_s, rc))
    err = float(summary.get("err_true_mm", float("inf")))
    branch_npz = summary.get("branch_npz_path")
    currents = summary.get("I_A")
    x_m = summary.get("x_m")
    if 5.0 < err <= 10.0 and branch_npz and currents is not None and x_m is not None:
        run_root, summary, wall_s, rc = _run(
            "legacy_same_branch_x_micro",
            [
                "--step-I-max",
                "0.125",
                "--step-x-max",
                "0.000125",
                "--channel-gate-file=" + str(gate.resolve()),
                "--channel-gate-missing-policy",
                "block_all",
                "--channel-gate-allow-limited",
                "--initial-I-A=" + ",".join(f"{float(v):.12g}" for v in currents),
                f"--initial-x-m={float(x_m):.12g}",
                "--initial-branch-state-npz=" + str(Path(branch_npz).resolve()),
            ],
            timeout_s=1200.0,
        )
        rows.append(_row("legacy_same_branch_x_micro", run_root, summary, wall_s, rc))
    _append_rows(rows)


if __name__ == "__main__":
    main()
