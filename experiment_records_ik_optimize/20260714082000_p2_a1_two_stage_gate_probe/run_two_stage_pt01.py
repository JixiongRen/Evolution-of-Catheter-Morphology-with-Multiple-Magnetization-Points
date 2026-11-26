"""Run a two-stage A1 point probe with basin and strict gates."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path("/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points")
EXP_DIR = REPO / "experiment_records_ik_optimize" / "20260714082000_p2_a1_two_stage_gate_probe"
PY = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
RUNNER = REPO / "DELIVER_TO_EXP" / "run_a1_axial_circle_formal_experiment.py"
STRICT_GATE = REPO / "experiment_records_ik_optimize" / "20260714070559_p1b_representative_sensitivity_audit" / "audit_output" / "a1_representative_channel_gate.csv"
BASIN_GATE = EXP_DIR / "a1_basin_gate_allow_x.csv"


COMMON_ARGS = [
    "--disable-gravity",
    "--eps-p-mm",
    "5",
    "--catheters",
    "A1",
    "--target-indices",
    "1",
    "--outer-max-iter",
    "50",
    "--per-point-timeout-s",
    "2700",
    "--step-I-max",
    "5",
    "--step-x-max",
    "0.02",
    "--x-sensitivity-method",
    "implicit",
    "--sensitivity-mode",
    "gn_svd",
    "--sensitivity-solve-mode",
    "svd",
    "--sensitivity-svd-rcond",
    "1e-9",
    "--sensitivity-spectral-filter",
    "hard",
    "--sensitivity-spectral-damping",
    "0",
    "--ridge-adjoint",
    "1e-10",
    "--p2-trust-region",
    "--p2-terminal-error-first",
    "--p2-reliable-subspace",
    "--p2-reliable-task-rcond",
    "0.001",
    "--p2-min-full-task-rank",
    "2",
    "--p2-max-backtracks",
    "3",
    "--p2-backtrack-shrink",
    "0.5",
    "--p2-disable-Ix-jacobian-below-dx",
    "0.001",
    "--force-formal-20",
]


def _write_basin_gate() -> None:
    with STRICT_GATE.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = f.seek(0) or csv.DictReader(f).fieldnames
    if not fields:
        raise RuntimeError("Failed to read gate fields.")
    for row in rows:
        if row.get("kind") == "x":
            row["classification"] = "可有限使用"
            row["decision"] = "诊断放行x"
            row["limited_safe_step_abs_max"] = "0.02"
    with BASIN_GATE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_run(before: set[Path]) -> Path:
    root = REPO / "DELIVER_TO_EXP" / "formal_experiments"
    candidates = {
        path
        for path in root.glob("*A1_A2_5mm_supervised_IK_no_gravity")
        if path.is_dir()
    }
    new = sorted(candidates - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if not new:
        raise RuntimeError("No new formal experiment directory was created.")
    return new[0]


def _run_stage(name: str, extra: list[str]) -> Path:
    root = REPO / "DELIVER_TO_EXP" / "formal_experiments"
    before = {path for path in root.glob("*A1_A2_5mm_supervised_IK_no_gravity") if path.is_dir()}
    cmd = [str(PY), str(RUNNER), *COMMON_ARGS, *extra]
    log = EXP_DIR / f"{name}.log"
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=REPO, env=_env(), stdout=f, stderr=subprocess.STDOUT, check=True)
    run_root = _latest_run(before)
    (EXP_DIR / f"{name}_run_root.txt").write_text(str(run_root.resolve()) + "\n", encoding="utf-8")
    return run_root


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_PLATFORMS"] = "cuda"
    return env


def _point_summary(run_root: Path) -> dict:
    point_dir = run_root / "A1" / "smoke" / "pt_01"
    if not point_dir.exists():
        point_dir = run_root / "A1" / "small_regression" / "pt_01"
    return json.loads((point_dir / "formal_point_summary.json").read_text(encoding="utf-8"))


def _write_summary(rows: list[dict]) -> None:
    fields = [
        "stage",
        "run_root",
        "err_true_mm",
        "success_5mm",
        "stop_reason",
        "wall_s",
        "I_A",
        "x_m",
        "branch_npz_path",
        "control_gate_allowed_channels",
    ]
    with (EXP_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    _write_basin_gate()
    (EXP_DIR / "run_status.txt").write_text(
        f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"git_commit={subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}\n",
        encoding="utf-8",
    )

    basin_root = _run_stage(
        "stage1_basin_allow_x",
        [
            "--channel-gate-file",
            str(BASIN_GATE),
            "--channel-gate-missing-policy",
            "block_all",
            "--channel-gate-allow-limited",
        ],
    )
    basin = _point_summary(basin_root)
    branch_npz = basin["branch_npz_path"]
    if not Path(branch_npz).is_absolute():
        branch_npz = str((basin_root / "A1" / "smoke" / "pt_01" / branch_npz).resolve())
        if not Path(branch_npz).exists():
            branch_npz = str((basin_root / "A1" / "small_regression" / "pt_01" / basin["branch_npz_path"]).resolve())

    strict_root = _run_stage(
        "stage2_strict_from_basin",
        [
            "--initial-I-A",
            ",".join(str(float(v)) for v in basin["I_A"]),
            "--initial-x-m",
            str(float(basin["x_m"])),
            "--initial-branch-state-npz",
            str(branch_npz),
            "--channel-gate-file",
            str(STRICT_GATE),
            "--channel-gate-missing-policy",
            "block_all",
            "--channel-gate-allow-limited",
        ],
    )
    strict = _point_summary(strict_root)

    rows = []
    for stage, root, item in [
        ("stage1_basin_allow_x", basin_root, basin),
        ("stage2_strict_from_basin", strict_root, strict),
    ]:
        rows.append(
            {
                "stage": stage,
                "run_root": str(root.resolve()),
                "err_true_mm": float(item["err_true_mm"]),
                "success_5mm": bool(item["success_5mm"]),
                "stop_reason": item["stop_reason"],
                "wall_s": float(item["wall_s"]),
                "I_A": json.dumps(item["I_A"]),
                "x_m": float(item["x_m"]),
                "branch_npz_path": item["branch_npz_path"],
                "control_gate_allowed_channels": item.get("control_gate_allowed_channels"),
            }
        )
    _write_summary(rows)
    shutil.copy2(strict_root / "summary.csv", EXP_DIR / "strict_stage_formal_summary.csv")
    with (EXP_DIR / "run_status.txt").open("a", encoding="utf-8") as f:
        f.write(f"stage1_root={basin_root.resolve()}\n")
        f.write(f"stage2_root={strict_root.resolve()}\n")
        f.write(f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
