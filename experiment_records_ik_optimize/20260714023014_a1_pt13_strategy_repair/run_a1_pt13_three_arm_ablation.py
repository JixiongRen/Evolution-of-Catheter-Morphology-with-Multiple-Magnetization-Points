"""Run A1-pt13 same-branch three-arm refinement ablation."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
FORMAL_SCRIPT = REPO / "DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py"
SOURCE_STATE = (
    REPO
    / "DELIVER_TO_EXP/formal_experiments/20260713232320_A1_A2_5mm_supervised_IK_no_gravity"
    / "A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"
)
OUT_ROOT = REPO / "DELIVER_TO_EXP/formal_experiments"


def _read_state_control() -> tuple[list[float], float]:
    data = np.load(SOURCE_STATE, allow_pickle=True)
    currents = [float(v) for v in np.asarray(data["I_A"], dtype=float).reshape(-1).tolist()]
    x_m = float(np.asarray(data["x_m"], dtype=float).reshape(-1)[0])
    if len(currents) != 8:
        raise ValueError(f"Expected 8 currents, got {len(currents)}")
    return currents, x_m


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _write_gate(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "kind", "channel", "classification", "evidence", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _formal_run_dirs_since(start_time: float) -> list[Path]:
    if not OUT_ROOT.exists():
        return []
    candidates: list[Path] = []
    for path in OUT_ROOT.glob("*_A1_A2_5mm_supervised_IK_no_gravity"):
        summary = path / "summary.json"
        if summary.exists() and summary.stat().st_mtime >= start_time - 1.0:
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _load_point_summary(run_root: Path) -> dict[str, Any]:
    point_summary = run_root / "A1/small_regression/pt_13/formal_point_summary.json"
    if not point_summary.exists():
        point_summary = run_root / "A1/formal_20pt/pt_13/formal_point_summary.json"
    if not point_summary.exists():
        matches = list((run_root / "A1").glob("*/pt_13/formal_point_summary.json"))
        if matches:
            point_summary = matches[0]
    if not point_summary.exists():
        raise FileNotFoundError(f"Cannot find pt_13 summary under {run_root}")
    return json.loads(point_summary.read_text(encoding="utf-8"))


def _run_arm(name: str, gate_path: Path, step_i_max: float, step_x_max: float) -> dict[str, Any]:
    currents, x_m = _read_state_control()
    arm_dir = EXP_DIR / "arms" / name
    arm_dir.mkdir(parents=True, exist_ok=True)
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
        "20",
        "--per-point-timeout-s",
        "1800",
        "--step-I-max",
        f"{step_i_max:.12g}",
        "--step-x-max",
        f"{step_x_max:.12g}",
        "--initial-I-A=" + ",".join(f"{v:.12g}" for v in currents),
        f"--initial-x-m={x_m:.12g}",
        "--initial-branch-state-npz=" + str(SOURCE_STATE),
        "--channel-gate-file=" + str(gate_path),
        "--channel-gate-missing-policy",
        "block_all",
        "--channel-gate-allow-limited",
        "--p2-trust-region",
        "--p2-terminal-error-first",
        "--p2-reliable-subspace",
        "--p2-max-backtracks",
        "3",
        "--p2-backtrack-shrink",
        "0.5",
    ]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda"}
    (arm_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + " ".join(shlex.quote(part) for part in cmd)
        + "\n",
        encoding="utf-8",
    )
    start = time.time()
    completed = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        stdout=(arm_dir / "stdout.log").open("w", encoding="utf-8"),
        stderr=(arm_dir / "stderr.log").open("w", encoding="utf-8"),
        text=True,
        check=False,
    )
    end = time.time()
    run_dirs = _formal_run_dirs_since(start)
    if not run_dirs:
        raise RuntimeError(f"No formal run directory found for arm {name}")
    run_root = run_dirs[-1]
    point_summary = _load_point_summary(run_root)
    arm_summary = {
        "arm": name,
        "return_code": int(completed.returncode),
        "wall_s_orchestrator": float(end - start),
        "formal_run_root": str(run_root.resolve()),
        "gate_path": str(gate_path.resolve()),
        "step_I_max_A": float(step_i_max),
        "step_x_max_m": float(step_x_max),
        "source_state": str(SOURCE_STATE.resolve()),
        "source_I_A": currents,
        "source_x_m": x_m,
        "err_true_mm": point_summary.get("err_true_mm"),
        "best_err_true_mm": point_summary.get("best_err_true_mm"),
        "success_5mm": point_summary.get("success_5mm"),
        "stop_reason": point_summary.get("stop_reason"),
        "accept_count": point_summary.get("accept_count"),
        "reject_count": point_summary.get("reject_count"),
        "outer_iter_count": point_summary.get("outer_iter_count"),
        "runtime_platform": point_summary.get("runtime_platform"),
        "runtime_device_kind": point_summary.get("runtime_device_kind"),
        "control_gate_allowed_channels": point_summary.get("control_gate_allowed_channels"),
        "control_gate_blocked_channels": point_summary.get("control_gate_blocked_channels"),
        "initial_branch_state_hash": point_summary.get("initial_branch_state_hash"),
        "initial_branch_I_delta_inf_A": point_summary.get("initial_branch_I_delta_inf_A"),
        "initial_branch_x_delta_abs_m": point_summary.get("initial_branch_x_delta_abs_m"),
        "z_star_hash": point_summary.get("z_star_hash"),
        "centerline_hash": point_summary.get("centerline_hash"),
        "branch_npz_path": point_summary.get("branch_npz_path"),
        "centerline_npy": point_summary.get("centerline_npy"),
        "point_summary": str((run_root / "A1/small_regression/pt_13/formal_point_summary.json").resolve()),
    }
    (arm_dir / "arm_summary.json").write_text(json.dumps(arm_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return arm_summary


def _write_summary(rows: list[dict[str, Any]]) -> None:
    csv_path = EXP_DIR / "ablation_summary.csv"
    keys = [
        "arm",
        "err_true_mm",
        "best_err_true_mm",
        "success_5mm",
        "stop_reason",
        "accept_count",
        "reject_count",
        "outer_iter_count",
        "step_I_max_A",
        "step_x_max_m",
        "runtime_platform",
        "runtime_device_kind",
        "formal_run_root",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    summary = {
        "experiment_id": EXP_DIR.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source_state": str(SOURCE_STATE.resolve()),
        "arms": rows,
    }
    (EXP_DIR / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_record(rows: list[dict[str, Any]]) -> None:
    baseline_mm = 5.778239956594556
    lines = [
        "# A1-pt13 同分支三臂精修消融实验",
        "",
        "## 实验目的",
        "",
        "本轮实验用于判断 A1-pt13 近解分支附近的有限控制自由度是否具有实际精修价值。实验只评估控制效益，不把推进量通道的经验改善解释为灵敏度已通过生产门禁。",
        "",
        "## 固定条件",
        "",
        "- 重力关闭。",
        "- `M_list=5`。",
        "- 正运动学容差为 `1e-5`。",
        "- 成功判据为最后刚性段远端真实末端误差小于 `5 mm`。",
        "- 三臂均从同一控制量和同一内部平衡状态独立起跑。",
        f"- 源近解真实末端误差：`{baseline_mm:.6f} mm`。",
        "",
        "## 三臂设置",
        "",
        "- I5-only：只放行 I5，单步电流上限 `0.125 A`。",
        "- x-only：只放行推进量，单步推进上限 `0.125 mm`，仅作经验消融。",
        "- I5+x：放行 I5 和推进量，使用同样步长上限。",
        "",
        "## 结果摘要",
        "",
        "| 组别 | 最终误差(mm) | 历史最小误差(mm) | 是否进入5mm | 终止原因 | 接受/拒绝 |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        err = float(row["err_true_mm"]) if row.get("err_true_mm") is not None else float("nan")
        best = float(row["best_err_true_mm"]) if row.get("best_err_true_mm") is not None else float("nan")
        lines.append(
            f"| {row['arm']} | {err:.6f} | {best:.6f} | {row.get('success_5mm')} | "
            f"{row.get('stop_reason')} | {row.get('accept_count')}/{row.get('reject_count')} |"
        )
    lines.extend(
        [
            "",
            "## 初步解释规则",
            "",
            "若 I5+x 相对 I5-only 稳定改善超过 `0.3 mm`，只能说明额外自由度具有经验控制价值，不能说明推进量灵敏度已经可作为生产雅可比列。若三臂均改善有限，应优先复核目标函数尺度、接受准则、同分支局部可达性和灵敏度公式链。",
        ]
    )
    (EXP_DIR / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_STATE.exists():
        raise FileNotFoundError(str(SOURCE_STATE))
    gates = EXP_DIR / "gates"
    _write_gate(
        gates / "i5_only.csv",
        [{"run": "A1_pt13", "kind": "I", "channel": 5, "classification": "可有限使用", "evidence": EXP_DIR.name}],
    )
    _write_gate(
        gates / "x_only.csv",
        [{"run": "A1_pt13", "kind": "x", "channel": "", "classification": "可有限使用", "evidence": EXP_DIR.name, "notes": "经验消融，不代表生产放行"}],
    )
    _write_gate(
        gates / "i5_plus_x.csv",
        [
            {"run": "A1_pt13", "kind": "I", "channel": 5, "classification": "可有限使用", "evidence": EXP_DIR.name},
            {"run": "A1_pt13", "kind": "x", "channel": "", "classification": "可有限使用", "evidence": EXP_DIR.name, "notes": "经验消融，不代表生产放行"},
        ],
    )
    commands = [
        "XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda "
        f"{PYTHON} {Path(__file__).resolve()}"
    ]
    (EXP_DIR / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
    rows = [
        _run_arm("I5_only", gates / "i5_only.csv", step_i_max=0.125, step_x_max=0.000125),
        _run_arm("x_only", gates / "x_only.csv", step_i_max=0.125, step_x_max=0.000125),
        _run_arm("I5_plus_x", gates / "i5_plus_x.csv", step_i_max=0.125, step_x_max=0.000125),
    ]
    _write_summary(rows)
    _write_record(rows)


if __name__ == "__main__":
    main()
