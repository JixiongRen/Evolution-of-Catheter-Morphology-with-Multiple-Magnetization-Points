"""Run a reproducible A1-pt13 candidate-pool probe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
FORMAL_RUNNER = REPO / "DELIVER_TO_EXP" / "run_a1_axial_circle_formal_experiment.py"
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
OUT_DIR = Path(__file__).resolve().parent


PT01_SUMMARY = (
    REPO
    / "DELIVER_TO_EXP/formal_experiments/20260714092428_A1_A2_5mm_supervised_IK_no_gravity"
    / "A1/small_regression/pt_01/formal_point_summary.json"
)
HIST_5778_SUMMARY = (
    REPO
    / "DELIVER_TO_EXP/formal_experiments/20260714092428_A1_A2_5mm_supervised_IK_no_gravity"
    / "A1/small_regression/pt_13/formal_point_summary.json"
)
HIST_5778_BRANCH = (
    REPO
    / "DELIVER_TO_EXP/formal_experiments/20260714092428_A1_A2_5mm_supervised_IK_no_gravity"
    / "A1/small_regression/pt_13/branch_states/states/eval_000043_final_iter_0043.npz"
)
HIST_2810_I = [
    2.91705600384217,
    7.596279636896001,
    1.6219165447766297,
    -3.0698419184870174,
    -3.094379908720095,
    4.995507834721822,
    -2.2362418574009153,
    -12.10113482992334,
]
HIST_2810_X = 0.1286594133256834


@dataclass(frozen=True)
class Candidate:
    name: str
    name_zh: str
    purpose_zh: str
    target_indices: list[int]
    outer_max_iter: int
    initial_I_A: list[float] | None = None
    initial_x_m: float | None = None
    initial_branch_state_npz: Path | None = None
    countable_as_cold_start: bool = False
    diagnostic_only: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _vec_csv(values: list[float]) -> str:
    return ",".join(f"{float(value):.16g}" for value in values)


def _summary_control(path: Path) -> tuple[list[float], float]:
    data = _read_json(path)
    currents = data.get("I_A")
    insertion = data.get("x_m")
    if not isinstance(currents, list) or len(currents) != 8:
        raise ValueError(f"Missing 8-coil current vector in {path}")
    if insertion is None:
        raise ValueError(f"Missing insertion depth in {path}")
    return [float(value) for value in currents], float(insertion)


def _deterministic_perturbation(base: list[float], *, scale_a: float, seed: int) -> list[float]:
    # Deterministic signs keep the probe reproducible without depending on RNG state.
    signs = [1.0 if ((seed + 3 * i) % 5) in {0, 1, 3} else -1.0 for i in range(len(base))]
    return [float(max(-20.0, min(20.0, value + scale_a * sign))) for value, sign in zip(base, signs, strict=True)]


def _build_candidates() -> list[Candidate]:
    pt01_I, pt01_x = _summary_control(PT01_SUMMARY)
    hist_5778_I, hist_5778_x = _summary_control(HIST_5778_SUMMARY)
    return [
        Candidate(
            name="cold_start",
            name_zh="标准冷启动",
            purpose_zh="标准初始形态直接求解 A1-pt13",
            target_indices=[13],
            outer_max_iter=50,
            countable_as_cold_start=True,
        ),
        Candidate(
            name="pt01_forward_control",
            name_zh="pt01 前向控制量候选",
            purpose_zh="使用 A1-pt01 成功控制量作为 A1-pt13 初值",
            target_indices=[13],
            outer_max_iter=50,
            initial_I_A=pt01_I,
            initial_x_m=pt01_x,
        ),
        Candidate(
            name="reverse_neighbor_pt14_to_pt13",
            name_zh="反向邻点候选 pt14->pt13",
            purpose_zh="先运行 pt14，再以顺序热启动机制反向进入 pt13",
            target_indices=[14, 13],
            outer_max_iter=50,
        ),
        Candidate(
            name="pt01_perturb_seed1",
            name_zh="pt01 确定性扰动候选 1",
            purpose_zh="在 pt01 控制量附近加入 0.5 A 确定性扰动",
            target_indices=[13],
            outer_max_iter=50,
            initial_I_A=_deterministic_perturbation(pt01_I, scale_a=0.5, seed=1),
            initial_x_m=pt01_x,
        ),
        Candidate(
            name="pt01_perturb_seed2",
            name_zh="pt01 确定性扰动候选 2",
            purpose_zh="在 pt01 控制量附近加入 0.5 A 确定性扰动",
            target_indices=[13],
            outer_max_iter=50,
            initial_I_A=_deterministic_perturbation(pt01_I, scale_a=0.5, seed=2),
            initial_x_m=pt01_x,
        ),
        Candidate(
            name="historical_5778_control_only",
            name_zh="历史 5.778 mm 控制量重放",
            purpose_zh="只给定历史近阈值控制量，不加载分支状态",
            target_indices=[13],
            outer_max_iter=0,
            initial_I_A=hist_5778_I,
            initial_x_m=hist_5778_x,
            diagnostic_only=True,
        ),
        Candidate(
            name="historical_5778_branch_state",
            name_zh="历史 5.778 mm 分支状态重放",
            purpose_zh="给定历史控制量并加载历史平衡分支状态",
            target_indices=[13],
            outer_max_iter=0,
            initial_I_A=hist_5778_I,
            initial_x_m=hist_5778_x,
            initial_branch_state_npz=HIST_5778_BRANCH,
            diagnostic_only=True,
        ),
        Candidate(
            name="historical_2810_control_only",
            name_zh="历史 2.810 mm 控制量重放",
            purpose_zh="只给定更早历史成功控制量，不加载分支状态",
            target_indices=[13],
            outer_max_iter=0,
            initial_I_A=HIST_2810_I,
            initial_x_m=HIST_2810_X,
            diagnostic_only=True,
        ),
    ]


def _command(candidate: Candidate, *, timeout_s: float) -> list[str]:
    cmd = [
        str(PYTHON),
        str(FORMAL_RUNNER),
        "--disable-gravity",
        "--eps-p-mm",
        "5",
        "--force-formal-20",
        "--catheters",
        "A1",
        "--target-indices",
        ",".join(str(index) for index in candidate.target_indices),
        "--per-point-timeout-s",
        f"{float(timeout_s):.12g}",
        "--outer-max-iter",
        str(int(candidate.outer_max_iter)),
        "--p2-trust-region",
        "--p2-terminal-error-first",
        "--p2-reliable-subspace",
        "--p2-max-backtracks",
        "3",
    ]
    if candidate.initial_I_A is not None:
        cmd.extend(["--initial-I-A", _vec_csv(candidate.initial_I_A)])
    if candidate.initial_x_m is not None:
        cmd.extend(["--initial-x-m", f"{float(candidate.initial_x_m):.16g}"])
    if candidate.initial_branch_state_npz is not None:
        cmd.extend(["--initial-branch-state-npz", str(candidate.initial_branch_state_npz)])
    return cmd


def _find_latest_run(before: set[Path]) -> Path | None:
    roots = set((REPO / "DELIVER_TO_EXP" / "formal_experiments").glob("*_A1_A2_5mm_supervised_IK_no_gravity"))
    new_roots = sorted(roots - before, key=lambda path: path.name)
    return new_roots[-1] if new_roots else None


def _target_row(run_root: Path | None, target_index: int = 13) -> dict[str, Any]:
    if run_root is None:
        return {}
    summary_path = run_root / "summary.csv"
    if not summary_path.exists():
        return {"run_root": str(run_root), "summary_exists": False}
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if row.get("catheter_type") == "A1" and str(row.get("target_index")) == str(target_index):
            row["run_root"] = str(run_root)
            row["summary_exists"] = True
            return row
    return {"run_root": str(run_root), "summary_exists": True, "target_found": False}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_commands(path: Path, candidates: list[Candidate], *, timeout_s: float) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for candidate in candidates:
        quoted = " ".join(_shell_quote(part) for part in _command(candidate, timeout_s=timeout_s))
        lines.append(f"# {candidate.name}: {candidate.name_zh}")
        lines.append(f"XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda {quoted}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "._/-=,:+" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_record(path: Path, rows: list[dict[str, Any]], *, dry_run: bool) -> None:
    lines = [
        "# A1-pt13 候选池执行记录",
        "",
        "## 一、目的",
        "",
        "本脚本用于把 A1-pt13 的候选池运行从手动命令固化为可复现实验流程。候选池覆盖标准冷启动、前向邻点、反向邻点、确定性扰动、历史控制量重放和历史分支状态重放。",
        "",
        "## 二、执行状态",
        "",
        f"- dry_run：`{dry_run}`",
        f"- 候选数量：`{len(rows)}`",
        "",
        "## 三、候选摘要",
        "",
        "| 候选 | 中文名 | 目标点序列 | 真实末端误差(mm) | 5mm | 10mm | 停止原因 | 运行目录 |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        err = row.get("err_true_mm")
        err_text = "" if err in (None, "") else f"{float(err):.3f}" if _is_float(err) else str(err)
        lines.append(
            f"| {row.get('candidate')} | {row.get('candidate_zh')} | {row.get('target_indices')} | "
            f"{err_text} | {row.get('success_5mm')} | {row.get('success_10mm')} | "
            f"{row.get('ik_stop_reason') or row.get('report_stop_reason') or row.get('stop_reason')} | "
            f"{row.get('run_root')} |"
        )
    lines.extend(
        [
            "",
            "## 四、判读原则",
            "",
            "1. 历史分支状态重放只证明该平衡分支存在，不能计入冷启动成功率。",
            "2. 只给控制量重放若失败，说明历史结果不能由控制量单独复现。",
            "3. 非分支状态候选若进入 5 mm，必须再做只给控制量的重放确认。",
            "4. 候选停在 5--10 mm 时，应进入小信赖域精修或标记为候选入盆不足。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only write commands and source tables.")
    parser.add_argument("--candidate", action="append", default=[], help="Run only selected candidate names.")
    parser.add_argument("--timeout-s", type=float, default=900.0, help="Per-target timeout passed to the formal runner.")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _build_candidates()
    if args.candidate:
        wanted = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate.name in wanted]
    _write_commands(OUT_DIR / "commands.sh", candidates, timeout_s=float(args.timeout_s))
    candidate_rows: list[dict[str, Any]] = []
    formal_root = REPO / "DELIVER_TO_EXP" / "formal_experiments"
    for candidate in candidates:
        base_row = {
            "candidate": candidate.name,
            "candidate_zh": candidate.name_zh,
            "purpose_zh": candidate.purpose_zh,
            "target_indices": ",".join(str(index) for index in candidate.target_indices),
            "outer_max_iter": int(candidate.outer_max_iter),
            "countable_as_cold_start": bool(candidate.countable_as_cold_start),
            "diagnostic_only": bool(candidate.diagnostic_only),
            "initial_branch_state_npz": str(candidate.initial_branch_state_npz) if candidate.initial_branch_state_npz else "",
        }
        if args.dry_run:
            candidate_rows.append(base_row | {"status": "dry_run"})
            continue
        before = set(formal_root.glob("*_A1_A2_5mm_supervised_IK_no_gravity"))
        log_path = OUT_DIR / f"{candidate.name}.log"
        cmd = _command(candidate, timeout_s=float(args.timeout_s))
        env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda"}
        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND:\n")
            log_file.write(" ".join(_shell_quote(part) for part in cmd))
            log_file.write("\n\nSTDOUT_STDERR:\n")
            completed = subprocess.run(cmd, cwd=REPO, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        wall_s = float(time.perf_counter() - t0)
        run_root = _find_latest_run(before)
        row = _target_row(run_root, target_index=13)
        candidate_rows.append(
            base_row
            | row
            | {
                "candidate_wall_s": wall_s,
                "candidate_return_code": int(completed.returncode),
                "candidate_log": str(log_path.resolve()),
            }
        )
    _write_csv(OUT_DIR / "candidate_pool_run_summary.csv", candidate_rows)
    _write_record(OUT_DIR / "record.md", candidate_rows, dry_run=bool(args.dry_run))
    print(json.dumps({"out_dir": str(OUT_DIR), "n_candidates": len(candidate_rows), "dry_run": bool(args.dry_run)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
