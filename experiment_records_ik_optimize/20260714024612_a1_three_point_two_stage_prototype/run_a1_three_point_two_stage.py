"""Run a two-stage A1 three-point supervised IK prototype."""

from __future__ import annotations

import csv
import json
import os
import shutil
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python")
FORMAL_SCRIPT = REPO / "DELIVER_TO_EXP/run_a1_axial_circle_formal_experiment.py"
OUT_ROOT = REPO / "DELIVER_TO_EXP/formal_experiments"
POINTS = [1, 12, 13]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _formal_run_dirs_since(start_time: float) -> list[Path]:
    candidates: list[Path] = []
    for path in OUT_ROOT.glob("*_A1_A2_5mm_supervised_IK_no_gravity"):
        summary = path / "summary.json"
        if summary.exists() and summary.stat().st_mtime >= start_time - 1.0:
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _point_dir(run_root: Path, point: int) -> Path:
    path = run_root / f"A1/smoke/pt_{point:02d}"
    if path.exists():
        return path
    matches = list((run_root / "A1").glob(f"*/pt_{point:02d}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find pt_{point:02d} under {run_root}")
    return matches[0]


def _load_summary(run_root: Path, point: int) -> dict[str, Any]:
    path = _point_dir(run_root, point) / "formal_point_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_gate(path: Path, *, allow_i5: bool = False, allow_x: bool = False) -> None:
    rows: list[dict[str, str]] = []
    if allow_i5:
        rows.append(
            {
                "point_key": "A1_pt13",
                "kind": "I",
                "channel": "5",
                "decision": "有限使用候选",
                "limited_safe_step_abs_max": "0.125",
                "notes": "I5 limited-use channel from A1-pt13 policy reaudit",
            }
        )
    if allow_x:
        rows.append(
            {
                "point_key": "A1_pt13",
                "kind": "x",
                "channel": "-1",
                "decision": "有限使用候选",
                "limited_safe_step_abs_max": "0.000125",
                "notes": "Empirical x micro-search, not production sensitivity approval",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["point_key", "kind", "channel", "decision", "limited_safe_step_abs_max", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_formal(
    *,
    point: int,
    stage: str,
    extra_args: list[str],
    timeout_s: float,
) -> tuple[Path, dict[str, Any], float, int]:
    stage_dir = EXP_DIR / "runs" / f"pt_{point:02d}" / stage
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
        str(point),
        "--outer-max-iter",
        "50",
        "--per-point-timeout-s",
        f"{timeout_s:.12g}",
        "--p2-trust-region",
        "--p2-terminal-error-first",
        "--p2-reliable-subspace",
        "--p2-max-backtracks",
        "3",
        "--p2-backtrack-shrink",
        "0.5",
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
        raise RuntimeError(f"No formal output found for pt_{point:02d} stage {stage}")
    run_root = dirs[-1]
    summary = _load_summary(run_root, point)
    snapshot = stage_dir / "formal_point_snapshot"
    shutil.copytree(_point_dir(run_root, point), snapshot, dirs_exist_ok=True)
    return run_root, summary, wall_s, int(completed.returncode)


def _summary_row(point: int, stage: str, run_root: Path, summary: dict[str, Any], wall_s: float, return_code: int) -> dict[str, Any]:
    return {
        "point": int(point),
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_record(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# A1 三点两阶段有监督逆运动学原型实验",
        "",
        "## 实验目的",
        "",
        "本轮用于验证两阶段外层策略的最小闭环：第一阶段无硬门控全局入盆；第二阶段仅对进入局部邻域的点执行同分支小步精修。该实验不改变正运动学模型、网格、容差、真实末端定义和控制边界。",
        "",
        "## 实验设置",
        "",
        "- 点位：A1-pt01、A1-pt12、A1-pt13。",
        "- 重力关闭，`M_list=5`，FK 容差 `1e-5`。",
        "- 成功阈值：真实刚性远端误差 `<5 mm`。",
        "- 全局入盆阶段：无硬门控，启用信赖域、真实末端优先接受和可靠子空间。",
        "- 同分支精修阶段：仅在全局阶段达到 `10 mm` 邻域但未进入 `5 mm` 时执行；推进量作为经验性小步搜索变量，不视作生产灵敏度放行。",
        "",
        "## 结果摘要",
        "",
        "| 点位 | 阶段 | 最终误差(mm) | 历史最小误差(mm) | 是否进入5mm | 终止原因 | 接受/拒绝 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        err = float(row["err_true_mm"]) if row.get("err_true_mm") is not None else float("nan")
        best = float(row["best_err_true_mm"]) if row.get("best_err_true_mm") is not None else float("nan")
        lines.append(
            f"| pt{int(row['point']):02d} | {row['stage']} | {err:.6f} | {best:.6f} | "
            f"{row.get('success_5mm')} | {row.get('stop_reason')} | {row.get('accept_count')}/{row.get('reject_count')} |"
        )
    lines.extend(
        [
            "",
            "## 初步判读",
            "",
            "pt01 用于检查旧成功点是否退化；pt13 用于检查同分支推进量小步精修是否能复现上一轮进入 `5 mm` 的经验效果；pt12 若仍远离目标，则暂不应依赖小步精修，而应继续分支池或可达性诊断。",
        ]
    )
    (EXP_DIR / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    (EXP_DIR / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda {PYTHON} {Path(__file__).resolve()}\n",
        encoding="utf-8",
    )
    gates = EXP_DIR / "gates"
    x_gate = gates / "pt13_x_micro.csv"
    _write_gate(x_gate, allow_x=True)
    rows: list[dict[str, Any]] = []
    for point in POINTS:
        run_root, summary, wall_s, rc = _run_formal(
            point=point,
            stage="global_basin",
            extra_args=["--step-I-max", "5", "--step-x-max", "0.02"],
            timeout_s=1800.0,
        )
        rows.append(_summary_row(point, "global_basin", run_root, summary, wall_s, rc))
        err = float(summary.get("err_true_mm", float("inf")))
        branch_npz = summary.get("branch_npz_path")
        currents = summary.get("I_A")
        x_m = summary.get("x_m")
        if 5.0 < err <= 10.0 and branch_npz and currents is not None and x_m is not None:
            run_root, refine_summary, refine_wall_s, refine_rc = _run_formal(
                point=point,
                stage="same_branch_x_micro",
                extra_args=[
                    "--step-I-max",
                    "0.125",
                    "--step-x-max",
                    "0.000125",
                    "--channel-gate-file=" + str(x_gate.resolve()),
                    "--channel-gate-missing-policy",
                    "block_all",
                    "--channel-gate-allow-limited",
                    "--initial-I-A=" + ",".join(f"{float(v):.12g}" for v in currents),
                    f"--initial-x-m={float(x_m):.12g}",
                    "--initial-branch-state-npz=" + str(Path(branch_npz).resolve()),
                ],
                timeout_s=1200.0,
            )
            rows.append(_summary_row(point, "same_branch_x_micro", run_root, refine_summary, refine_wall_s, refine_rc))
    _write_rows(EXP_DIR / "two_stage_summary.csv", rows)
    (EXP_DIR / "two_stage_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": EXP_DIR.name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "git_commit": _git_commit(),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_record(rows)


if __name__ == "__main__":
    main()
