from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
for path in (ROOT, P2C174_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c174_n2_1d_branch_loop_gate as p2c174


MODE_INDEX = 1
RADII_A = (1.0e-4, 2.5e-4, 5.0e-4, 1.0e-3)


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot mode-2 branch-loop gate errors."""
    configure_matplotlib_cjk()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for direction in ("正向", "反向"):
        selected = [
            row for row in rows
            if row["direction"] == direction and row["start"] == "zero"
        ]
        selected.sort(key=lambda row: row["radius_A"])
        ax.plot(
            [row["radius_A"] for row in selected],
            [row["task_pred_vs_actual_rel_err"] for row in selected],
            marker="o",
            label=direction,
        )
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="门限")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("电流半径/A")
    ax.set_ylabel("任务线性化误差")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle("n2 第 2 主方向回环门禁")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_mode2_branch_loop_gate_cn.png", dpi=200)
    plt.close(fig)


def summarize(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    """Summarize mode-2 branch-loop rows."""
    passed = [row for row in rows if row["gate_pass"]]
    by_radius = {}
    for radius in RADII_A:
        selected = [row for row in rows if abs(row["radius_A"] - radius) < 1.0e-15]
        by_radius[f"{radius:.1e}"] = {
            "tested_count": len(selected),
            "pass_count": sum(row["gate_pass"] for row in selected),
            "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in selected),
            "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in selected),
            "max_return_q_norm": max(row["return_q_norm"] for row in selected),
        }
    if passed:
        status = "mode2_branch_loop_partial"
        decision = "第 2 主方向存在局部通过样本，但不是完整半径范围稳定候选。"
    else:
        status = "mode2_branch_loop_failed"
        decision = "第 2 主方向未通过严格回环门禁；n2 当前工程边界收敛为一维有效子空间。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "mode_index": 2,
        "tested_count": len(rows),
        "pass_count": len(passed),
        "radius_summary": by_radius,
        "publishable_mode2": bool(passed),
        "publishable_2d_candidate": False,
        "outer_ik_allowed": False,
        "wall_s": float(wall_s),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in p2c174.jax.devices()],
        "rows": rows,
    }


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 半径/A | 方向 | 初值 | 任务误差 | 输出误差 | 回环误差 | 结论 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        table.append(
            f"| {row['radius_A']:.1e} | {row['direction']} | {row['start']} | "
            f"{row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['output_linear_at_actual_q_rel_err']:.3e} | "
            f"{row['return_q_norm']:.3e} | {'通过' if row['gate_pass'] else '不可发布'} |"
        )
    text = f"""# P2c176 两点磁控 n2 第 2 主方向回环门禁

## 实验目的

P2c175 将第 2 主方向列为待复核候选。本轮对第 2 主方向执行与 P2c174 相同的正负扰动、多初值和回环门禁，判断 n2 是否能从一维有效子空间扩展到二维有效子空间。

## 实验方案

固定同一工作点和第一步后局部分支。主方向取第 2 主方向。半径取 `{list(RADII_A)}`。每个半径执行正负扰动，每个扰动使用零初值、预测初值、反向预测初值和半预测初值，并做回到原电流的回环检查。

## 结果

{chr(10).join(table)}

## 实验结论

{summary['decision']}

## 下一步计划

若第 2 主方向未通过，应停止推进 n2 二维候选，将当前 n2 工程边界收敛为第 1 主方向一维有效子空间，并转向跨工作点复核一维候选或回到单点磁控有效子空间。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the mode-2 branch-loop gate."""
    t0 = time.perf_counter()
    base = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch1)
    old_mode = p2c174.MODE_INDEX
    old_radius = p2c174.RADIUS_A
    rows: list[dict[str, Any]] = []
    try:
        p2c174.MODE_INDEX = MODE_INDEX
        for radius in RADII_A:
            p2c174.RADIUS_A = radius
            for sign, label in ((1.0, "正向"), (-1.0, "反向")):
                _summary, local_rows = p2c174.line_metrics(branch1, system, sign, label)
                for row in local_rows:
                    row["radius_A"] = float(radius)
                    rows.append(row)
    finally:
        p2c174.MODE_INDEX = old_mode
        p2c174.RADIUS_A = old_radius
    summary = summarize(rows, time.perf_counter() - t0)
    summary["case"] = direction_name
    summary["path"] = "mode2_positive_negative_multistart_return_loop"
    write_csv(EXP_DIR / "mode2_branch_loop_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                f.write(json.dumps({"failure_class": "第 2 主方向回环门禁失败", "evidence": jsonable(row)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705031500_p2c176_n2_mode2_branch_loop_gate/run_p2c176_n2_mode2_branch_loop_gate.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
