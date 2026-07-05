"""Run cross-workpoint branch-loop checks for the n2 first principal mode."""

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

import jax
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C148_DIR = ROOT / "experiment_records_ik_optimize/20260630103000_p2c148_n2_all_principal_safe_radius"
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
for path in (ROOT, P2C148_DIR, P2C174_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c148_n2_all_principal_safe_radius as p2c148  # noqa: E402
import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402


MODE_INDEX = 0
RADIUS_A = 1.0e-4


def jsonable(value: Any) -> Any:
    """Convert arrays and NumPy scalars to JSON-compatible values."""
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
    """Write rows to a CSV file."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_workpoint(base: dict[str, Any], name: str, direction: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the strict mode-1 branch-loop gate at one candidate workpoint."""
    branch0 = p2c148.build_candidate_branch(base, name, direction)
    first_row, branch1 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(
        branch0,
        step_index=1,
        mode_index=MODE_INDEX,
    )
    system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch1)
    direction_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    old_radius = p2c174.RADIUS_A
    old_mode = p2c174.MODE_INDEX
    try:
        p2c174.RADIUS_A = RADIUS_A
        p2c174.MODE_INDEX = MODE_INDEX
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            direction_summary, direction_rows = p2c174.line_metrics(branch1, system, sign, label)
            direction_summary["workpoint"] = name
            direction_summaries.append(direction_summary)
            for row in direction_rows:
                row["workpoint"] = name
                row["first_step_linear_rel_err"] = first_row.get("linear_rel_err")
                row["first_step_release_class"] = first_row.get("release_class")
                rows.append(row)
    finally:
        p2c174.RADIUS_A = old_radius
        p2c174.MODE_INDEX = old_mode
    summary = {
        "workpoint": name,
        "tested_count": len(rows),
        "pass_count": sum(1 for row in rows if row["gate_pass"]),
        "direction_summaries": direction_summaries,
        "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in rows),
        "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in rows),
        "max_return_q_norm": max(row["return_q_norm"] for row in rows),
        "max_multistart_q_diff": max(row["q_consistency_to_zero_start"] for row in rows),
        "max_continuity_norm": max(row["current_aware_continuity_norm"] for row in rows),
        "all_pass": all(row["gate_pass"] for row in rows),
    }
    return summary, rows


def plot(workpoint_summaries: list[dict[str, Any]]) -> None:
    """Plot pass status and dominant errors for each workpoint."""
    configure_matplotlib_cjk()
    labels = [item["workpoint"] for item in workpoint_summaries]
    x = np.arange(len(labels))
    colors = ["#2a9d8f" if item["all_pass"] else "#e76f51" for item in workpoint_summaries]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].bar(x, [item["pass_count"] / max(item["tested_count"], 1) for item in workpoint_summaries], color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("严格门禁通过比例")
    axes[0].set_ylabel("通过比例")
    axes[1].bar(x, [item["max_task_error"] for item in workpoint_summaries], color=colors)
    axes[1].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("最大任务线性化误差")
    axes[2].bar(x, [item["max_return_q_norm"] for item in workpoint_summaries], color=colors)
    axes[2].axhline(p2c174.RETURN_Q_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("最大回环状态误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2第1主方向跨工作点严格回环复核")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_mode1_cross_workpoint_branch_loop_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 通过数 | 最大任务误差 | 最大输出误差 | 最大回环误差 | 最大多初值差异 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["workpoint_summaries"]:
        table.append(
            f"| {item['workpoint']} | {item['pass_count']}/{item['tested_count']} | "
            f"{item['max_task_error']:.3e} | {item['max_output_error']:.3e} | "
            f"{item['max_return_q_norm']:.3e} | {item['max_multistart_q_diff']:.3e} | "
            f"{'通过' if item['all_pass'] else '不可发布'} |"
        )
    text = f"""# P2c179 两点磁控第1主方向跨工作点严格回环复核

## 实验目的

P2c174 只证明 rand_05 工作点的第 1 主方向通过严格回环门禁。本轮将同一门禁扩展到 P2c145 的 5 个两点磁控候选工作点，用于判断“一维有效子空间候选”是否具有跨工作点可重复性。

## 实验方案

每个工作点先沿第 1 主方向完成一步局部分支推进，然后在该分支附近施加正负 `{RADIUS_A:.1e} A` 电流扰动。每个扰动使用零初值、线性预测初值、反向预测初值和半预测初值。每个扰动解再回到原电流，检查是否返回同一分支。

通过门禁包括：一阶最优性、接口连续残差、中心线漂移、状态线性化误差、任务线性化误差、末端输出线性化误差、多初值一致性和回环误差。

## 结果

{chr(10).join(table)}

## 实验结论

{summary['decision']}

## 工程解释

若全部工作点通过，说明两点磁控第 1 主方向可作为分支条件下的一维局部灵敏度候选；若只有部分工作点通过，则只能给出工作点依赖的工程边界；若多数失败，则 P2c174 的结果应收缩为单工作点现象。

## 下一步计划

通过时，整理一维工程接口和外层 IK 禁用边界；失败时，按失败工作点定位主导误差块，优先判断是分支漂移、输出非线性还是载荷传播失配。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the cross-workpoint strict branch-loop audit."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    workpoint_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        work_summary, work_rows = run_workpoint(base, name, direction)
        workpoint_summaries.append(work_summary)
        rows.extend(work_rows)
        write_csv(EXP_DIR / "branch_loop_rows.partial.csv", rows)
    all_pass = all(item["all_pass"] for item in workpoint_summaries)
    if all_pass:
        status = "n2_mode1_cross_workpoint_candidate"
        decision = "5 个工作点均通过严格回环；第 1 主方向具备跨工作点一维有效子空间候选资格。"
    elif any(item["all_pass"] for item in workpoint_summaries):
        status = "n2_mode1_cross_workpoint_partial"
        decision = "仅部分工作点通过严格回环；第 1 主方向只能发布为工作点依赖的一维候选。"
    else:
        status = "n2_mode1_cross_workpoint_failed"
        decision = "第 1 主方向未形成跨工作点严格回环证据；P2c174 只能视为单工作点现象。"
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "radius_A": RADIUS_A,
        "mode_index": MODE_INDEX + 1,
        "workpoint_count": len(workpoint_summaries),
        "tested_count": len(rows),
        "pass_count": sum(1 for row in rows if row["gate_pass"]),
        "workpoint_pass_count": sum(1 for item in workpoint_summaries if item["all_pass"]),
        "publishable_1d_cross_workpoint_candidate": bool(all_pass),
        "outer_ik_allowed": False,
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "workpoint_summaries": workpoint_summaries,
        "rows": rows,
    }
    write_csv(EXP_DIR / "branch_loop_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                event = {"failure_class": "跨工作点第1主方向严格回环失败", "evidence": row}
                f.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")
    plot(workpoint_summaries)
    write_record(summary)
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c179_n2_mode1_cross_workpoint_branch_loop.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
