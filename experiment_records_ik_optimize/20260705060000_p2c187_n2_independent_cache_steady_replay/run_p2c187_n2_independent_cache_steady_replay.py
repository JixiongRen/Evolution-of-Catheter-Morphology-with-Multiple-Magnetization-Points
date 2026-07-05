"""Measure warm and steady replay times for independent n2 branch caches."""

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
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
P2C186_DIR = ROOT / "experiment_records_ik_optimize/20260705043000_p2c186_n2_independent_cache_replay"
for path in (ROOT, P2C174_DIR, P2C186_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402
import run_p2c186_n2_independent_cache_replay as p2c186  # noqa: E402


MODE_INDEX = 0
RADIUS_A = 1.0e-4


def jsonable(value: Any) -> Any:
    """Convert nested values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def replay_with_timing(branch: dict[str, Any], pass_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay strict gates and measure system-build and gate time."""
    old_radius = p2c174.RADIUS_A
    old_mode = p2c174.MODE_INDEX
    p2c174.RADIUS_A = RADIUS_A
    p2c174.MODE_INDEX = MODE_INDEX
    try:
        t_system = time.perf_counter()
        system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch)
        system_s = time.perf_counter() - t_system
        t_gate = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            _direction_summary, direction_rows = p2c174.line_metrics(branch, system, sign, label)
            rows.extend(direction_rows)
        gate_s = time.perf_counter() - t_gate
    finally:
        p2c174.RADIUS_A = old_radius
        p2c174.MODE_INDEX = old_mode
    for row in rows:
        row["pass_name"] = pass_name
    summary = {
        "pass_name": pass_name,
        "tested_count": len(rows),
        "pass_count": sum(1 for row in rows if row["gate_pass"]),
        "system_build_s": float(system_s),
        "gate_s": float(gate_s),
        "solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in rows),
        "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in rows),
        "max_return_q_norm": max(row["return_q_norm"] for row in rows),
    }
    return summary, rows


def run_workpoint(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run warm and steady replay for one cached workpoint."""
    t_load = time.perf_counter()
    branch, load_rebuild_s = p2c186.load_branch(name)
    load_total_s = time.perf_counter() - t_load
    warm_summary, warm_rows = replay_with_timing(branch, "warm")
    steady_summary, steady_rows = replay_with_timing(branch, "steady")
    rows = []
    for row in warm_rows + steady_rows:
        row["workpoint"] = name
        rows.append(row)
    work_summary = {
        "workpoint": name,
        "load_rebuild_s": float(load_rebuild_s),
        "load_total_s": float(load_total_s),
        "warm": warm_summary,
        "steady": steady_summary,
        "speedup_gate": float(warm_summary["gate_s"] / max(steady_summary["gate_s"], 1.0e-12)),
        "speedup_system": float(warm_summary["system_build_s"] / max(steady_summary["system_build_s"], 1.0e-12)),
    }
    return work_summary, rows


def plot(summary: dict[str, Any]) -> None:
    """Plot warm and steady timing."""
    configure_matplotlib_cjk()
    labels = [item["workpoint"] for item in summary["workpoint_summaries"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(x - 0.18, [item["warm"]["system_build_s"] for item in summary["workpoint_summaries"]], width=0.36, label="预热")
    axes[0].bar(x + 0.18, [item["steady"]["system_build_s"] for item in summary["workpoint_summaries"]], width=0.36, label="稳态")
    axes[0].set_yscale("log")
    axes[0].set_title("最优性系统构建耗时")
    axes[1].bar(x - 0.18, [item["warm"]["gate_s"] for item in summary["workpoint_summaries"]], width=0.36, label="预热")
    axes[1].bar(x + 0.18, [item["steady"]["gate_s"] for item in summary["workpoint_summaries"]], width=0.36, label="稳态")
    axes[1].set_yscale("log")
    axes[1].set_title("门禁回放耗时")
    axes[2].bar(x, [item["steady"]["max_task_error"] for item in summary["workpoint_summaries"]], color="#2a9d8f")
    axes[2].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("稳态最大任务误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("n2独立缓存预热与稳态回放")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_independent_cache_steady_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 预热系统/s | 稳态系统/s | 预热门禁/s | 稳态门禁/s | 稳态通过数 | 稳态最大任务误差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["workpoint_summaries"]:
        table.append(
            f"| {item['workpoint']} | {item['warm']['system_build_s']:.2f} | {item['steady']['system_build_s']:.2f} | "
            f"{item['warm']['gate_s']:.2f} | {item['steady']['gate_s']:.2f} | "
            f"{item['steady']['pass_count']}/{item['steady']['tested_count']} | {item['steady']['max_task_error']:.3e} |"
        )
    text = f"""# P2c187 两点磁控独立缓存稳态回放计时

## 实验目的

P2c186 已证明独立缓存回放保持 `40/40` 通过，但回放耗时包含首次编译开销。本轮在同一进程内对每个工作点连续执行两次回放：第一遍作为预热，第二遍作为稳态计时，用于分离首次编译与稳定运行耗时。

## 实验方案

从 P2c186 缓存读取数组和物理上下文，重建 JAX 函数。每个工作点先执行一次完整严格门禁作为预热，再立即执行一次相同门禁作为稳态测量。两轮均记录最优性系统构建耗时、门禁回放耗时、重求解净耗时和门禁通过数。

## 结果

{chr(10).join(table)}

## 总结指标

- 稳态通过数：{summary['steady_pass_count']} / {summary['steady_tested_count']}。
- 预热总耗时：{summary['warm_total_s']:.2f} s。
- 稳态总耗时：{summary['steady_total_s']:.2f} s。
- 稳态重求解净耗时：{summary['steady_solve_s']:.4f} s。
- 运行设备：{', '.join(summary['jax_devices'])}。

## 实验结论

{summary['decision']}

## 下一步计划

若稳态耗时显著低于预热耗时，应将报告中的产品级耗时改为“预热后稳态耗时”，并继续做批量多初值求解。若稳态仍很慢，下一步应直接优化门禁回放中的重复 `least_squares` 调用和 Python 循环。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run warm and steady replay timing."""
    t0 = time.perf_counter()
    names = [path.name.split("_mode1_branch_cache", 1)[0] for path in sorted(p2c186.CACHE_DIR.glob("*_mode1_branch_cache.npz"))]
    if not names:
        raise RuntimeError("P2c186 cache files are missing. Run P2c186 build first.")
    work_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name in names:
        work_summary, work_rows = run_workpoint(name)
        work_summaries.append(work_summary)
        rows.extend(work_rows)
    steady_pass = sum(item["steady"]["pass_count"] for item in work_summaries)
    steady_tested = sum(item["steady"]["tested_count"] for item in work_summaries)
    status = "steady_replay_passed" if steady_pass == steady_tested else "steady_replay_failed"
    decision = (
        "稳态回放保持全部门禁通过；可将预热后稳态耗时作为缓存产品化参考。"
        if steady_pass == steady_tested
        else "稳态回放未保持全部门禁通过；不能进入效率产品化。"
    )
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "workpoint_count": len(work_summaries),
        "steady_pass_count": steady_pass,
        "steady_tested_count": steady_tested,
        "wall_s": float(time.perf_counter() - t0),
        "warm_total_s": float(sum(item["warm"]["system_build_s"] + item["warm"]["gate_s"] for item in work_summaries)),
        "steady_total_s": float(sum(item["steady"]["system_build_s"] + item["steady"]["gate_s"] for item in work_summaries)),
        "warm_solve_s": float(sum(item["warm"]["solve_steady_s"] for item in work_summaries)),
        "steady_solve_s": float(sum(item["steady"]["solve_steady_s"] for item in work_summaries)),
        "jax_devices": [str(device) for device in jax.devices()],
        "workpoint_summaries": work_summaries,
        "rows": rows,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(EXP_DIR / "steady_replay_rows.csv", rows)
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if row["pass_name"] == "steady" and not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "稳态缓存回放门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c187_n2_independent_cache_steady_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "steady_pass_count": steady_pass,
                "steady_tested_count": steady_tested,
                "warm_total_s": summary["warm_total_s"],
                "steady_total_s": summary["steady_total_s"],
                "jax_devices": summary["jax_devices"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
