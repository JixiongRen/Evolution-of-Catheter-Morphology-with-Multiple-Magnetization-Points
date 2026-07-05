"""Replay cached n2 mode-1 gates on the active JAX device."""

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
P2C191_DIR = ROOT / "experiment_records_ik_optimize/20260705093000_p2c191_n2_batch_system_cache_steady_replay"
for path in (ROOT, P2C191_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c191_n2_batch_system_cache_steady_replay as p2c191  # noqa: E402


def jsonable(value: Any) -> Any:
    """Convert nested values to JSON-compatible objects."""
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


def plot(summary: dict[str, Any]) -> None:
    """Plot GPU replay timing and quality."""
    configure_matplotlib_cjk()
    labels = [row["workpoint"] for row in summary["steady_work_rows"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axes[0].bar(x, [row["gate_wall_s"] for row in summary["warm_work_rows"]], color="#e9c46a", label="预热")
    axes[0].bar(x, [row["gate_wall_s"] for row in summary["steady_work_rows"]], color="#2a9d8f", alpha=0.75, label="稳态")
    axes[0].set_yscale("log")
    axes[0].set_title("GPU 回放耗时")
    axes[0].legend()
    axes[1].bar(x, [row["pass_count"] / row["tested_count"] for row in summary["steady_work_rows"]], color="#457b9d")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("稳态门禁通过比例")
    axes[2].bar(x, [row["max_task_error"] for row in summary["steady_work_rows"]], color="#f4a261")
    axes[2].axhline(p2c191.p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("最大任务线性化误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c195 两点磁控 GPU 缓存稳态回放")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c195_n2_gpu_cached_steady_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 预热/s | 稳态/s | 通过数 | 最大任务误差 | 最大输出误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    warm_by_name = {row["workpoint"]: row for row in summary["warm_work_rows"]}
    for row in summary["steady_work_rows"]:
        warm = warm_by_name[row["workpoint"]]
        table.append(
            f"| {row['workpoint']} | {warm['gate_wall_s']:.2f} | {row['gate_wall_s']:.2f} | "
            f"{row['pass_count']}/{row['tested_count']} | {row['max_task_error']:.3e} | {row['max_output_error']:.3e} |"
        )
    text = f"""# P2c195 两点磁控 GPU 缓存稳态回放

## 实验目的

此前 P2c191 已在 CPU 上完成五工作点缓存稳态回放，但 GPU 后端不可用。本轮在 JAX 已识别 `cuda:0` 后，复用 P2c191 的可信分支缓存和局部线性系统缓存，重新运行五工作点预热与稳态严格门禁，补齐 GPU 效率证据。

## 实验方案

不重建线性系统，不改变门禁阈值，不改变扰动半径。每个工作点仍执行正负扰动、四初值、回环求解和所有硬门禁。第一轮作为预热，第二轮作为稳态。

## 结果

{chr(10).join(table)}

## 汇总指标

- 预热通过数：{summary['warm_pass_count']} / {summary['warm_tested_count']}。
- 稳态通过数：{summary['steady_pass_count']} / {summary['steady_tested_count']}。
- 预热墙钟耗时：{summary['warm_wall_s']:.2f} s。
- 稳态墙钟耗时：{summary['steady_wall_s']:.2f} s。
- 稳态最小二乘求解总耗时：{summary['steady_solve_total_s']:.4f} s。
- JAX 设备：{', '.join(summary['jax_devices'])}。

## 实验结论

{summary['decision']}

## 下一步计划

若后续要继续提速，应优先减少门禁内部的接口连续性计算和回环诊断；若继续推进三维任务，应先修复第 2/第 3 主方向的物理闭合和局部参数化。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c195_n2_gpu_cached_steady_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run GPU cached steady replay."""
    t_wall = time.perf_counter()
    write_commands()
    names = p2c191.workpoint_names()
    runtime_entries, runtime_load_s = p2c191.load_runtime_entries(names)
    warm_work_rows, warm_rows, warm_wall_s = p2c191.replay_cached_systems(runtime_entries, "warm")
    steady_work_rows, steady_rows, steady_wall_s = p2c191.replay_cached_systems(runtime_entries, "steady")
    warm_pass = sum(1 for row in warm_rows if row["gate_pass"])
    steady_pass = sum(1 for row in steady_rows if row["gate_pass"])
    status = "gpu_cached_steady_passed" if warm_pass == len(warm_rows) and steady_pass == len(steady_rows) else "gpu_cached_steady_failed"
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": (
            "GPU 缓存稳态回放保持全部严格门禁通过；可补充 CPU 以外的效率证据。"
            if status == "gpu_cached_steady_passed"
            else "GPU 缓存回放未保持全部门禁通过；不能替代 CPU 证据。"
        ),
        "workpoints": names,
        "runtime_load_s": float(runtime_load_s),
        "warm_wall_s": float(warm_wall_s),
        "steady_wall_s": float(steady_wall_s),
        "wall_s": float(time.perf_counter() - t_wall),
        "warm_tested_count": len(warm_rows),
        "warm_pass_count": warm_pass,
        "steady_tested_count": len(steady_rows),
        "steady_pass_count": steady_pass,
        "steady_solve_total_s": float(sum(row["solve_total_s"] for row in steady_work_rows)),
        "warm_work_rows": warm_work_rows,
        "steady_work_rows": steady_work_rows,
        "rows": steady_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    write_csv(EXP_DIR / "warm_replay_rows.csv", warm_rows)
    write_csv(EXP_DIR / "steady_replay_rows.csv", steady_rows)
    write_csv(EXP_DIR / "steady_work_rows.csv", steady_work_rows)
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in warm_rows + steady_rows:
            if not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "GPU 缓存回放门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(summary)
    write_record(summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
