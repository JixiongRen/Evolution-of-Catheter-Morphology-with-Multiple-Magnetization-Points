"""Replay all n2 mode-1 workpoints with cached optimality systems."""

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
CACHE_DIR = EXP_DIR / "cache"
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
P2C186_DIR = ROOT / "experiment_records_ik_optimize/20260705043000_p2c186_n2_independent_cache_replay"
P2C189_DIR = ROOT / "experiment_records_ik_optimize/20260705080000_p2c189_instrumented_solve_timing_single_workpoint"
P2C190_DIR = ROOT / "experiment_records_ik_optimize/20260705090000_p2c190_n2_optimality_system_cache_replay"
for path in (ROOT, P2C174_DIR, P2C186_DIR, P2C189_DIR, P2C190_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402
import run_p2c186_n2_independent_cache_replay as p2c186  # noqa: E402
import run_p2c189_instrumented_solve_timing_single_workpoint as p2c189  # noqa: E402
import run_p2c190_n2_optimality_system_cache_replay as p2c190  # noqa: E402


MODE_INDEX = 0
RADIUS_A = 1.0e-4


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


def workpoint_names() -> list[str]:
    """Return cached P2c186 workpoint names."""
    return [
        path.name.split("_mode1_branch_cache", 1)[0]
        for path in sorted(p2c186.CACHE_DIR.glob("*_mode1_branch_cache.npz"))
    ]


def system_path(name: str) -> Path:
    """Return the optimality system cache path."""
    return CACHE_DIR / f"{name}_mode1_optimality_system.npz"


def save_system(name: str, system: dict[str, Any]) -> None:
    """Save one cacheable optimality system."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = system_path(name)
    np.savez_compressed(
        path,
        g_q=np.asarray(system["g_q"], dtype=float),
        g_I=np.asarray(system["g_I"], dtype=float),
        u=np.asarray(system["u"], dtype=float),
        s=np.asarray(system["s"], dtype=float),
        vt=np.asarray(system["vt"], dtype=float),
        t_eff=np.asarray(system["t_eff"], dtype=float),
        raw_condition=np.asarray([float(system["raw_condition"])], dtype=float),
    )


def load_system(name: str) -> tuple[dict[str, Any], float]:
    """Load one cached optimality system."""
    t0 = time.perf_counter()
    data = np.load(system_path(name))
    system = {
        "g_q": np.asarray(data["g_q"], dtype=float),
        "g_I": np.asarray(data["g_I"], dtype=float),
        "u": np.asarray(data["u"], dtype=float),
        "s": np.asarray(data["s"], dtype=float),
        "vt": np.asarray(data["vt"], dtype=float),
        "t_eff": np.asarray(data["t_eff"], dtype=float),
        "raw_condition": float(np.asarray(data["raw_condition"], dtype=float).reshape(-1)[0]),
    }
    return system, time.perf_counter() - t0


def replay_gate(branch: dict[str, Any], system: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Replay the strict gate for one branch and cached system."""
    old_radius = p2c174.RADIUS_A
    old_mode = p2c174.MODE_INDEX
    p2c174.RADIUS_A = RADIUS_A
    p2c174.MODE_INDEX = MODE_INDEX
    try:
        t0 = time.perf_counter()
        summaries: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            direction_summary, direction_rows = p2c189.line_metrics_instrumented(branch, system, sign, label)
            summaries.append(direction_summary)
            rows.extend(direction_rows)
        return summaries, rows, time.perf_counter() - t0
    finally:
        p2c174.RADIUS_A = old_radius
        p2c174.MODE_INDEX = old_mode


def build_system_cache(names: list[str]) -> list[dict[str, Any]]:
    """Build all optimality system caches."""
    rows: list[dict[str, Any]] = []
    for name in names:
        t_work = time.perf_counter()
        branch, load_rebuild_s = p2c186.load_branch(name)
        t_system = time.perf_counter()
        system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch)
        system_build_s = time.perf_counter() - t_system
        save_system(name, system)
        rows.append(
            {
                "workpoint": name,
                "load_rebuild_s": float(load_rebuild_s),
                "system_build_s": float(system_build_s),
                "total_s": float(time.perf_counter() - t_work),
                "raw_condition": float(system["raw_condition"]),
            }
        )
    return rows


def load_runtime_entries(names: list[str]) -> tuple[list[dict[str, Any]], float]:
    """Load branches and cached systems once for warm and steady replay."""
    t_load = time.perf_counter()
    entries: list[dict[str, Any]] = []
    for name in names:
        branch, load_rebuild_s = p2c186.load_branch(name)
        system, system_load_s = load_system(name)
        entries.append(
            {
                "name": name,
                "branch": branch,
                "system": system,
                "load_rebuild_s": float(load_rebuild_s),
                "system_load_s": float(system_load_s),
            }
        )
    return entries, time.perf_counter() - t_load


def replay_cached_systems(entries: list[dict[str, Any]], phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Replay all cached systems once using already loaded runtime objects."""
    t_phase = time.perf_counter()
    work_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        name = entry["name"]
        branch = entry["branch"]
        system = entry["system"]
        direction_summaries, gate_rows, gate_wall_s = replay_gate(branch, system)
        for row in gate_rows:
            row["workpoint"] = name
            row["phase"] = phase
        rows.extend(gate_rows)
        work_rows.append(
            {
                "phase": phase,
                "workpoint": name,
                "tested_count": len(gate_rows),
                "pass_count": sum(1 for row in gate_rows if row["gate_pass"]),
                "load_rebuild_s": float(entry["load_rebuild_s"]),
                "system_load_s": float(entry["system_load_s"]),
                "gate_wall_s": float(gate_wall_s),
                "total_s": float(gate_wall_s),
                "solve_total_s": float(sum(row["solve_total_s"] + row["return_total_s"] for row in gate_rows)),
                "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in gate_rows),
                "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in gate_rows),
                "direction_summaries": direction_summaries,
            }
        )
    return work_rows, rows, time.perf_counter() - t_phase


def summarize(
    names: list[str],
    build_rows: list[dict[str, Any]],
    runtime_load_s: float,
    warm_work_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
    warm_wall_s: float,
    steady_work_rows: list[dict[str, Any]],
    steady_rows: list[dict[str, Any]],
    steady_wall_s: float,
    wall_s: float,
) -> dict[str, Any]:
    """Summarize batch cache replay."""
    warm_pass = sum(1 for row in warm_rows if row["gate_pass"])
    steady_pass = sum(1 for row in steady_rows if row["gate_pass"])
    total_build_s = sum(row["system_build_s"] for row in build_rows)
    return {
        "experiment_id": EXP_DIR.name,
        "status": "batch_system_cache_steady_passed" if warm_pass == len(warm_rows) and steady_pass == len(steady_rows) else "batch_system_cache_steady_failed",
        "decision": (
            "五工作点线性系统缓存回放保持全部严格门禁通过；可作为两点磁控一维有效子空间审计的工程化路径。"
            if warm_pass == len(warm_rows) and steady_pass == len(steady_rows)
            else "批量线性系统缓存回放未保持全部门禁通过；不能作为工程化路径。"
        ),
        "workpoints": names,
        "workpoint_count": len(names),
        "warm_tested_count": len(warm_rows),
        "warm_pass_count": warm_pass,
        "steady_tested_count": len(steady_rows),
        "steady_pass_count": steady_pass,
        "system_build_total_s": float(total_build_s),
        "runtime_load_s": float(runtime_load_s),
        "warm_wall_s": float(warm_wall_s),
        "steady_wall_s": float(steady_wall_s),
        "wall_s": float(wall_s),
        "steady_solve_total_s": float(sum(row["solve_total_s"] for row in steady_work_rows)),
        "steady_gate_total_s": float(sum(row["gate_wall_s"] for row in steady_work_rows)),
        "steady_cached_total_s": float(sum(row["total_s"] for row in steady_work_rows)),
        "estimated_speedup_vs_rebuild_each_time": float((total_build_s + sum(row["total_s"] for row in steady_work_rows)) / max(sum(row["total_s"] for row in steady_work_rows), 1.0e-12)),
        "build_rows": build_rows,
        "warm_work_rows": warm_work_rows,
        "steady_work_rows": steady_work_rows,
        "rows": steady_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }


def plot(summary: dict[str, Any]) -> None:
    """Plot batch cache replay runtime and gate quality."""
    configure_matplotlib_cjk()
    labels = [row["workpoint"] for row in summary["steady_work_rows"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axes[0].bar(x, [row["system_build_s"] for row in summary["build_rows"]], color="#e76f51", label="线性系统构建")
    axes[0].bar(x, [row["total_s"] for row in summary["steady_work_rows"]], color="#2a9d8f", label="缓存稳态回放")
    axes[0].set_yscale("log")
    axes[0].set_title("耗时对比")
    axes[0].legend()
    axes[1].bar(x, [row["pass_count"] / row["tested_count"] for row in summary["steady_work_rows"]], color="#457b9d")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("稳态门禁通过比例")
    axes[2].bar(x, [row["max_task_error"] for row in summary["steady_work_rows"]], color="#e9c46a")
    axes[2].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("最大任务线性化误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c191 五工作点线性系统缓存稳态回放")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c191_batch_system_cache_steady_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 工作点 | 构建线性系统/s | 稳态缓存回放/s | 通过数 | 最大任务误差 | 最大输出误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    build_by_name = {row["workpoint"]: row for row in summary["build_rows"]}
    for row in summary["steady_work_rows"]:
        build = build_by_name[row["workpoint"]]
        table.append(
            f"| {row['workpoint']} | {build['system_build_s']:.2f} | {row['total_s']:.2f} | "
            f"{row['pass_count']}/{row['tested_count']} | {row['max_task_error']:.3e} | {row['max_output_error']:.3e} |"
        )
    text = f"""# P2c191 两点磁控五工作点线性系统缓存稳态回放

## 实验目的

P2c190 已证明单工作点线性系统可缓存。本轮将该策略推广到五个工作点，验证批量审计中是否仍保持严格门禁，并记录预热后稳态耗时。

## 实验方案

复用 P2c186 的五个可信分支缓存。对每个工作点先构建并保存局部线性系统，随后加载缓存系统执行两轮严格门禁。第一轮作为预热回放，第二轮作为稳态回放。门禁仍采用正负扰动、四初值、回环求解和 P2c174 硬阈值。

## 结果

{chr(10).join(table)}

## 汇总指标

- 工作点数量：{summary['workpoint_count']}。
- 预热回放通过数：{summary['warm_pass_count']} / {summary['warm_tested_count']}。
- 稳态回放通过数：{summary['steady_pass_count']} / {summary['steady_tested_count']}。
- 线性系统构建总耗时：{summary['system_build_total_s']:.2f} s。
- 运行时分支与系统加载耗时：{summary['runtime_load_s']:.2f} s。
- 预热回放墙钟耗时：{summary['warm_wall_s']:.2f} s。
- 稳态回放墙钟耗时：{summary['steady_wall_s']:.2f} s。
- 稳态缓存回放总耗时：{summary['steady_cached_total_s']:.2f} s。
- 相对“每次重建线性系统”估计提速：{summary['estimated_speedup_vs_rebuild_each_time']:.2f} 倍。
- 运行设备：{', '.join(summary['jax_devices'])}。

## 数据分析

若稳态回放保持 `40/40` 通过，说明线性系统缓存不会破坏分支条件下的一维局部灵敏度结论。该结果仍不是完整三维末端位置灵敏度，只能作为两点磁控第 1 主方向的有效子空间结果。

## 实验结论

{summary['decision']}

## 下一步计划

下一步应继续减少稳态回放内部的接口连续性计算和回环计算开销；同时另行推进完整三维末端位置灵敏度失败边界归因。GPU 后端仍需修复，否则所有耗时结论只能视为 CPU 保守估计。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c191_n2_batch_system_cache_steady_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run batch cached-system replay."""
    t_wall = time.perf_counter()
    write_commands()
    names = workpoint_names()
    build_rows = build_system_cache(names)
    runtime_entries, runtime_load_s = load_runtime_entries(names)
    warm_work_rows, warm_rows, warm_wall_s = replay_cached_systems(runtime_entries, "warm")
    steady_work_rows, steady_rows, steady_wall_s = replay_cached_systems(runtime_entries, "steady")
    summary = summarize(
        names,
        build_rows,
        runtime_load_s,
        warm_work_rows,
        warm_rows,
        warm_wall_s,
        steady_work_rows,
        steady_rows,
        steady_wall_s,
        time.perf_counter() - t_wall,
    )
    write_csv(EXP_DIR / "build_rows.csv", build_rows)
    write_csv(EXP_DIR / "warm_replay_rows.csv", warm_rows)
    write_csv(EXP_DIR / "steady_replay_rows.csv", steady_rows)
    write_csv(EXP_DIR / "steady_work_rows.csv", steady_work_rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in warm_rows + steady_rows:
            if not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "批量系统缓存回放门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
