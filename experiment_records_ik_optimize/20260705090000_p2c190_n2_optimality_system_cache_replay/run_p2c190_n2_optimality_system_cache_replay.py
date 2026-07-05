"""Replay the strict n2 gate with a cached optimality system."""

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
for path in (ROOT, P2C174_DIR, P2C186_DIR, P2C189_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402
import run_p2c186_n2_independent_cache_replay as p2c186  # noqa: E402
import run_p2c189_instrumented_solve_timing_single_workpoint as p2c189  # noqa: E402


WORKPOINT = "rand_05"
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


def save_system(system: dict[str, Any]) -> Path:
    """Save the cacheable linear optimality system."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{WORKPOINT}_mode1_optimality_system.npz"
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
    return path


def load_system(path: Path) -> tuple[dict[str, Any], float]:
    """Load a cached linear optimality system."""
    t0 = time.perf_counter()
    data = np.load(path)
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


def system_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Return max absolute differences after cache round-trip."""
    keys = ("g_q", "g_I", "u", "s", "vt", "t_eff")
    diffs = {key: float(np.max(np.abs(np.asarray(a[key], dtype=float) - np.asarray(b[key], dtype=float)))) for key in keys}
    diffs["raw_condition_abs_diff"] = float(abs(float(a["raw_condition"]) - float(b["raw_condition"])))
    return diffs


def replay_gate(branch: dict[str, Any], system: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Replay the strict branch-loop gate with an injected system."""
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


def timing_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate row timing components."""
    return p2c189.timing_totals(rows)


def summarize(
    *,
    rows: list[dict[str, Any]],
    direction_summaries: list[dict[str, Any]],
    load_rebuild_s: float,
    system_build_s: float,
    system_load_s: float,
    gate_wall_s: float,
    wall_s: float,
    diffs: dict[str, float],
) -> dict[str, Any]:
    """Summarize cached-system replay."""
    totals = timing_totals(rows)
    pass_count = sum(1 for row in rows if row["gate_pass"])
    steady_replay_s = load_rebuild_s + system_load_s + gate_wall_s
    baseline_full_s = load_rebuild_s + system_build_s + gate_wall_s
    return {
        "experiment_id": EXP_DIR.name,
        "status": "system_cache_replay_passed" if pass_count == len(rows) else "system_cache_replay_failed",
        "decision": (
            "缓存线性系统后严格门禁保持通过；系统构建可从重复审计路径中移除。"
            if pass_count == len(rows)
            else "缓存线性系统回放未保持全部门禁通过；不能用于提速。"
        ),
        "workpoint": WORKPOINT,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "tested_count": len(rows),
        "pass_count": pass_count,
        "load_rebuild_s": float(load_rebuild_s),
        "system_build_s": float(system_build_s),
        "system_load_s": float(system_load_s),
        "gate_wall_s": float(gate_wall_s),
        "wall_s": float(wall_s),
        "baseline_full_s": float(baseline_full_s),
        "cached_steady_replay_s": float(steady_replay_s),
        "estimated_speedup_vs_rebuild": float(baseline_full_s / steady_replay_s) if steady_replay_s > 0 else None,
        "timing_totals": totals,
        "system_cache_roundtrip_max_abs_diff": diffs,
        "direction_summaries": direction_summaries,
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }


def plot(summary: dict[str, Any]) -> None:
    """Plot runtime comparison and gate quality."""
    configure_matplotlib_cjk()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = ["重建线性系统", "缓存回放"]
    values = [summary["baseline_full_s"], summary["cached_steady_replay_s"]]
    axes[0].bar(labels, values, color=["#e76f51", "#2a9d8f"])
    axes[0].set_ylabel("耗时/s")
    axes[0].set_title("单工作点审计耗时对比")
    axes[0].grid(axis="y", alpha=0.25)
    row_labels = [f"{row['direction']}-{row['start']}" for row in summary["rows"]]
    x = np.arange(len(row_labels))
    axes[1].bar(x, [row["task_pred_vs_actual_rel_err"] for row in summary["rows"]], color="#457b9d")
    axes[1].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, row_labels, rotation=30, ha="right")
    axes[1].set_ylabel("任务线性化误差")
    axes[1].set_title("缓存回放门禁质量")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c190 线性系统缓存回放")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c190_optimality_system_cache_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    rows_table = [
        "| 方向 | 初值 | 任务误差 | 输出误差 | 回环误差 | 门禁 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        rows_table.append(
            f"| {row['direction']} | {row['start']} | {row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['output_linear_at_actual_q_rel_err']:.3e} | {row['return_q_norm']:.3e} | "
            f"{'通过' if row['gate_pass'] else '不可发布'} |"
        )
    text = f"""# P2c190 两点磁控线性系统缓存回放验证

## 实验目的

P2c189 显示单工作点审计中，局部线性系统构建耗时约 `82.48 s`，显著高于严格门禁回放本身。本轮验证该线性系统是否可以作为数组缓存复用，从而降低重复审计耗时。

## 实验方案

使用 `rand_05` 工作点。先按原流程构建局部线性系统，保存 `g_q`、`g_I`、奇异值分解结果、任务输出线性矩阵和条件数。随后重新从缓存加载该线性系统，并执行与 P2c174/P2c189 相同的正负扰动、四初值和回环门禁。

## 结果

{chr(10).join(rows_table)}

## 耗时

- 分支加载与函数重建：{summary['load_rebuild_s']:.2f} s。
- 原始线性系统构建：{summary['system_build_s']:.2f} s。
- 线性系统缓存加载：{summary['system_load_s']:.4f} s。
- 缓存后严格门禁：{summary['gate_wall_s']:.2f} s。
- 重建路径估计耗时：{summary['baseline_full_s']:.2f} s。
- 缓存回放稳态耗时：{summary['cached_steady_replay_s']:.2f} s。
- 相对重建路径提速：{summary['estimated_speedup_vs_rebuild']:.2f} 倍。
- 运行设备：{', '.join(summary['jax_devices'])}。

## 数据分析

缓存往返最大数组差异为 `{max(summary['system_cache_roundtrip_max_abs_diff'].values()):.3e}`。通过样本数为 {summary['pass_count']} / {summary['tested_count']}。若缓存回放全部通过，说明局部线性系统可作为可信分支库的一部分保存；重复审计时不必再次计算高阶雅可比相关量。

## 实验结论

{summary['decision']}

## 下一步计划

下一轮应将该系统缓存推广到五个工作点，并评估批量回放是否仍保持 `40/40` 门禁通过。随后再处理门禁内部的接口连续性计算耗时。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c190_n2_optimality_system_cache_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run cached-system replay validation."""
    t_wall = time.perf_counter()
    write_commands()
    branch, load_rebuild_s = p2c186.load_branch(WORKPOINT)
    t_system = time.perf_counter()
    system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch)
    system_build_s = time.perf_counter() - t_system
    cache_path = save_system(system)
    cached_system, system_load_s = load_system(cache_path)
    diffs = system_diff(system, cached_system)
    direction_summaries, rows, gate_wall_s = replay_gate(branch, cached_system)
    summary = summarize(
        rows=rows,
        direction_summaries=direction_summaries,
        load_rebuild_s=load_rebuild_s,
        system_build_s=system_build_s,
        system_load_s=system_load_s,
        gate_wall_s=gate_wall_s,
        wall_s=time.perf_counter() - t_wall,
        diffs=diffs,
    )
    write_csv(EXP_DIR / "system_cache_replay_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "线性系统缓存回放门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
