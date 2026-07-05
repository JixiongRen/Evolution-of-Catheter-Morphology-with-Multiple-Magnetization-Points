"""Build and replay a single-workpoint n2 branch cache."""

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
P2C148_DIR = ROOT / "experiment_records_ik_optimize/20260630103000_p2c148_n2_all_principal_safe_radius"
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
for path in (ROOT, P2C148_DIR, P2C174_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c148_n2_all_principal_safe_radius as p2c148  # noqa: E402
import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402


WORKPOINT = "rand_05"
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


def build_source_branch() -> tuple[dict[str, Any], dict[str, Any], float]:
    """Build the source branch and return the post-step local branch."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    directions = dict(p2c148.p2c145.p2c144.candidate_directions())
    branch0 = p2c148.build_candidate_branch(base, WORKPOINT, directions[WORKPOINT])
    first_row, branch1 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(
        branch0,
        step_index=1,
        mode_index=MODE_INDEX,
    )
    return first_row, branch1, time.perf_counter() - t0


def save_cache(branch: dict[str, Any], first_row: dict[str, Any]) -> float:
    """Save cacheable branch arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    np.savez_compressed(
        CACHE_DIR / "rand_05_mode1_branch_cache.npz",
        I0=np.asarray(branch["I0"], dtype=float),
        y=np.asarray(branch["y"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        S=np.asarray(branch["S"], dtype=float),
        U=np.asarray(branch["U"], dtype=float),
        Vt=np.asarray(branch["Vt"], dtype=float),
        singular_values=np.asarray(branch["singular_values"], dtype=float),
        base_tip=np.asarray(branch["base_out"]["tip_p"], dtype=float),
        base_centerline=np.asarray(branch["base_out"]["centerline"], dtype=float),
    )
    meta = {
        "workpoint": WORKPOINT,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "first_step_release_class": first_row.get("release_class"),
        "first_step_linear_rel_err": first_row.get("linear_rel_err"),
        "source": "P2c184 runtime-built branch; funcs are intentionally not serialized.",
    }
    (CACHE_DIR / "rand_05_mode1_branch_cache_metadata.json").write_text(
        json.dumps(jsonable(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return time.perf_counter() - t0


def load_cached_branch(branch: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Load cached arrays and attach runtime functions."""
    t0 = time.perf_counter()
    data = np.load(CACHE_DIR / "rand_05_mode1_branch_cache.npz")
    cached = dict(branch)
    cached["I0"] = np.asarray(data["I0"], dtype=float)
    cached["y"] = np.asarray(data["y"], dtype=float)
    cached["basis"] = np.asarray(data["basis"], dtype=float)
    cached["S"] = np.asarray(data["S"], dtype=float)
    cached["U"] = np.asarray(data["U"], dtype=float)
    cached["Vt"] = np.asarray(data["Vt"], dtype=float)
    cached["singular_values"] = np.asarray(data["singular_values"], dtype=float)
    cached["base_out"] = {
        "tip_p": np.asarray(data["base_tip"], dtype=float),
        "centerline": np.asarray(data["base_centerline"], dtype=float),
    }
    return cached, time.perf_counter() - t0


def replay_gate(branch: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Replay the strict branch-loop gate from cached arrays."""
    old_radius = p2c174.RADIUS_A
    old_mode = p2c174.MODE_INDEX
    p2c174.RADIUS_A = RADIUS_A
    p2c174.MODE_INDEX = MODE_INDEX
    try:
        system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch)
        t0 = time.perf_counter()
        summaries: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            direction_summary, direction_rows = p2c174.line_metrics(branch, system, sign, label)
            summaries.append(direction_summary)
            rows.extend(direction_rows)
        return summaries, rows, time.perf_counter() - t0
    finally:
        p2c174.RADIUS_A = old_radius
        p2c174.MODE_INDEX = old_mode


def plot(summary: dict[str, Any]) -> None:
    """Plot runtime attribution."""
    configure_matplotlib_cjk()
    labels = ["分支构建", "缓存保存", "缓存加载", "门禁回放"]
    values = [
        summary["build_s"],
        summary["save_s"],
        summary["load_s"],
        summary["replay_gate_s"],
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values, color=["#e76f51", "#f4a261", "#2a9d8f", "#457b9d"])
    ax.set_yscale("log")
    ax.set_ylabel("耗时/s")
    ax.set_title("n2单工作点缓存回放耗时")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_single_workpoint_cache_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    rows = summary["rows"]
    table = [
        "| 方向 | 初值 | 任务误差 | 输出误差 | 回环误差 | 多初值差异 | 结论 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['start']} | {row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['output_linear_at_actual_q_rel_err']:.3e} | {row['return_q_norm']:.3e} | "
            f"{row['q_consistency_to_zero_start']:.3e} | {'通过' if row['gate_pass'] else '不可发布'} |"
        )
    text = f"""# P2c184 两点磁控单工作点分支缓存回放原型

## 实验目的

P2c183 定义了可信分支缓存包。本轮选择 `rand_05` 工作点执行真实单点缓存原型：先构建可信分支并保存数组，再从缓存数组加载并回放 P2c174 的正负扰动、多初值和回环门禁。

## 实验方案

缓存内容包括基线电流、状态向量、约化基、局部灵敏度矩阵、左右奇异向量、奇异值、基线末端点和基线中心线。JAX 函数不缓存，仍在运行时由分支构建过程提供。本轮目的不是宣称最终产品级提速，而是验证数组缓存不会破坏门禁结果。

## 结果

{chr(10).join(table)}

## 耗时

| 阶段 | 耗时/s |
|---|---:|
| 分支构建 | {summary['build_s']:.2f} |
| 缓存保存 | {summary['save_s']:.4f} |
| 缓存加载 | {summary['load_s']:.4f} |
| 门禁回放 | {summary['replay_gate_s']:.4f} |
| 重求解净耗时 | {summary['total_solve_steady_s']:.4f} |

## 实验结论

{summary['decision']}

## 下一步计划

若本轮通过，应扩展到 5 个工作点并把函数重建耗时单独计量。若 5 个工作点回放仍保持 40/40 通过，再将缓存构建与缓存回放拆成两个命令，比较冷启动、缓存加载和稳定门禁时间。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the single-workpoint cache replay."""
    t0 = time.perf_counter()
    first_row, source_branch, build_s = build_source_branch()
    save_s = save_cache(source_branch, first_row)
    cached_branch, load_s = load_cached_branch(source_branch)
    direction_summaries, rows, replay_gate_s = replay_gate(cached_branch)
    pass_count = sum(1 for row in rows if row["gate_pass"])
    all_pass = pass_count == len(rows)
    if all_pass:
        status = "single_workpoint_cache_replay_passed"
        decision = "缓存数组回放保持 rand_05 工作点 8/8 严格门禁通过；可扩展到跨工作点缓存实验。"
    else:
        status = "single_workpoint_cache_replay_failed"
        decision = "缓存数组回放未保持严格门禁；需检查缓存字段或函数重建约定。"
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "workpoint": WORKPOINT,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "tested_count": len(rows),
        "pass_count": pass_count,
        "publishable_boundary_changed": False,
        "build_s": float(build_s),
        "save_s": float(save_s),
        "load_s": float(load_s),
        "replay_gate_s": float(replay_gate_s),
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "direction_summaries": direction_summaries,
        "rows": rows,
    }
    write_csv(EXP_DIR / "cache_replay_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                event = {"failure_class": "缓存回放门禁失败", "evidence": row}
                f.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c184_n2_single_workpoint_cache_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
