"""Build and replay n2 branch caches across candidate workpoints."""

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


def build_source_branch(base: dict[str, Any], name: str, direction: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Build the source branch and return the post-step local branch."""
    t0 = time.perf_counter()
    branch0 = p2c148.build_candidate_branch(base, name, direction)
    first_row, branch1 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(
        branch0,
        step_index=1,
        mode_index=MODE_INDEX,
    )
    return first_row, branch1, time.perf_counter() - t0


def cache_paths(name: str) -> tuple[Path, Path]:
    """Return the array and metadata cache paths."""
    stem = f"{name}_mode1_branch_cache"
    return CACHE_DIR / f"{stem}.npz", CACHE_DIR / f"{stem}_metadata.json"


def save_cache(name: str, branch: dict[str, Any], first_row: dict[str, Any]) -> float:
    """Save cacheable branch arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array_path, meta_path = cache_paths(name)
    t0 = time.perf_counter()
    np.savez_compressed(
        array_path,
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
        "workpoint": name,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "first_step_release_class": first_row.get("release_class"),
        "first_step_linear_rel_err": first_row.get("linear_rel_err"),
        "source": "P2c185 runtime-built branch; funcs are intentionally not serialized.",
    }
    meta_path.write_text(json.dumps(jsonable(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return time.perf_counter() - t0


def load_cached_branch(name: str, branch: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Load cached arrays and attach runtime functions from the source branch."""
    array_path, _meta_path = cache_paths(name)
    t0 = time.perf_counter()
    data = np.load(array_path)
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


def run_workpoint(base: dict[str, Any], name: str, direction: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build, cache, reload, and replay one workpoint."""
    first_row, source_branch, build_s = build_source_branch(base, name, direction)
    save_s = save_cache(name, source_branch, first_row)
    cached_branch, load_s = load_cached_branch(name, source_branch)
    direction_summaries, rows, replay_gate_s = replay_gate(cached_branch)
    for row in rows:
        row["workpoint"] = name
    pass_count = sum(1 for row in rows if row["gate_pass"])
    work_summary = {
        "workpoint": name,
        "tested_count": len(rows),
        "pass_count": pass_count,
        "all_pass": pass_count == len(rows),
        "build_s": float(build_s),
        "save_s": float(save_s),
        "load_s": float(load_s),
        "replay_gate_s": float(replay_gate_s),
        "solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in rows),
        "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in rows),
        "max_return_q_norm": max(row["return_q_norm"] for row in rows),
        "direction_summaries": direction_summaries,
    }
    return work_summary, rows


def plot(summary: dict[str, Any]) -> None:
    """Plot runtime and gate summaries."""
    configure_matplotlib_cjk()
    labels = [item["workpoint"] for item in summary["workpoint_summaries"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].bar(x, [item["build_s"] for item in summary["workpoint_summaries"]], color="#e76f51", label="构建")
    axes[0].bar(x, [item["replay_gate_s"] for item in summary["workpoint_summaries"]], color="#457b9d", label="回放")
    axes[0].set_yscale("log")
    axes[0].set_title("构建与回放耗时")
    axes[0].legend()
    axes[1].bar(x, [item["pass_count"] / item["tested_count"] for item in summary["workpoint_summaries"]], color="#2a9d8f")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("严格门禁通过比例")
    axes[2].bar(x, [item["max_task_error"] for item in summary["workpoint_summaries"]], color="#f4a261")
    axes[2].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("最大任务线性化误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2跨工作点缓存回放结果")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_cross_workpoint_cache_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 通过数 | 构建/s | 保存/s | 加载/s | 回放/s | 最大任务误差 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["workpoint_summaries"]:
        table.append(
            f"| {item['workpoint']} | {item['pass_count']}/{item['tested_count']} | "
            f"{item['build_s']:.2f} | {item['save_s']:.4f} | {item['load_s']:.4f} | "
            f"{item['replay_gate_s']:.2f} | {item['max_task_error']:.3e} | "
            f"{'通过' if item['all_pass'] else '不可发布'} |"
        )
    text = f"""# P2c185 两点磁控跨工作点分支缓存回放

## 实验目的

P2c184 已证明单工作点数组缓存不会破坏严格门禁。本轮扩展到 P2c179 的 5 个工作点，验证缓存数组回放是否仍保持 `40/40` 门禁通过，并拆分分支构建、缓存保存、缓存加载和门禁回放耗时。

## 实验方案

每个工作点先构建可信分支并保存数组缓存。缓存内容包括基线电流、状态向量、约化基、局部灵敏度矩阵、左右奇异向量、奇异值、基线末端点和基线中心线。随后从缓存数组加载分支，并回放正负扰动、多初值和回环门禁。JAX 函数仍不序列化，运行时从源分支继承。

## 结果

{chr(10).join(table)}

## 总结指标

- 总通过数：{summary['pass_count']} / {summary['tested_count']}。
- 总墙钟耗时：{summary['wall_s']:.2f} s。
- 总分支构建耗时：{summary['total_build_s']:.2f} s。
- 总缓存加载耗时：{summary['total_load_s']:.4f} s。
- 总门禁回放耗时：{summary['total_replay_gate_s']:.2f} s。
- 总重求解净耗时：{summary['total_solve_steady_s']:.4f} s。
- 运行设备：{', '.join(summary['jax_devices'])}。

## 实验结论

{summary['decision']}

## 下一步计划

若本轮通过，应把缓存构建和缓存回放拆为两个独立命令，并在第二个命令中只从缓存读取数组、重建函数、执行门禁。若仍保持 40/40 通过，再进一步尝试批量多初值求解，减少 Python 层循环。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run cross-workpoint cache replay."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    work_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        work_summary, work_rows = run_workpoint(base, name, direction)
        work_summaries.append(work_summary)
        rows.extend(work_rows)
        write_csv(EXP_DIR / "cache_replay_rows.partial.csv", rows)
    pass_count = sum(1 for row in rows if row["gate_pass"])
    all_pass = pass_count == len(rows)
    if all_pass:
        status = "cross_workpoint_cache_replay_passed"
        decision = "5 个工作点缓存数组回放保持 40/40 严格门禁通过；缓存方案可进入独立构建/独立回放阶段。"
    else:
        status = "cross_workpoint_cache_replay_failed"
        decision = "缓存数组回放未保持全部门禁通过；需检查缓存字段或函数重建约定。"
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "workpoint_count": len(work_summaries),
        "tested_count": len(rows),
        "pass_count": pass_count,
        "publishable_boundary_changed": False,
        "wall_s": float(time.perf_counter() - t0),
        "total_build_s": float(sum(item["build_s"] for item in work_summaries)),
        "total_save_s": float(sum(item["save_s"] for item in work_summaries)),
        "total_load_s": float(sum(item["load_s"] for item in work_summaries)),
        "total_replay_gate_s": float(sum(item["replay_gate_s"] for item in work_summaries)),
        "total_solve_steady_s": float(sum(item["solve_steady_s"] for item in work_summaries)),
        "jax_devices": [str(device) for device in jax.devices()],
        "workpoint_summaries": work_summaries,
        "rows": rows,
    }
    write_csv(EXP_DIR / "cache_replay_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                event = {"failure_class": "跨工作点缓存回放门禁失败", "evidence": row}
                f.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c185_n2_cross_workpoint_cache_replay.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "pass_count": pass_count,
                "tested_count": len(rows),
                "wall_s": summary["wall_s"],
                "total_build_s": summary["total_build_s"],
                "total_replay_gate_s": summary["total_replay_gate_s"],
                "jax_devices": summary["jax_devices"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
