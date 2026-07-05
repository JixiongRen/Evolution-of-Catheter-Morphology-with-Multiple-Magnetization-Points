"""Build and independently replay n2 branch caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
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


def cache_paths(name: str) -> tuple[Path, Path, Path]:
    """Return array, metadata, and context cache paths."""
    stem = f"{name}_mode1_branch_cache"
    return (
        CACHE_DIR / f"{stem}.npz",
        CACHE_DIR / f"{stem}_metadata.json",
        CACHE_DIR / f"{stem}_context.pkl",
    )


def candidate_target(base: dict[str, Any], direction: np.ndarray) -> np.ndarray:
    """Return the target point used by P2c148 for a candidate branch."""
    return np.asarray(base["target_p"], dtype=float) + p2c148.p2c145.p2c144.TARGET_STEP_M * np.asarray(
        direction,
        dtype=float,
    )


def build_branch(base: dict[str, Any], name: str, direction: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the source branch and its serializable function context."""
    branch0 = p2c148.build_candidate_branch(base, name, direction)
    first_row, branch1 = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(
        branch0,
        step_index=1,
        mode_index=MODE_INDEX,
    )
    context = {
        "params": base["params"],
        "target": candidate_target(base, direction),
        "y_ref": np.asarray(base["y_ref"], dtype=float),
    }
    return first_row, branch1, context


def save_cache(name: str, branch: dict[str, Any], first_row: dict[str, Any], context: dict[str, Any]) -> None:
    """Save cacheable branch arrays and rebuild context."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array_path, meta_path, context_path = cache_paths(name)
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
    with context_path.open("wb") as f:
        pickle.dump(context, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "workpoint": name,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "first_step_release_class": first_row.get("release_class"),
        "first_step_linear_rel_err": first_row.get("linear_rel_err"),
        "context_pickle": context_path.name,
        "note": "Arrays and physical rebuild context only; no JAX function is serialized.",
    }
    meta_path.write_text(json.dumps(jsonable(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_cache() -> dict[str, Any]:
    """Build cache files for all candidate workpoints."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    rows: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        t_work = time.perf_counter()
        first_row, branch, context = build_branch(base, name, direction)
        save_cache(name, branch, first_row, context)
        rows.append(
            {
                "workpoint": name,
                "build_s": float(time.perf_counter() - t_work),
                "first_step_release_class": first_row.get("release_class"),
                "first_step_linear_rel_err": first_row.get("linear_rel_err"),
            }
        )
    summary = {
        "phase": "build",
        "status": "cache_build_complete",
        "workpoint_count": len(rows),
        "wall_s": float(time.perf_counter() - t0),
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (EXP_DIR / "build_summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(EXP_DIR / "cache_build_rows.csv", rows)
    return summary


def rebuild_funcs(context: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Rebuild JAX functions from cached physical context."""
    return p2c148.p2c145.p2c79.p2c69._make_funcs(
        context["params"],
        np.asarray(context["target"], dtype=float),
        np.asarray(context["y_ref"], dtype=float),
        p2c148.p2c145.p2c79.p2c69.STAGES[-1][0],
    )


def load_branch(name: str) -> tuple[dict[str, Any], float]:
    """Load cached arrays and rebuild runtime functions."""
    array_path, _meta_path, context_path = cache_paths(name)
    t0 = time.perf_counter()
    with context_path.open("rb") as f:
        context = pickle.load(f)
    _args, funcs = rebuild_funcs(context)
    data = np.load(array_path)
    branch = {
        "name": name,
        "params": context["params"],
        "I0": np.asarray(data["I0"], dtype=float),
        "y": np.asarray(data["y"], dtype=float),
        "basis": np.asarray(data["basis"], dtype=float),
        "S": np.asarray(data["S"], dtype=float),
        "U": np.asarray(data["U"], dtype=float),
        "Vt": np.asarray(data["Vt"], dtype=float),
        "singular_values": np.asarray(data["singular_values"], dtype=float),
        "base_out": {
            "tip_p": np.asarray(data["base_tip"], dtype=float),
            "centerline": np.asarray(data["base_centerline"], dtype=float),
        },
        "funcs": funcs,
    }
    return branch, time.perf_counter() - t0


def replay_gate(branch: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Replay the strict branch-loop gate."""
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


def replay_cache() -> dict[str, Any]:
    """Replay cached branches without source branch objects."""
    t0 = time.perf_counter()
    work_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    names = [path.name.split("_mode1_branch_cache", 1)[0] for path in sorted(CACHE_DIR.glob("*_mode1_branch_cache.npz"))]
    for name in names:
        branch, load_rebuild_s = load_branch(name)
        direction_summaries, work_rows, replay_s = replay_gate(branch)
        for row in work_rows:
            row["workpoint"] = name
        rows.extend(work_rows)
        work_summaries.append(
            {
                "workpoint": name,
                "tested_count": len(work_rows),
                "pass_count": sum(1 for row in work_rows if row["gate_pass"]),
                "load_rebuild_s": float(load_rebuild_s),
                "replay_gate_s": float(replay_s),
                "solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in work_rows)),
                "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in work_rows),
                "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in work_rows),
                "direction_summaries": direction_summaries,
            }
        )
    pass_count = sum(1 for row in rows if row["gate_pass"])
    status = "independent_cache_replay_passed" if pass_count == len(rows) else "independent_cache_replay_failed"
    decision = (
        "独立缓存回放保持全部严格门禁通过；可进入批量多初值求解优化。"
        if pass_count == len(rows)
        else "独立缓存回放未保持全部门禁通过；需检查函数重建约定。"
    )
    summary = {
        "phase": "replay",
        "status": status,
        "decision": decision,
        "workpoint_count": len(work_summaries),
        "tested_count": len(rows),
        "pass_count": pass_count,
        "wall_s": float(time.perf_counter() - t0),
        "total_load_rebuild_s": float(sum(item["load_rebuild_s"] for item in work_summaries)),
        "total_replay_gate_s": float(sum(item["replay_gate_s"] for item in work_summaries)),
        "total_solve_steady_s": float(sum(item["solve_steady_s"] for item in work_summaries)),
        "jax_devices": [str(device) for device in jax.devices()],
        "workpoint_summaries": work_summaries,
        "rows": rows,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(EXP_DIR / "cache_replay_rows.csv", rows)
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "独立缓存回放门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    return summary


def plot(summary: dict[str, Any]) -> None:
    """Plot independent replay runtime and gate metrics."""
    configure_matplotlib_cjk()
    labels = [item["workpoint"] for item in summary["workpoint_summaries"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].bar(x, [item["load_rebuild_s"] for item in summary["workpoint_summaries"]], color="#f4a261", label="加载+函数重建")
    axes[0].bar(x, [item["replay_gate_s"] for item in summary["workpoint_summaries"]], color="#457b9d", label="门禁回放")
    axes[0].set_yscale("log")
    axes[0].set_title("独立回放耗时")
    axes[0].legend()
    axes[1].bar(x, [item["pass_count"] / item["tested_count"] for item in summary["workpoint_summaries"]], color="#2a9d8f")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("严格门禁通过比例")
    axes[2].bar(x, [item["max_task_error"] for item in summary["workpoint_summaries"]], color="#e9c46a")
    axes[2].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_title("最大任务线性化误差")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2独立缓存回放")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_independent_cache_replay_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 通过数 | 加载+函数重建/s | 门禁回放/s | 最大任务误差 | 最大输出误差 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["workpoint_summaries"]:
        table.append(
            f"| {item['workpoint']} | {item['pass_count']}/{item['tested_count']} | "
            f"{item['load_rebuild_s']:.2f} | {item['replay_gate_s']:.2f} | "
            f"{item['max_task_error']:.3e} | {item['max_output_error']:.3e} | "
            f"{'通过' if item['pass_count'] == item['tested_count'] else '不可发布'} |"
        )
    build_summary_path = EXP_DIR / "build_summary.json"
    build_note = ""
    if build_summary_path.exists():
        build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
        build_note = f"- 缓存构建墙钟耗时：{build_summary['wall_s']:.2f} s。"
    text = f"""# P2c186 两点磁控独立缓存构建与回放

## 实验目的

P2c185 在同一进程中从源分支对象继承 JAX 函数。本轮将缓存构建和缓存回放拆开：缓存只保存数组、物理参数、目标点和参考状态；回放阶段重新加载缓存并重建 JAX 函数，验证门禁结果是否仍保持 `40/40` 通过。

## 实验方案

构建阶段保存每个工作点的状态向量、约化基、局部灵敏度矩阵、左右奇异向量、基线末端点、基线中心线，以及用于重建函数的物理参数、目标点和参考状态。回放阶段不使用源分支对象，只从缓存读取上述数据并重建残差函数、雅可比函数和输出函数。

## 结果

{chr(10).join(table)}

## 总结指标

- 总通过数：{summary['pass_count']} / {summary['tested_count']}。
- 独立回放墙钟耗时：{summary['wall_s']:.2f} s。
- 加载和函数重建总耗时：{summary['total_load_rebuild_s']:.2f} s。
- 门禁回放总耗时：{summary['total_replay_gate_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.4f} s。
- 运行设备：{', '.join(summary['jax_devices'])}。
{build_note}

## 实验结论

{summary['decision']}

## 下一步计划

若本轮通过，应进入批量多初值求解优化，减少同一扰动下四类初值的 Python 层循环。同时继续修复 GPU 后端；CPU 结果只能作为保守耗时参考。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c186_n2_independent_cache_replay.py --phase build
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c186_n2_independent_cache_replay.py --phase replay
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("build", "replay", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    """Run cache build, replay, or both."""
    args = parse_args()
    write_commands()
    result: dict[str, Any] | None = None
    if args.phase in {"build", "both"}:
        result = build_cache()
    if args.phase in {"replay", "both"}:
        result = replay_cache()
    print(json.dumps(jsonable(result or {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
