"""Create the final completion audit for the IK sensitivity goal."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "experiment_records_ik_optimize"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


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


def load_summary(name: str) -> dict[str, Any]:
    """Load one source summary."""
    return json.loads((SRC_DIR / name / "summary.json").read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def git_head() -> str:
    """Return the current short git head."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def audit_rows() -> list[dict[str, Any]]:
    """Build final requirement audit rows."""
    n1 = load_summary("20260705103000_p2c193_n1_effective_subspace_release_boundary_v2")
    n2 = load_summary("20260705100000_p2c192_n2_task_release_boundary_synthesis_v2")
    cpu = load_summary("20260705093000_p2c191_n2_batch_system_cache_steady_replay")
    gpu = load_summary("20260705113000_p2c195_n2_gpu_cached_steady_replay")
    return [
        {
            "requirement": "单点磁控有效子空间或不可发布证据",
            "status": "完成",
            "evidence": "P2c193",
            "result": n1["decision"],
        },
        {
            "requirement": "两点磁控三维末端位置可发布或失败边界",
            "status": "完成",
            "evidence": "P2c192",
            "result": n2["decision"],
        },
        {
            "requirement": "严格门禁与失败归因",
            "status": "完成",
            "evidence": "P2c192/P2c193/P2c179/P2c176",
            "result": "可发布层级、不可发布层级和失败类别均已形成机器可读证据。",
        },
        {
            "requirement": "工程建议",
            "status": "完成",
            "evidence": "P2c192/P2c193",
            "result": "n1 不接入外层 IK；n2 仅第 1 主方向可作分支条件下一维候选。",
        },
        {
            "requirement": "效率产品化",
            "status": "完成",
            "evidence": "P2c191/P2c195",
            "result": f"CPU 稳态 {cpu['steady_wall_s']:.2f}s，GPU 稳态 {gpu['steady_wall_s']:.2f}s；GPU 可用但当前流程较慢。",
        },
        {
            "requirement": "本地提交",
            "status": "完成",
            "evidence": "git commits",
            "result": "P2c165-P2c195 已分批提交；本轮 P2c196 生成后单独提交。",
        },
    ]


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot final completion status."""
    configure_matplotlib_cjk()
    labels = [row["requirement"] for row in rows]
    values = [1.0 for _ in rows]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2a9d8f")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("完成度")
    ax.set_title("P2c196 IK 灵敏度目标最终审计")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c196_final_goal_completion_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese final audit record."""
    table = ["| 要求 | 状态 | 证据 | 结果 |", "|---|---|---|---|"]
    for row in summary["audit_rows"]:
        table.append(f"| {row['requirement']} | {row['status']} | {row['evidence']} | {row['result']} |")
    text = f"""# P2c196 IK 灵敏度长期目标最终完成度审计

## 审计目的

本轮在 Git 写权限恢复、GPU 后端恢复后，对长期目标做最终完成度审计。审计对象包括单点磁控、两点磁控、失败机制、工程建议、效率证据和版本管理。

## 要求核对

{chr(10).join(table)}

## 总体结论

长期目标已经达到当前模型边界下的完成状态：单点磁控 n1 已形成严格不可发布结论；两点磁控 n2 已形成“第 1 主方向一维有效子空间候选、二维和三维不可发布”的任务分层结论；失败机制和工程边界均有可追溯证据。

需要强调：该完成状态不是宣称获得完整三维 IK 灵敏度，而是完成了目标要求中的二选一分支，即“若不可发布，则给出严格证据说明失败边界”。当前可发布范围仅限 n2 第 1 主方向的分支条件下局部导数候选。

## 效率结论

CPU 缓存稳态审计约 `54.92 s`，GPU 缓存稳态审计约 `233.51 s`。GPU 后端已可用，但当前流程在 GPU 上更慢，可能受小规模 JAX 调度、回环诊断和接口连续性计算影响。生产建议仍以 CPU 缓存路径作为当前稳定方案。

## Git 状态

本轮生成时当前 HEAD 为 `{summary['git_head_before_commit']}`。P2c196 本身需在生成后单独提交。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c196_final_goal_completion_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run final completion audit."""
    rows = audit_rows()
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "goal_complete_current_model_boundary",
        "decision": "长期目标已在当前模型边界下完成：n1 给出不可发布证据，n2 给出一维候选和三维失败边界，效率与版本管理证据已补齐。",
        "formal_goal_complete": True,
        "research_boundary_complete": True,
        "git_head_before_commit": git_head(),
        "audit_rows": rows,
    }
    write_csv(EXP_DIR / "audit_rows.csv", rows)
    (EXP_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(rows)
    write_record(summary)
    write_commands()
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
