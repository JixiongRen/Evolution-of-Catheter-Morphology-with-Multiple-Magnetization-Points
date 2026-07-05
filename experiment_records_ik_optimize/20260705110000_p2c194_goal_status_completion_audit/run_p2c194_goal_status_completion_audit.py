"""Audit current completion status for the IK sensitivity goal."""

from __future__ import annotations

import csv
import json
import os
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


def audit_rows() -> list[dict[str, Any]]:
    """Build requirement-level audit rows."""
    n2 = load_summary("20260705100000_p2c192_n2_task_release_boundary_synthesis_v2")
    n1 = load_summary("20260705103000_p2c193_n1_effective_subspace_release_boundary_v2")
    p179 = load_summary("20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop")
    p191 = load_summary("20260705093000_p2c191_n2_batch_system_cache_steady_replay")
    return [
        {
            "requirement": "单点磁控有效子空间或不可发布证据",
            "status": "已满足",
            "evidence": "P2c193",
            "detail": n1["decision"],
            "remaining_risk": "后续若修复刚段载荷传播，需重新审计。",
        },
        {
            "requirement": "两点磁控三维末端位置可发布或失败边界",
            "status": "已满足",
            "evidence": "P2c192",
            "detail": n2["decision"],
            "remaining_risk": "当前只允许一维有效子空间；三维不可发布。",
        },
        {
            "requirement": "可发布结果必须过严格门禁",
            "status": "部分满足",
            "evidence": "P2c179/P2c191",
            "detail": f"n2 第 1 主方向严格回环 {p179['pass_count']}/{p179['tested_count']}；工程缓存回放 {p191['steady_pass_count']}/{p191['steady_tested_count']}。",
            "remaining_risk": "该结果只能称为分支条件下一维候选，不是完整三维灵敏度。",
        },
        {
            "requirement": "不可发布样本需失败归因",
            "status": "已满足",
            "evidence": "P2c192/P2c193 failure_events.jsonl",
            "detail": "n1 归因为刚段载荷传播失配、条件数病态、任务降维未闭合等；n2 归因为有效秩不足、条件数病态、局部线性化失败。",
            "remaining_risk": "失败机制仍是模型层推断，需后续物理模型修复验证。",
        },
        {
            "requirement": "输出工程建议",
            "status": "已满足",
            "evidence": "P2c192/P2c193 record.md",
            "detail": "n2 仅用第 1 主方向局部小扰动；n1 不接入外层 IK；二维/三维任务不发布。",
            "remaining_risk": "工程建议限当前模型和当前磁化配置。",
        },
        {
            "requirement": "效率优化与基线对比",
            "status": "已满足",
            "evidence": "P2c187-P2c191",
            "detail": f"五工作点稳态审计 {p191['steady_wall_s']:.2f}s；线性系统缓存相对重复重建估计提速 {p191['estimated_speedup_vs_rebuild_each_time']:.2f}x。",
            "remaining_risk": "GPU 后端不可用，耗时结论是 CPU 保守估计。",
        },
        {
            "requirement": "每轮本地 Git 提交",
            "status": "未满足",
            "evidence": "git add repeatedly failed",
            "detail": ".git 为只读文件系统，无法创建 .git/index.lock。",
            "remaining_risk": "版本管理要求需外部环境解除只读后补提交。",
        },
        {
            "requirement": "GPU 运行效率验证",
            "status": "未满足",
            "evidence": "stderr CUDA error 100",
            "detail": "JAX CUDA 初始化失败，实际设备为 TFRT_CPU_0。",
            "remaining_risk": "不能声明 GPU 效率和显存表现。",
        },
    ]


def milestone_rows() -> list[dict[str, Any]]:
    """Build concise milestone rows."""
    return [
        {
            "milestone": "n1 发布边界",
            "result": "不可发布",
            "evidence_dir": "20260705103000_p2c193_n1_effective_subspace_release_boundary_v2",
            "claim": "无可发布末端位置或位姿灵敏度；二维有效秩仅作诊断。",
        },
        {
            "milestone": "n2 发布边界",
            "result": "一维候选，三维不可发布",
            "evidence_dir": "20260705100000_p2c192_n2_task_release_boundary_synthesis_v2",
            "claim": "第 1 主方向可作为分支条件下局部导数候选；二维/三维不可发布。",
        },
        {
            "milestone": "n2 效率工程化",
            "result": "可复用审计路径",
            "evidence_dir": "20260705093000_p2c191_n2_batch_system_cache_steady_replay",
            "claim": "五工作点稳态回放 40/40，通过，约 54.92s。",
        },
        {
            "milestone": "版本管理",
            "result": "环境阻断",
            "evidence_dir": "各轮 record.md",
            "claim": "所有后续实验已落盘，但无法 git add。",
        },
    ]


def failure_rows(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return incomplete or blocked rows."""
    return [
        {
            "failure_class": "版本管理环境阻断",
            "evidence": ".git/index.lock 无法创建；文件系统只读。",
            "impact": "无法满足每轮本地提交要求。",
            "next_action": "解除 .git 写权限后分批提交所有实验目录。",
        },
        {
            "failure_class": "GPU 后端不可用",
            "evidence": "JAX CUDA 初始化失败，设备回退到 TFRT_CPU_0。",
            "impact": "不能声明 GPU 效率。",
            "next_action": "修复 CUDA/JAX 运行环境后重跑 P2c191 稳态回放。",
        },
    ]


def plot(audit: list[dict[str, Any]]) -> None:
    """Plot completion status."""
    configure_matplotlib_cjk()
    labels = [row["requirement"] for row in audit]
    score = [{"已满足": 1.0, "部分满足": 0.5, "未满足": 0.0}[row["status"]] for row in audit]
    colors = ["#2a9d8f" if v == 1.0 else "#e9c46a" if v == 0.5 else "#e76f51" for v in score]
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    y = np.arange(len(labels))
    ax.barh(y, score, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("完成度")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("P2c194 IK 灵敏度长期目标完成度审计")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c194_goal_status_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese audit record."""
    audit_table = ["| 要求 | 状态 | 证据 | 说明 |", "|---|---|---|---|"]
    for row in summary["audit_rows"]:
        audit_table.append(f"| {row['requirement']} | {row['status']} | {row['evidence']} | {row['detail']} |")
    milestone_table = ["| 里程碑 | 结果 | 证据目录 | 可声明结论 |", "|---|---|---|---|"]
    for row in summary["milestone_rows"]:
        milestone_table.append(f"| {row['milestone']} | {row['result']} | `{row['evidence_dir']}` | {row['claim']} |")
    text = f"""# P2c194 IK 灵敏度长期目标完成度审计

## 审计目的

本轮不重新求解，而是按长期目标逐条审计当前证据是否足以支撑阶段性结论。重点区分研究结论、工程效率、版本管理和 GPU 环境。

## 要求核对

{chr(10).join(audit_table)}

## 阶段里程碑

{chr(10).join(milestone_table)}

## 总体判断

当前研究结论已经形成清晰边界：单点磁控 n1 当前模型不可发布任何末端位置或位姿灵敏度；两点磁控 n2 不能发布二维或三维末端位置灵敏度，但第 1 主方向可作为分支条件下的一维局部导数候选。该结论不依赖末端拟合表象，而是来自一阶最优性、条件数、载荷传播、局部线性化、多初值分支回环和扰动重求解证据。

形式上，长期目标尚不能标记完全完成，原因有两个外部阻断：第一，`.git` 目录只读，无法完成本地提交；第二，GPU 后端不可用，效率结论只能视为 CPU 保守估计。

## 下一步计划

1. 环境层面：恢复 `.git` 写权限后，分批提交 P2c165-P2c194 实验目录。
2. 环境层面：修复 JAX CUDA 后端后，重跑 P2c191 以获得 GPU 稳态审计耗时。
3. 研究层面：若继续 n1，应先修复刚性磁体段载荷传播模型。
4. 研究层面：若继续 n2，应优先修复第 2 主方向局部坐标或模型耦合，不应直接扩展到三维扰动扫描。

## 版本管理状态

本轮报告生成后将尝试加入本地 `ik_optimize` 分支提交队列。若 `.git` 仍为只读文件系统，则记录为环境阻断。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write reproducible command script."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c194_goal_status_completion_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run goal completion audit."""
    audit = audit_rows()
    milestones = milestone_rows()
    failures = failure_rows(audit)
    complete_count = sum(1 for row in audit if row["status"] == "已满足")
    partial_count = sum(1 for row in audit if row["status"] == "部分满足")
    incomplete_count = sum(1 for row in audit if row["status"] == "未满足")
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "goal_research_boundary_complete_but_environment_blocked",
        "decision": "研究边界与工程边界已形成；由于 Git 只读和 GPU 不可用，长期目标不能形式化标记完成。",
        "research_boundary_complete": True,
        "formal_goal_complete": False,
        "complete_count": complete_count,
        "partial_count": partial_count,
        "incomplete_count": incomplete_count,
        "audit_rows": audit,
        "milestone_rows": milestones,
        "failure_rows": failures,
    }
    write_csv(EXP_DIR / "audit_rows.csv", audit)
    write_csv(EXP_DIR / "milestone_rows.csv", milestones)
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(audit)
    write_record(summary)
    write_commands()
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
