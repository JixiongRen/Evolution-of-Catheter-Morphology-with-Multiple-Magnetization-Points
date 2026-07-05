"""Synthesize the n2 task-release boundary from existing audit evidence."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "experiment_records_ik_optimize"

import sys

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
    """Load one existing experiment summary."""
    return json.loads((SRC_DIR / name / "summary.json").read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def evidence_rows() -> list[dict[str, Any]]:
    """Build compact evidence rows from source summaries."""
    p168 = load_summary("20260704231500_p2c168_n2_linear_model_breakdown_audit")
    p169 = load_summary("20260704234500_p2c169_n2_radius_bracket_linearization_audit")
    p170 = load_summary("20260705001000_p2c170_n2_ill_conditioned_direction_audit")
    p173 = load_summary("20260705013000_p2c173_n2_exact_jacobian_reopt_gate")
    p175 = load_summary("20260705024000_p2c175_n2_cross_workpoint_boundary_synthesis")
    p176 = load_summary("20260705031500_p2c176_n2_mode2_branch_loop_gate")
    p179 = load_summary("20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop")
    p191 = load_summary("20260705093000_p2c191_n2_batch_system_cache_steady_replay")

    rows = [
        {
            "source": "P2c168",
            "task_layer": "三维末端位置",
            "evidence_type": "线性模型断点",
            "status": p168["status"],
            "key_metric": "raw_optimality_condition",
            "value": p168.get("optimality_condition", ""),
            "decision": "原始最优性方程严重病态，不能支撑完整三维灵敏度。",
        },
        {
            "source": "P2c169",
            "task_layer": "三维末端位置",
            "evidence_type": "半径夹逼",
            "status": p169["status"],
            "key_metric": "decision",
            "value": p169["decision"],
            "decision": "缩小扰动半径后输出线性化仍未恢复，完整三维方向不可直接发布。",
        },
        {
            "source": "P2c170",
            "task_layer": "三维末端位置",
            "evidence_type": "病态方向审计",
            "status": p170["status"],
            "key_metric": "decision",
            "value": p170["decision"],
            "decision": "弱奇异方向主导，需降维或重构局部坐标。",
        },
        {
            "source": "P2c173",
            "task_layer": "三维末端位置",
            "evidence_type": "精确雅可比重求解",
            "status": p173["status"],
            "key_metric": "pass_count/tested_count",
            "value": f"{p173['pass_count']}/{p173['tested_count']}",
            "decision": "仅 1/12 通过，不能发布完整三维末端位置灵敏度。",
        },
        {
            "source": "P2c175",
            "task_layer": "主方向分层",
            "evidence_type": "跨工作点边界综合",
            "status": p175["status"],
            "key_metric": "mode_summary",
            "value": json.dumps(p175["mode_summary"], ensure_ascii=False),
            "decision": "第 1 主方向可进入严格复核，第 2 主方向待复核，第 3 主方向不可发布。",
        },
        {
            "source": "P2c176",
            "task_layer": "第 2 主方向",
            "evidence_type": "严格回环",
            "status": p176["status"],
            "key_metric": "pass_count/tested_count",
            "value": f"{p176['pass_count']}/{p176['tested_count']}",
            "decision": "第 2 主方向严格回环全失败，不能作为二维或三维发布基础。",
        },
        {
            "source": "P2c179",
            "task_layer": "第 1 主方向",
            "evidence_type": "跨工作点严格回环",
            "status": p179["status"],
            "key_metric": "pass_count/tested_count",
            "value": f"{p179['pass_count']}/{p179['tested_count']}",
            "decision": "第 1 主方向在五工作点、正负扰动、多初值、回环下全部通过。",
        },
        {
            "source": "P2c191",
            "task_layer": "第 1 主方向工程化",
            "evidence_type": "系统缓存稳态回放",
            "status": p191["status"],
            "key_metric": "steady_pass_count/steady_tested_count",
            "value": f"{p191['steady_pass_count']}/{p191['steady_tested_count']}",
            "decision": "第 1 主方向工程复用路径保持全部门禁通过。",
        },
    ]
    return rows


def release_rows() -> list[dict[str, Any]]:
    """Build task-layer release decisions."""
    p175 = load_summary("20260705024000_p2c175_n2_cross_workpoint_boundary_synthesis")
    p176 = load_summary("20260705031500_p2c176_n2_mode2_branch_loop_gate")
    p179 = load_summary("20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop")
    p191 = load_summary("20260705093000_p2c191_n2_batch_system_cache_steady_replay")
    mode = p175["mode_summary"]
    return [
        {
            "task_layer": "第 1 主方向末端位置灵敏度",
            "task_dimension": 1,
            "effective_rank_requirement": "满足一维有效子空间",
            "strict_gate": f"{p179['pass_count']}/{p179['tested_count']}",
            "engineering_gate": f"{p191['steady_pass_count']}/{p191['steady_tested_count']}",
            "max_linear_error": max(item["max_task_error"] for item in p179["workpoint_summaries"]),
            "decision_level": "分支条件下可用候选",
            "outer_ik_allowed": False,
            "publish_statement": "可作为两点磁控第 1 主方向局部导数候选；需声明分支条件和任务降维。",
        },
        {
            "task_layer": "前两主方向末端位置灵敏度",
            "task_dimension": 2,
            "effective_rank_requirement": "第 2 主方向需严格回环",
            "strict_gate": f"{p176['pass_count']}/{p176['tested_count']}",
            "engineering_gate": "未进入",
            "max_linear_error": max(row["task_pred_vs_actual_rel_err"] for row in p176["rows"]),
            "decision_level": "不可发布",
            "outer_ik_allowed": False,
            "publish_statement": "第 2 主方向严格回环失败，不能扩展为二维任务灵敏度。",
        },
        {
            "task_layer": "完整三维末端位置灵敏度",
            "task_dimension": 3,
            "effective_rank_requirement": "三个方向均需线性化和回环通过",
            "strict_gate": "第 3 主方向 0/5 线性化通过",
            "engineering_gate": "未进入",
            "max_linear_error": mode["3"]["max_linear_error"],
            "decision_level": "不可发布",
            "outer_ik_allowed": False,
            "publish_statement": "第 3 主方向跨工作点线性化失败，且原始最优性系统存在病态；不得发布完整三维末端位置灵敏度。",
        },
    ]


def failure_rows() -> list[dict[str, Any]]:
    """Return classified failure events."""
    return [
        {
            "failure_class": "有效秩不足",
            "task_layer": "完整三维末端位置",
            "evidence": "三维任务中仅第 1 主方向通过严格跨工作点回环；第 2 主方向 0/32，第 3 主方向跨工作点线性化 0/5。",
            "action": "只发布有效一维子空间；不得接入完整三维外层 IK。",
        },
        {
            "failure_class": "条件数病态",
            "task_layer": "完整三维末端位置",
            "evidence": "P2c173 原始最优性条件数约 2.27e13，高于 1e9 硬边界。",
            "action": "禁止做大规模三维扰动扫描；先做任务降维或局部坐标重构。",
        },
        {
            "failure_class": "局部线性化失败",
            "task_layer": "第 2/第 3 主方向",
            "evidence": "第 2 主方向严格回环任务误差超 0.05；第 3 主方向跨工作点最大线性误差约 0.711。",
            "action": "若继续研究，先缩小任务到单方向，再做分支夹逼；不要直接扩展到三维。",
        },
    ]


def plot(releases: list[dict[str, Any]]) -> None:
    """Plot release boundary by task dimension."""
    configure_matplotlib_cjk()
    labels = [row["task_layer"].replace("末端位置灵敏度", "") for row in releases]
    errors = [float(row["max_linear_error"]) for row in releases]
    colors = ["#2a9d8f", "#e76f51", "#e76f51"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(np.arange(len(releases)), errors, color=colors)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(len(releases)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("最大局部线性化误差")
    axes[0].set_title("任务维度与线性化边界")
    axes[0].grid(axis="y", alpha=0.25)
    pass_rates = [1.0, 0.0, 0.0]
    axes[1].bar(np.arange(len(releases)), pass_rates, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(np.arange(len(releases)), labels, rotation=20, ha="right")
    axes[1].set_ylabel("严格门禁通过比例")
    axes[1].set_title("可发布层级")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c192 两点磁控任务发布边界")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c192_n2_task_release_boundary_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    release_table = [
        "| 任务层级 | 维度 | 关键门禁 | 最大误差 | 发布结论 |",
        "|---|---:|---|---:|---|",
    ]
    for row in summary["release_rows"]:
        release_table.append(
            f"| {row['task_layer']} | {row['task_dimension']} | {row['strict_gate']} | "
            f"{row['max_linear_error']:.3e} | {row['decision_level']} |"
        )
    evidence_table = [
        "| 来源 | 证据类型 | 状态 | 结论 |",
        "|---|---|---|---|",
    ]
    for row in summary["evidence_rows"]:
        evidence_table.append(f"| {row['source']} | {row['evidence_type']} | {row['status']} | {row['decision']} |")

    text = f"""# P2c192 两点磁控任务发布边界综合报告

## 实验目的

本轮不重新求解，而是综合 P2c168-P2c191 的证据，给出两点磁控导管末端位置灵敏度的任务分层边界。核心问题是：哪些任务层级可以作为分支条件下的局部灵敏度候选，哪些任务层级必须判为不可发布。

## 实验方案

读取既有实验摘要，按三类证据汇总：局部线性化、严格分支回环、工程化缓存回放。报告区分完整三维末端位置灵敏度、前两主方向二维灵敏度、第 1 主方向一维有效子空间灵敏度。

## 发布边界

{chr(10).join(release_table)}

## 证据链

{chr(10).join(evidence_table)}

## 数据分析

两点磁控当前不能发布完整三维末端位置灵敏度。原因不是末端拟合失败，而是任务空间中只有第 1 主方向通过了正负扰动、多初值和回环门禁。第 2 主方向在严格回环中 `0/32` 通过；第 3 主方向跨工作点线性化失败，最大误差约 `{summary['mode3_max_linear_error']:.3e}`。同时，原始最优性系统条件数约 `{summary['raw_optimality_condition']:.3e}`，说明完整三维任务包含病态方向。

第 1 主方向具备较强证据：五个工作点 `40/40` 严格门禁通过，缓存工程化回放也 `40/40` 通过。该结果只能表述为“分支条件下的一维有效子空间局部导数”，不能伪装成完整三维灵敏度，也不能直接接入完整外层 IK。

## 实验结论

{summary['decision']}

## 工程建议

1. 当前两点磁控配置仅建议使用第 1 主方向进行局部小扰动诊断。
2. 不建议发布二维或三维末端位置灵敏度。
3. 不建议接入外层 IK 自动求电流；若必须使用，只能在可信分支库内做一维微调。
4. 若要推进三维任务，应优先修复病态方向和第 2/第 3 主方向的局部参数化，而不是继续扩大扰动扫描。

## 下一步计划

下一步应转向两个方向：其一，针对第 2 主方向失败做最小模型修复或局部坐标重构；其二，回到单点磁控，形成类似的一维有效子空间或不可发布边界报告。

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
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c192_n2_task_release_boundary_synthesis_v2.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run synthesis."""
    evidence = evidence_rows()
    releases = release_rows()
    failures = failure_rows()
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "n2_task_release_boundary_synthesized_v2",
        "decision": "当前 n2 只支持分支条件下第 1 主方向一维有效子空间候选；二维和完整三维末端位置灵敏度不可发布。",
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "two_dimensional_position_publishable": False,
        "mode1_effective_subspace_candidate": True,
        "raw_optimality_condition": 22657955671247.1,
        "mode3_max_linear_error": 0.711183799733406,
        "evidence_rows": evidence,
        "release_rows": releases,
        "failure_rows": failures,
    }
    write_csv(EXP_DIR / "evidence_rows.csv", evidence)
    write_csv(EXP_DIR / "release_rows.csv", releases)
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(releases)
    write_record(summary)
    write_commands()
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
