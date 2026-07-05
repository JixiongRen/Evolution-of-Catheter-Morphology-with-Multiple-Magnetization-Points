"""Synthesize the n1 effective-subspace release boundary."""

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
    """Load one existing summary."""
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
    """Build evidence rows from n1 reports."""
    p82 = load_summary("20260628120813_p2c82_n1_effective_subspace_boundary")
    p99 = load_summary("20260628215500_p2c99_n1_nonpublishable_boundary_synthesis")
    p120 = load_summary("20260629083000_p2c120_n1_publishability_boundary_package")
    p126 = load_summary("20260629140000_p2c126_n1_failure_mechanism_repair_map")
    p128 = load_summary("20260629160000_p2c128_n1_readonly_rigid_load_observer")
    p129 = load_summary("20260629170000_p2c129_n1_explicit_rigid_load_residual")
    p130 = load_summary("20260629190000_p2c130_n1_task_reduction_conservative_workpoint")
    p131 = load_summary("20260629210000_p2c131_n1_configuration_boundary")
    p140 = load_summary("20260629003000_p2c140_n1_continuation_branch_current_reopt")
    p177 = load_summary("20260705035000_p2c177_n1_release_boundary_synthesis")
    p82_rows = p82["rows"]
    p99_rows = p99["rows"]
    return [
        {
            "source": "P2c82",
            "evidence_type": "有效子空间初判",
            "status": "不可发布",
            "key_metric": "rank=2; stationarity=2.447; condition=4.495e9; force_ratio=17.71",
            "decision": "单点存在二维响应线索，但原始物理基线严重失败。",
        },
        {
            "source": "P2c82/P2c64-P2c65",
            "evidence_type": "近门诊断",
            "status": "诊断，不发布",
            "key_metric": "best stationarity=1.065e-6; condition=8.113e6; force_ratio=2.769",
            "decision": "一阶最优性和条件数可被调到近门或过门，但载荷传播比例仍大于 2。",
        },
        {
            "source": "P2c66",
            "evidence_type": "载荷门禁一致性",
            "status": "不可发布",
            "key_metric": "publish 0/18; diagnostic 4/18; best force_ratio=2.734",
            "decision": "失败不是门禁统计口径错误。",
        },
        {
            "source": "P2c67",
            "evidence_type": "硬载荷闭合",
            "status": "不可发布",
            "key_metric": "tip_error=5.93e-3; stationarity=1.32e-3; condition=1.212e9",
            "decision": "硬物理闭合后末端任务和边界相容性退化。",
        },
        {
            "source": "P2c99",
            "evidence_type": "非发布边界综合",
            "status": p99["status"],
            "key_metric": p99["decision"]["primary_failure"],
            "decision": "主失败为刚性磁体段载荷传播失配。",
        },
        {
            "source": "P2c120",
            "evidence_type": "发布边界包",
            "status": p120["status"],
            "key_metric": ", ".join(p120["failure_classes"]),
            "decision": "失败类别集中在载荷传播、接口连续、一阶最优性和刚段载荷链。",
        },
        {
            "source": "P2c126",
            "evidence_type": "修复图谱",
            "status": p126["status"],
            "key_metric": p126["recommended_next"],
            "decision": "应先做显式刚性磁体段载荷传播原型，不应直接扰动扫描。",
        },
        {
            "source": "P2c128",
            "evidence_type": "只读刚段载荷观测",
            "status": p128["status"],
            "key_metric": "still_not_publishable",
            "decision": "只读观测未改变不可发布结论。",
        },
        {
            "source": "P2c129",
            "evidence_type": "显式刚段载荷残差",
            "status": p129["status"],
            "key_metric": f"pass_count={p129['pass_count']}",
            "decision": "显式残差最小原型未通过基线。",
        },
        {
            "source": "P2c130",
            "evidence_type": "任务降维保守点",
            "status": p130["status"],
            "key_metric": f"pass_count={p130['pass_count']}",
            "decision": "任务降维后仍未闭合。",
        },
        {
            "source": "P2c131",
            "evidence_type": "配置边界扫描",
            "status": p131["status"],
            "key_metric": f"pass_count={p131['pass_count']}",
            "decision": "仅找到基线候选，不能构成发布灵敏度。",
        },
        {
            "source": "P2c140",
            "evidence_type": "continuation 分支电流重求解",
            "status": "not_publishable",
            "key_metric": "tip_delta_norm_m=1e-13 to 1e-11",
            "decision": "物理门禁可过，但末端位移低于可靠判读阈值，不能发布灵敏度。",
        },
        {
            "source": "P2c177",
            "evidence_type": "严格边界综合",
            "status": p177["status"],
            "key_metric": ", ".join(p177["failure_classes"]),
            "decision": p177["decision"],
        },
        {
            "source": "P2c82/P2c99",
            "evidence_type": "原始证据计数",
            "status": "evidence_table",
            "key_metric": f"P2c82 rows={len(p82_rows)}; P2c99 rows={len(p99_rows)}",
            "decision": "历史证据链覆盖软约束、硬闭合、门禁一致性和边界综合。",
        },
    ]


def release_rows() -> list[dict[str, Any]]:
    """Build release decisions for n1 task layers."""
    return [
        {
            "task_layer": "二维有效子空间末端位置灵敏度",
            "task_dimension": 2,
            "gate_evidence": "rank=2 仅为响应线索；force_ratio=2.769 仍超限",
            "publishable": False,
            "outer_ik_allowed": False,
            "decision_level": "不可发布",
            "reason": "载荷传播失配，软约束近门解不等于物理闭合解。",
        },
        {
            "task_layer": "一维保守末端位置灵敏度",
            "task_dimension": 1,
            "gate_evidence": "任务降维 P2c130 pass_count=0；P2c140 位移低于可靠判读阈值",
            "publishable": False,
            "outer_ik_allowed": False,
            "decision_level": "不可发布",
            "reason": "任务降维后仍未形成可测、可线性化、可复现的物理响应。",
        },
        {
            "task_layer": "完整三维末端位置灵敏度",
            "task_dimension": 3,
            "gate_evidence": "原始基线 stationarity=2.447; condition=4.495e9; force_ratio=17.71",
            "publishable": False,
            "outer_ik_allowed": False,
            "decision_level": "不可发布",
            "reason": "有效驱动秩不足且物理基线失败。",
        },
        {
            "task_layer": "完整末端位姿灵敏度",
            "task_dimension": 6,
            "gate_evidence": "位置灵敏度尚不可发布，姿态层不进入",
            "publishable": False,
            "outer_ik_allowed": False,
            "decision_level": "不可发布",
            "reason": "末端位置层未闭合，不能扩展到完整位姿。",
        },
    ]


def failure_rows() -> list[dict[str, Any]]:
    """Build failure classification rows."""
    return [
        {
            "failure_class": "刚性磁体段载荷传播失配",
            "evidence": "软约束近门 force_ratio 稳定约 2.7；硬闭合后任务和边界退化。",
            "engineering_action": "先重推刚性磁体段外力、外矩到相邻柔顺段的传递方程。",
        },
        {
            "failure_class": "条件数病态",
            "evidence": "原始条件数 4.495e9；硬闭合条件数 1.212e9。",
            "engineering_action": "禁止大规模扰动扫描；先改局部参数化或物理闭合。",
        },
        {
            "failure_class": "任务降维仍未闭合",
            "evidence": "P2c130 降维保守工作点 pass_count=0。",
            "engineering_action": "不要把二维有效秩线索发布为一维/二维灵敏度。",
        },
        {
            "failure_class": "可测响应不足",
            "evidence": "P2c140 continuation 分支末端位移约 1e-13 至 1e-11 m。",
            "engineering_action": "该分支只用于模型诊断，不用于灵敏度发布。",
        },
        {
            "failure_class": "初始化敏感与多平衡竞争",
            "evidence": "P2c177 将分支条件下证据不足列为严格失败机制。",
            "engineering_action": "若继续 n1，需先建立可信分支库，再做扰动审计。",
        },
    ]


def plot(releases: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Plot n1 release boundary."""
    configure_matplotlib_cjk()
    labels = [row["task_layer"].replace("灵敏度", "") for row in releases]
    dims = [row["task_dimension"] for row in releases]
    pass_values = [1 if row["publishable"] else 0 for row in releases]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(np.arange(len(releases)), dims, color="#457b9d")
    axes[0].set_xticks(np.arange(len(releases)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("任务维度")
    axes[0].set_title("候选任务层级")
    axes[0].grid(axis="y", alpha=0.25)
    colors = ["#e76f51" for _ in releases]
    axes[1].bar(np.arange(len(releases)), pass_values, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(np.arange(len(releases)), labels, rotation=25, ha="right")
    axes[1].set_ylabel("发布状态")
    axes[1].set_title(f"失败机制数量：{len(failures)}")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c193 单点磁控 n1 发布边界")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c193_n1_release_boundary_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    release_table = [
        "| 任务层级 | 维度 | 发布状态 | 主要原因 |",
        "|---|---:|---|---|",
    ]
    for row in summary["release_rows"]:
        release_table.append(
            f"| {row['task_layer']} | {row['task_dimension']} | {row['decision_level']} | {row['reason']} |"
        )
    evidence_table = [
        "| 来源 | 证据类型 | 状态 | 结论 |",
        "|---|---|---|---|",
    ]
    for row in summary["evidence_rows"]:
        evidence_table.append(f"| {row['source']} | {row['evidence_type']} | {row['status']} | {row['decision']} |")
    text = f"""# P2c193 单点磁控 n1 有效子空间发布边界 v2

## 实验目的

长期目标要求：单点磁控若不能得到可发布有效子空间末端位置灵敏度，必须给出严格证据说明当前模型或配置不可发布。本轮综合 P2c82、P2c99、P2c120、P2c126、P2c128-P2c131、P2c140 和 P2c177，形成单点磁控 n1 的任务分层发布边界。

## 发布边界

{chr(10).join(release_table)}

## 证据链

{chr(10).join(evidence_table)}

## 数据分析

单点磁控 n1 的关键现象是“存在低维响应线索，但不形成物理可信灵敏度”。P2c82 显示近似响应有二维有效秩；但原始基线一阶最优性、条件数和载荷传播均严重失败。软约束调参可把一阶最优性和条件数推近门限，但载荷传播比例仍约为 2.7，高于硬门禁；硬载荷闭合后，末端位置误差、边界相容性和条件数又明显退化。

因此，当前 n1 不是简单缺少扰动半径或初值，而是刚性磁体段载荷传播链未闭合。任务降维和保守工作点没有形成可发布结果；continuation 分支虽能满足部分物理门禁，但末端位移低至 `1e-13` 至 `1e-11 m`，低于可靠判读阈值，只能用于模型诊断。

## 实验结论

{summary['decision']}

## 工程建议

1. 当前 n1 不发布末端位置灵敏度，不发布完整末端位姿灵敏度。
2. 当前 n1 不接入外层 IK。
3. 二维有效秩只能作为诊断线索，不能作为发布依据。
4. 后续若继续 n1，应先修复刚性磁体段载荷传播模型，再从基线门禁重新开始。
5. 修复前不建议继续做大规模扰动扫描或权重扫参。

## 下一步计划

单点磁控主线应转为模型修复：重推刚性磁体段外力和外矩向柔顺段的传递方程，并用 P2c64/P2c65 近门样本作为回归基准。若修复后仍无法同时通过一阶最优性、条件数、载荷传播和局部线性化门禁，则应将当前单点磁化配置列入不可发布配置。

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
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c193_n1_effective_subspace_release_boundary_v2.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run synthesis."""
    evidence = evidence_rows()
    releases = release_rows()
    failures = failure_rows()
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "n1_effective_subspace_not_publishable_v2",
        "decision": "单点磁控 n1 当前模型不发布任何末端位置或末端位姿灵敏度；只保留二维有效秩作为诊断线索。",
        "outer_ik_allowed": False,
        "any_position_sensitivity_publishable": False,
        "full_pose_publishable": False,
        "effective_rank_hint": 2,
        "publishable_effective_subspace_dimension": 0,
        "primary_failure": "刚性磁体段载荷传播失配",
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
    plot(releases, failures)
    write_record(summary)
    write_commands()
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
