"""Create a milestone synthesis for IK sensitivity release boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

import sys

sys.path.insert(0, str(REPO / "experiment_records_ik_optimize"))
from plotting_fonts import configure_matplotlib_cjk  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    """Write JSON with stable indentation."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Write milestone artifacts from existing experiment evidence."""
    configure_matplotlib_cjk()

    summary = {
        "experiment_id": "20260704235600_p2c178_ik_sensitivity_release_boundary_milestone",
        "status": "release_boundary_milestone_partial_success",
        "n1_status": "current_model_not_publishable",
        "n2_status": "one_dimensional_effective_subspace_candidate_only",
        "n1_publishable_terminal_position": False,
        "n1_publishable_full_pose": False,
        "n1_outer_ik_allowed": False,
        "n2_publishable_full_3d_position": False,
        "n2_publishable_full_pose": False,
        "n2_publishable_mode1_candidate": True,
        "n2_publishable_mode2": False,
        "n2_publishable_mode3": False,
        "n2_outer_ik_allowed": False,
        "mode1_strict_branch_loop_pass": "8/8 at rand_05, radius=1e-4 A",
        "mode2_strict_branch_loop_pass": "0/32 at rand_05",
        "mode3_linearization_status": "failed in cross-workpoint audit",
        "primary_blockers": [
            "single-point rigid magnetic segment load propagation mismatch",
            "two-point full-position task dimension exceeds stable effective subspace",
            "ill-conditioned reduced optimality equations",
            "branch stability depends on direction and workpoint",
        ],
        "next_recommended_step": (
            "Run cross-workpoint strict branch-loop validation for the n2 first principal direction; "
            "do not connect n1 or full n2 sensitivity to outer IK yet."
        ),
        "git_commit_status": "blocked_by_read_only_git_directory",
    }
    write_json(ROOT / "summary.json", summary)

    failures = [
        {
            "case": "n1_current_model",
            "classification": "rigid_magnet_load_propagation_mismatch",
            "evidence": "P2c126/P2c128/P2c129/P2c159/P2c177",
            "decision": "not_publishable",
        },
        {
            "case": "n2_full_3d_position",
            "classification": "effective_rank_and_branch_linearity_limit",
            "evidence": "P2c173/P2c175/P2c176",
            "decision": "not_publishable",
        },
        {
            "case": "n2_mode2",
            "classification": "strict_branch_loop_failure",
            "evidence": "P2c176 pass_count=0/32",
            "decision": "not_publishable",
        },
        {
            "case": "git_checkpoint",
            "classification": "repository_metadata_read_only",
            "evidence": ".git not writable; index.lock cannot be created",
            "decision": "commit_pending",
        },
    ]
    (ROOT / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    labels = ["n1末端位置", "n2第1主方向", "n2第2主方向", "n2第3主方向", "n2三维位置"]
    values = [0, 1, 0, 0, 0]
    colors = ["#b23b3b", "#2f8f46", "#b23b3b", "#b23b3b", "#b23b3b"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["不可发布", "候选可发布"])
    ax.set_title("IK灵敏度当前发布边界")
    ax.set_ylabel("严格门禁结论")
    for idx, val in enumerate(values):
        text = "候选" if val else "不可发布"
        ax.text(idx, val + 0.04, text, ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "release_boundary_overview_cn.png", dpi=180)
    plt.close(fig)

    record = """# P2c178 IK 灵敏度发布边界里程碑综合

## 实验目的

本轮不新增求解器实验，而是对近期单点磁控与两点磁控灵敏度证据进行里程碑汇总。目标是明确哪些结果可以作为后续研究主线，哪些结果只能作为失败机制和工程边界证据。

## 证据来源

- 单点磁控：P2c120、P2c126、P2c128-P2c132、P2c138-P2c140、P2c159、P2c177。
- 两点磁控：P2c145、P2c148、P2c165-P2c176。

## 当前结论

单点磁控当前模型不可发布末端位置灵敏度，也不可发布完整末端位姿灵敏度。主要原因不是末端拟合误差本身，而是刚性磁体段载荷传播链路没有稳定闭合，且存在条件数病态、初始化敏感和多平衡竞争。

两点磁控当前不支持发布完整三维末端位置灵敏度。第 1 主方向在 rand_05 工作点通过正负扰动、多初值和回环检查，可作为一维有效子空间候选；第 2 主方向严格回环门禁 0/32 通过；第 3 主方向在跨工作点线性化审计中失败。因此两点磁控当前边界应收敛为“一维有效子空间候选”，不能宣称三维任务灵敏度已经可发布。

## 关键指标

| 对象 | 当前结论 | 关键证据 |
|---|---|---|
| 单点磁控末端位置 | 不可发布 | 刚性磁体段载荷传播失配；P2c177 综合边界 |
| 两点磁控第 1 主方向 | 候选可发布 | P2c174，8/8 严格回环通过 |
| 两点磁控第 2 主方向 | 不可发布 | P2c176，0/32 严格回环通过 |
| 两点磁控第 3 主方向 | 不可发布 | P2c148/P2c175，跨工作点线性化失败 |
| 两点磁控三维末端位置 | 不可发布 | 任务维度超过稳定有效子空间 |

## 工程边界

1. 当前单点磁控不能接入外层逆运动学，只能用于模型失配分析和载荷传播修复对照。
2. 当前两点磁控不能以三维末端位置灵敏度接入外层逆运动学。
3. 两点磁控第 1 主方向可以继续做跨工作点严格回环验证；验证通过前只能称为候选。
4. 所有多平衡区域只能报告分支条件下的局部导数，不能报告全局灵敏度。

## 下一步计划

优先执行两点磁控第 1 主方向的跨工作点严格回环复核。若跨工作点通过，可形成一维有效子空间工程接口；若失败，应转为工作点选择准则和不可发布边界说明。单点磁控下一步不应继续扫参，而应先修复刚性磁体段载荷传播模型。

## 耗时

本轮为证据综合和图表生成，未重跑长耗时求解。
"""
    (ROOT / "record.md").write_text(record, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail

cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260704235600_p2c178_ik_sensitivity_release_boundary_milestone/run_p2c178_ik_sensitivity_release_boundary_milestone.py
"""
    (ROOT / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
