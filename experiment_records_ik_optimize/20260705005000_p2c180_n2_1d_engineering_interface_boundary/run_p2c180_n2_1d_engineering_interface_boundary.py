"""Summarize the n2 one-dimensional sensitivity interface boundary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(ROOT))
from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write a JSON file."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Create a release-boundary package from existing n2 evidence."""
    configure_matplotlib_cjk()
    p2c179 = load_json(
        ROOT
        / "experiment_records_ik_optimize/20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop/summary.json"
    )
    p2c176 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705031500_p2c176_n2_mode2_branch_loop_gate/summary.json"
    )
    p2c175 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705024000_p2c175_n2_cross_workpoint_boundary_synthesis/summary.json"
    )
    p2c177 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705035000_p2c177_n1_release_boundary_synthesis/summary.json"
    )

    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "n2_1d_interface_candidate_with_outer_ik_forbidden",
        "decision": "两点磁控第1主方向具备跨工作点一维局部灵敏度候选资格；完整三维位置和完整位姿仍不可发布。",
        "n2_mode1_cross_workpoint_pass": f"{p2c179['workpoint_pass_count']}/{p2c179['workpoint_count']} workpoints, {p2c179['pass_count']}/{p2c179['tested_count']} samples",
        "n2_mode2_strict_pass": f"{p2c176['pass_count']}/{p2c176['tested_count']} samples",
        "n2_mode3_status": "cross-workpoint linearization failed in prior audit",
        "publishable_1d_branch_conditioned_position": True,
        "publishable_full_3d_position": False,
        "publishable_full_pose": False,
        "outer_ik_allowed": False,
        "engineering_interface": {
            "allowed_task": "terminal position displacement only along the first local principal direction",
            "allowed_radius_A": p2c179["radius_A"],
            "required_runtime_checks": [
                "stationarity <= 1e-5",
                "centerline branch drift <= 1e-3 m",
                "local linearization error < 0.05",
                "current-aware continuity residual <= 1e-3",
                "positive and negative perturbations return to the same branch",
            ],
            "forbidden_tasks": [
                "full three-dimensional terminal position sensitivity",
                "full terminal pose sensitivity",
                "direct outer-IK coupling without rebuilding local branch checks",
            ],
        },
        "single_point_boundary": p2c177["status"],
        "evidence_sources": {
            "n2_mode1": "P2c174 and P2c179",
            "n2_mode2": "P2c176",
            "n2_mode3": "P2c148 and P2c175",
            "n1": "P2c177",
        },
    }
    write_json(EXP_DIR / "summary.json", summary)

    failures = [
        {
            "case": "n2_full_3d_position",
            "failure_class": "task_dimension_exceeds_stable_effective_subspace",
            "evidence": "mode1 passed; mode2 strict branch loop failed; mode3 linearization failed",
            "decision": "not_publishable",
        },
        {
            "case": "n2_full_pose",
            "failure_class": "insufficient_physical_and_linearization_evidence",
            "evidence": "full pose was not part of a passed gate",
            "decision": "not_publishable",
        },
        {
            "case": "outer_ik_connection",
            "failure_class": "insufficient_task_dimension_and_global_branch_evidence",
            "evidence": "only branch-conditioned one-dimensional position sensitivity passed",
            "decision": "forbidden",
        },
    ]
    (EXP_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    labels = ["第1主方向", "第2主方向", "第3主方向", "三维位置", "完整位姿"]
    values = [1, 0, 0, 0, 0]
    colors = ["#2a9d8f" if value else "#e76f51" for value in values]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["不可发布", "候选可发布"])
    ax.set_title("两点磁控局部灵敏度工程发布边界")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.04, "候选" if value else "禁用", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_1d_engineering_boundary_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c180 两点磁控一维灵敏度工程接口与禁用边界

## 实验目的

本轮综合 P2c174、P2c175、P2c176、P2c179 和 P2c177，形成当前两点磁控灵敏度研究的工程接口边界。重点不是扩大结论，而是明确哪些结果可以作为受限候选，哪些结果必须禁止发布或禁止接入外层逆运动学。

## 证据摘要

- 第 1 主方向：P2c179 中 5 个工作点全部通过严格回环，合计 40/40 个样本通过。
- 第 2 主方向：P2c176 中 0/32 个严格回环样本通过。
- 第 3 主方向：P2c148 和 P2c175 显示跨工作点局部线性化失败。
- 单点磁控：P2c177 已给出当前模型不可发布边界。

## 当前可用结论

两点磁控第 1 主方向可以作为“分支条件下的一维末端位置局部灵敏度候选”。这里的一维指末端位移只允许沿局部灵敏度矩阵的第一主方向变化；分支条件指必须从已验证的物理解分支出发，并在扰动后通过回环检查返回同一分支。

该结论不能推广为完整三维末端位置灵敏度，也不能推广为完整末端位姿灵敏度。

## 工程接口建议

1. 允许任务：仅允许沿第 1 主方向的小幅末端位置增量。
2. 推荐扰动半径：`{p2c179['radius_A']:.1e} A`。
3. 必须保留运行时门禁：一阶最优性、中心线分支漂移、局部线性化误差、接口连续残差、正负扰动回环。
4. 禁止任务：三维末端位置直接控制、完整末端位姿控制、未经重建分支门禁的外层 IK 调用。
5. 若工作点变化，必须重新构造局部灵敏度和严格回环，不得复用旧工作点导数。

## 不可发布边界

完整三维位置不可发布，因为第 2 主方向未通过严格回环，第 3 主方向局部线性化失败。完整位姿不可发布，因为当前通过证据只覆盖末端位置的一维方向，没有姿态扰动的物理闭合和分支稳定证据。

## 下一步计划

下一阶段可选择两条路线。若目标是工程可用性，应把第 1 主方向封装成受限诊断接口，并加入每步重建分支和回环检查。若目标是扩大可控维度，应回到模型层面修复第 2、第 3 主方向失败原因，重点检查弱方向输出非线性、状态病态方向和刚性磁体载荷传播耦合。

## 耗时

本轮为证据综合和图表生成，未重跑长耗时求解。主要耗时证据来自 P2c179：墙钟 `{p2c179['wall_s']:.2f} s`，重求解净耗时 `{p2c179['total_solve_steady_s']:.2f} s`，运行设备 `{p2c179['jax_devices'][0] if p2c179['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c180_n2_1d_engineering_interface_boundary.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
