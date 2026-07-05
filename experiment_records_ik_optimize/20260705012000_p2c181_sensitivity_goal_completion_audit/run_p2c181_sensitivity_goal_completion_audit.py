"""Create an evidence audit for the long-running IK sensitivity goal."""

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
    """Load UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write UTF-8 JSON."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build the completion audit artifacts."""
    configure_matplotlib_cjk()
    n1 = load_json(ROOT / "experiment_records_ik_optimize/20260705035000_p2c177_n1_release_boundary_synthesis/summary.json")
    n2_mode1 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop/summary.json"
    )
    n2_boundary = load_json(
        ROOT / "experiment_records_ik_optimize/20260705005000_p2c180_n2_1d_engineering_interface_boundary/summary.json"
    )
    milestone = load_json(
        ROOT / "experiment_records_ik_optimize/20260704235600_p2c178_ik_sensitivity_release_boundary_milestone/summary.json"
    )

    requirements = [
        {
            "item": "单点磁控有效子空间或不可发布证据",
            "status": "已满足",
            "evidence": "P2c177：当前模型不可发布边界",
        },
        {
            "item": "两点磁控三维末端位置或失败边界",
            "status": "已满足",
            "evidence": "P2c175/P2c176/P2c180：完整三维不可发布；第1主方向一维候选",
        },
        {
            "item": "可发布结果通过分支一致性和线性化",
            "status": "部分满足",
            "evidence": "P2c179：一维候选 40/40 通过；不覆盖三维或位姿",
        },
        {
            "item": "不可发布样本失败归因",
            "status": "已满足",
            "evidence": "P2c177/P2c180：载荷传播失配、有效子空间不足、弱方向线性化失败",
        },
        {
            "item": "工程建议",
            "status": "已满足",
            "evidence": "P2c180：允许一维小扰动，禁用三维位置、完整位姿和外层IK直接调用",
        },
        {
            "item": "效率基线和优化对比",
            "status": "部分满足",
            "evidence": "P2c179 记录墙钟与重求解净耗时；GPU 失效导致效率结论不完整",
        },
        {
            "item": "本地 Git 提交",
            "status": "未满足",
            "evidence": ".git 只读，无法创建 index.lock",
        },
    ]
    status_score = {"已满足": 1.0, "部分满足": 0.5, "未满足": 0.0}
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "goal_completion_audit_not_complete",
        "decision": "研究证据已形成单点不可发布边界和两点一维候选边界，但完整目标未完成；主要剩余缺口是三维可发布灵敏度不存在、GPU效率未恢复、Git提交受阻。",
        "n1_status": n1["status"],
        "n2_mode1_status": n2_mode1["status"],
        "n2_boundary_status": n2_boundary["status"],
        "milestone_status": milestone["status"],
        "requirements": requirements,
        "completion_score": sum(status_score[item["status"]] for item in requirements) / len(requirements),
        "publishable_results": [
            "n2 branch-conditioned one-dimensional terminal-position local sensitivity candidate",
        ],
        "non_publishable_results": [
            "n1 current model terminal-position sensitivity",
            "n2 full three-dimensional terminal-position sensitivity",
            "n2 full terminal-pose sensitivity",
        ],
        "blocking_items": [
            "repository metadata is read-only, so local commit cannot be created",
            "JAX CUDA initialization failed in latest runs, so GPU efficiency cannot be claimed",
            "n2 result is one-dimensional only and cannot satisfy full three-dimensional IK control",
        ],
    }
    write_json(EXP_DIR / "summary.json", summary)

    failures = [
        {
            "case": "git_checkpoint",
            "failure_class": "repository_metadata_read_only",
            "evidence": "git add cannot create .git/index.lock",
            "decision": "commit_pending",
        },
        {
            "case": "runtime_efficiency_claim",
            "failure_class": "gpu_unavailable",
            "evidence": "latest P2c179 device is TFRT_CPU_0",
            "decision": "gpu_efficiency_not_claimed",
        },
        {
            "case": "n2_full_task",
            "failure_class": "task_dimension_exceeds_stable_effective_subspace",
            "evidence": "only first principal direction passed strict cross-workpoint gate",
            "decision": "full_3d_not_publishable",
        },
    ]
    (EXP_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    labels = [item["item"] for item in requirements]
    values = [status_score[item["status"]] for item in requirements]
    colors = ["#2a9d8f" if v == 1.0 else "#f4a261" if v == 0.5 else "#e76f51" for v in values]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["未满足", "部分满足", "已满足"])
    ax.set_title("IK灵敏度长期目标完成度审计")
    ax.grid(axis="x", alpha=0.25)
    for idx, item in enumerate(requirements):
        ax.text(values[idx] + 0.02, idx, item["status"], va="center")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "goal_completion_audit_cn.png", dpi=200)
    plt.close(fig)

    rows = [
        "| 要求 | 状态 | 证据 |",
        "|---|---|---|",
    ]
    for item in requirements:
        rows.append(f"| {item['item']} | {item['status']} | {item['evidence']} |")

    record = f"""# P2c181 IK 灵敏度长期目标完成度审计

## 审计目的

本轮不新增数值求解，而是对长期目标做逐条证据审计。审计原则是：只有现有文件、实验结果和命令输出能够直接支持的结论，才记为完成；间接推断或单工作点证据不扩大解释范围。

## 审计结果

{chr(10).join(rows)}

## 当前可以成立的结论

单点磁控当前模型已形成严格不可发布边界。主要原因包括刚性磁体段载荷传播失配、条件数病态、初始化敏感和多平衡竞争。因此单点磁控当前不应接入外层逆运动学。

两点磁控第 1 主方向已形成跨工作点一维局部灵敏度候选。P2c179 中 5 个工作点、40 个严格回环样本全部通过，说明该方向具备较强的分支条件稳定性和局部线性化一致性。

两点磁控完整三维末端位置灵敏度仍不可发布。第 2 主方向严格回环失败，第 3 主方向局部线性化失败。因此当前只能发布受限的一维有效子空间候选，不能宣称完整三维控制能力。

## 剩余缺口

1. Git 元数据目录只读，无法完成本地提交。
2. 最新实验运行在 CPU，GPU 初始化失败，因此效率结论只能记录为 CPU 条件下的墙钟和净求解耗时。
3. 当前通过结果只覆盖两点磁控第 1 主方向，不覆盖完整三维位置和完整末端位姿。
4. 单点磁控如果要继续推进，必须先修复刚性磁体段载荷传播模型，而不是继续扫参。

## 下一步建议

短期先等待或修复 Git 可写状态，分批提交 P2c165-P2c181 积压记录。研究上，若目标是工程可用性，应把两点磁控第 1 主方向封装为受限诊断接口，并把运行时门禁固化。若目标是扩大任务维度，应转向第 2、第 3 主方向失败机制修复，不再把完整三维灵敏度当作当前已达成结果。

## 耗时

本轮为证据审计和图表生成，未重跑长耗时求解。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c181_sensitivity_goal_completion_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
