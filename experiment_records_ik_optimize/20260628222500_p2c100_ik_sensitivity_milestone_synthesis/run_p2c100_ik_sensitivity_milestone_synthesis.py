"""Create a milestone synthesis for n1 and n2 IK sensitivity."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


OUT_DIR = Path(__file__).resolve().parent
configure_matplotlib_cjk()

SOURCES = {
    "P2c97": ROOT / "experiment_records_ik_optimize/20260628211500_p2c97_n2_publish_candidate_synthesis/summary.json",
    "P2c98": ROOT / "experiment_records_ik_optimize/20260628213500_p2c98_n2_eps0125_random_current_batch/summary.json",
    "P2c99": ROOT / "experiment_records_ik_optimize/20260628215500_p2c99_n1_nonpublishable_boundary_synthesis/summary.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build milestone decision rows."""
    p98 = data["P2c98"]
    p98_rows = p98["audit_rows"]
    p99 = data["P2c99"]
    n1_max_load = max(row["load_ratio"] for row in p99["rows"])
    return [
        {
            "object": "n1 单点磁控",
            "task": "有效子空间末端位置",
            "decision": "当前模型不可发布",
            "publishable": False,
            "main_evidence": "P2c99",
            "hard_gate_summary": f"载荷传播比例约 {n1_max_load:.3f}，高于 1.5",
            "failure_class": "刚性磁体段载荷传播失配；模型耦合缺失",
            "allowed_use": "诊断与模型修复",
            "forbidden_use": "发布灵敏度；接入外层 IK",
            "next_step": "重写接口力连续残差或显式刚段力传递参数",
        },
        {
            "object": "n2 两点磁控",
            "task": "三维末端位置",
            "decision": "分支条件发布候选",
            "publishable": True,
            "main_evidence": "P2c97-P2c98",
            "hard_gate_summary": (
                f"P2c98 随机电流方向 {sum(row['status'] == 'publishable' for row in p98_rows)}/{len(p98_rows)} 通过，"
                f"最大线性化误差 {max(row['vector_rel_err'] for row in p98_rows):.3e}"
            ),
            "failure_class": "初始化敏感；多平衡分支竞争",
            "allowed_use": "受信任分支条件局部导数；position-only 诊断",
            "forbidden_use": "初始化无关结论；完整末端位姿灵敏度；默认生产 IK",
            "next_step": "可靠分支选择与效率优化",
        },
        {
            "object": "n2 两点磁控",
            "task": "完整末端位姿",
            "decision": "不可发布",
            "publishable": False,
            "main_evidence": "P2c97",
            "hard_gate_summary": "近期仅推进 position-only；姿态任务未通过完整证据链",
            "failure_class": "证据不足；任务维度尚未建立可发布门禁",
            "allowed_use": "诊断记录",
            "forbidden_use": "发布；接入外层 IK",
            "next_step": "待 position-only 分支选择稳定后再审计",
        },
    ]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot milestone decisions."""
    labels = [row["object"] + "\n" + row["task"] for row in rows]
    publish = [1 if row["publishable"] else 0 for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#2a9d8f" if value else "#e76f51" for value in publish]
    ax.bar(labels, publish, color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1], ["不可发布", "候选可发布"])
    ax.set_title("P2c100 IK 灵敏度阶段性工程判定")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ik_sensitivity_milestone_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    """Render the Chinese milestone report."""
    table = [
        "| 对象 | 任务 | 判定 | 主证据 | 硬门禁摘要 | 失败或风险分类 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['object']} | {row['task']} | {row['decision']} | {row['main_evidence']} | "
            f"{row['hard_gate_summary']} | {row['failure_class']} |"
        )
    use_table = [
        "| 对象 | 可用范围 | 禁止范围 | 下一步 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        use_table.append(
            f"| {row['object']} {row['task']} | {row['allowed_use']} | {row['forbidden_use']} | {row['next_step']} |"
        )
    return "\n".join(
        [
            "# P2c100 IK 灵敏度阶段性里程碑综合报告",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是综合 P2c97、P2c98 和 P2c99，给出当前单点与两点磁控导管逆运动学灵敏度研究的阶段性工程判定。报告目标是明确哪些结果已经可以作为分支条件候选，哪些结果应停止发布尝试并转入模型修复。",
            "",
            "## 总体判定",
            "",
            *table,
            "",
            "## 可用范围与禁止范围",
            "",
            *use_table,
            "",
            "## 主要结论",
            "",
            "两点磁控 n2 已经得到较强的分支条件 position-only 灵敏度证据。该证据同时包含目标点闭环、三档电流扰动重求解和随机电流方向批量复核。它可以作为“受信任分支条件下三维末端位置局部导数”的阶段性发布候选。",
            "",
            "单点磁控 n1 目前不能发布有效子空间灵敏度。尽管近门样本的一阶最优性和条件数已达到继续分析要求，但载荷传播比例稳定超限，且多轮反事实诊断排除了简单权重或边界修复。其问题应归入刚性磁体段载荷传播失配和模型耦合缺失。",
            "",
            "完整末端位姿灵敏度仍不具备发布条件。当前证据链只支持 position-only 任务；若强行加入姿态约束，会把任务维度、分支选择和物理闭合问题混合，降低可解释性。",
            "",
            "## 工程建议",
            "",
            "1. n2 可先作为离线诊断模块使用，输出必须标注“受信任分支条件”。",
            "2. n2 若要进入外层 IK，应先实现可靠分支选择机制，并降低批量审计耗时。",
            "3. n1 不应继续做扰动灵敏度扫描；下一步应重写或修复刚段-柔性段接口力连续模型。",
            "4. 完整末端位姿任务暂缓，待 position-only 分支选择和模型闭合稳定后再推进。",
            "",
            "## 下一步计划",
            "",
            "1. n2：整理可复用的分支条件灵敏度接口，记录输入、输出、门禁和缓存策略。",
            "2. n2：做效率优化，目标是在不降低门禁的前提下减少批量复核耗时。",
            "3. n1：开展接口力连续模型修复，复测 P2c84 近门样本。",
            "",
        ]
    )


def main() -> None:
    data = {name: _read_json(path) for name, path in SOURCES.items()}
    rows = _rows(data)
    payload = {
        "experiment": OUT_DIR.name,
        "status": "milestone_synthesis",
        "sources": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
        "rows": rows,
        "decision": {
            "n2_position_only_branch_conditioned": "publish_candidate",
            "n2_full_pose": "not_publishable",
            "n2_production_default": "not_ready",
            "n1_current_model": "not_publishable",
            "recommended_next": ["n2_sensitivity_interface_and_efficiency", "n1_interface_force_model_repair"],
        },
    }
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [{"failure": row["failure_class"], "evidence": row} for row in rows if not row["publishable"]]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
