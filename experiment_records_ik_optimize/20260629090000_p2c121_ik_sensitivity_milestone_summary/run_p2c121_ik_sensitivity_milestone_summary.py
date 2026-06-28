"""Summarize the current IK sensitivity milestone for n1 and n2."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


SOURCE_PATHS = {
    "n2_contract": ROOT / "experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract/summary.json",
    "n2_policy": ROOT / "experiment_records_ik_optimize/20260629080000_p2c119_n2_diagnostic_outer_ik_policy/summary.json",
    "n1_boundary": ROOT / "experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/summary.json",
}


def read_json(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_decision_rows(sources: dict[str, dict]) -> list[dict]:
    """Build the milestone decision table."""
    n2_contract = sources["n2_contract"]
    n2_policy = sources["n2_policy"]
    n1_boundary = sources["n1_boundary"]
    return [
        {
            "case": "n2",
            "task": "两点磁控三维末端位置",
            "decision": "分支条件下可发布；只限诊断模式局部导数",
            "publishable": True,
            "outer_ik_default_allowed": False,
            "allowed_use": "可信分支、小电流半径内、位置任务的小步诊断推进",
            "forbidden_use": "完整位姿、跨分支、跨基准电流、大步线性外推、默认生产 IK",
            "key_gate": f"一阶最优性 {n2_contract['interface_snapshot']['evidence']['stationarity_norm']:.3e}; 载荷传播 {n2_contract['interface_snapshot']['evidence']['force_ratio']:.3e}; 线性半径 {n2_contract['max_linear_current_step_A']:.4g} A",
            "efficiency_status": "缓存复核和策略判定已可低成本执行；真实重求解仍需按小步复核",
            "next": "做工作空间方向扫描，形成局部可控锥图；保持诊断模式，不接生产默认外层 IK",
            "source": "P2c117; P2c119",
        },
        {
            "case": "n1",
            "task": "单点磁控有效子空间末端位置",
            "decision": "当前模型和工作点不可发布",
            "publishable": False,
            "outer_ik_default_allowed": False,
            "allowed_use": n1_boundary["decision_interface"]["allowed_use"],
            "forbidden_use": n1_boundary["decision_interface"]["forbidden_use"],
            "key_gate": f"可发布证据 {n1_boundary['publishable_count']}/{n1_boundary['evidence_count']}; 主失败 {n1_boundary['decision_interface']['primary_failure']}",
            "efficiency_status": "停止扰动扫描；避免在失败基线上浪费计算",
            "next": "先修复刚性磁体段载荷传播模型，或降低任务维度并选择更保守目标",
            "source": "P2c120",
        },
    ]


def build_recommendation_rows() -> list[dict]:
    """Build engineering recommendations."""
    return [
        {
            "topic": "n2 可用范围",
            "recommendation": "只发布可信分支条件下的三维末端位置局部导数",
            "reason": "已通过一阶最优性、载荷传播、条件数、随机方向、重复审计和小半径重求解闭环",
        },
        {
            "topic": "n2 步长",
            "recommendation": "单步线性外推半径限定为 0.0125 A",
            "reason": "0.0125 A 通过真实重求解闭环；5 A 级外推在 P2c115 失败",
        },
        {
            "topic": "n2 外层 IK",
            "recommendation": "只允许诊断模式小步推进，不允许默认生产接入",
            "reason": "方向性很强，弱方向可能需要数百到数千步",
        },
        {
            "topic": "n1 当前工作点",
            "recommendation": "不可发布，不扰动扫描，不接外层 IK",
            "reason": "物理闭合、任务拟合、一阶最优性和条件数不能同时满足",
        },
        {
            "topic": "n1 后续路线",
            "recommendation": "先做模型修复或任务降维，再重启基线门禁",
            "reason": "主失败是刚性磁体段载荷传播失配与模型耦合缺失",
        },
    ]


def plot_decisions(rows: list[dict]) -> None:
    """Plot milestone decisions."""
    configure_matplotlib_cjk()
    labels = [row["case"] for row in rows]
    publishable = [1 if row["publishable"] else 0 for row in rows]
    default_ik = [1 if row["outer_ik_default_allowed"] else 0 for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], publishable, width=0.36, label="可发布局部灵敏度")
    ax.bar([i + 0.18 for i in x], default_ik, width=0.36, label="默认外层 IK")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("状态")
    ax.set_title("IK 灵敏度当前里程碑判定")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT_DIR / "ik_sensitivity_milestone_summary_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Generate the milestone summary package."""
    t0 = time.perf_counter()
    sources = {key: read_json(path) for key, path in SOURCE_PATHS.items()}
    decision_rows = build_decision_rows(sources)
    recommendation_rows = build_recommendation_rows()
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "milestone_summarized_goal_not_complete",
        "n2_status": "branch_conditioned_position_sensitivity_publishable_diagnostic_only",
        "n1_status": "not_publishable_current_model",
        "completion_audit": {
            "n2_three_dimensional_position": "achieved with strict scope limits",
            "n1_effective_subspace_or_boundary": "boundary achieved for current model/workpoint",
            "full_pose_sensitivity": "not achieved and explicitly out of current publishable scope",
            "global_outer_ik": "not achieved; only diagnostic local stepping is supported",
            "remaining_required_work": "workspace direction scan for n2; optional n1 physics-model repair before retest",
        },
        "source_experiments": {key: str(path.relative_to(ROOT)) for key, path in SOURCE_PATHS.items()},
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "decision_rows.csv", decision_rows)
    write_csv(OUT_DIR / "recommendation_rows.csv", recommendation_rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    plot_decisions(decision_rows)

    decision_lines = "\n".join(
        f"- {row['case']}：{row['decision']}。{row['key_gate']}。"
        for row in decision_rows
    )
    rec_lines = "\n".join(
        f"- {row['topic']}：{row['recommendation']}。原因：{row['reason']}。"
        for row in recommendation_rows
    )
    record = f"""# P2c121 IK 灵敏度阶段里程碑综合报告

## 实验目的

本轮不新增求解实验，而是综合 P2c117、P2c119 和 P2c120 的当前证据，形成单点磁控和两点磁控导管逆运动学灵敏度研究的阶段性判定。重点区分“可发布局部灵敏度”“诊断模式外层 IK”和“不可发布边界”。

## 当前判定

{decision_lines}

## 工程建议

{rec_lines}

## 结论

两点磁控导管已获得受限可发布结果：同一模型、同一目标点、同一基准电流、同一可信分支下的三维末端位置局部导数。该结果通过物理门禁、随机方向审计、重复审计和小电流半径真实重求解闭环；但只能用于诊断模式小步推进，不允许完整位姿发布、跨分支复用或默认生产外层 IK。

单点磁控导管当前模型和工作点被判定为不可发布。证据链显示，载荷传播闭合、任务拟合、一阶最优性和条件数不能同时满足。其主失败机制是刚性磁体段外力向柔性段内力链传播的模型耦合缺失。

## 完成度审计

- 两点磁控三维末端位置灵敏度：已达到受限发布条件。
- 单点磁控有效子空间或严格边界：当前工作点已形成严格不可发布边界。
- 完整末端位姿灵敏度：未达成，且当前不作为发布目标。
- 全局外层 IK：未达成；当前只支持诊断模式局部小步策略。

## 下一步计划

优先做 n2 工作空间方向扫描，输出局部可控锥和弱方向图。若继续 n1，应先提出刚性磁体段载荷传播模型修复方案，再重新从基线门禁开始，而不是继续扰动扫描。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
