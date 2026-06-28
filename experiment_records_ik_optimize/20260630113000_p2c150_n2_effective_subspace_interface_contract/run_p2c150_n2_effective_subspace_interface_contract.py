"""Generate the n2 effective-subspace interface contract artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from n2_effective_subspace_interface_contract import N2EffectiveSubspaceInterfaceContract


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    if not rows:
        raise ValueError("no rows")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_examples(contract: N2EffectiveSubspaceInterfaceContract) -> list[dict[str, object]]:
    """Build representative request validation examples."""
    valid = {
        "schema_version": contract.schema_version,
        "magnet_count": 2,
        "terminal_quantity": "position",
        "task_representation": "effective_subspace",
        "requested_task_dimension": 2,
        "branch_conditioned": True,
        "current_step_radius_A": 0.001,
        "use_for_default_outer_ik": False,
        "request_full_3d_position": False,
        "request_full_pose": False,
        "generate_current_step": False,
    }
    examples = [
        ("合法只读诊断请求", valid),
        ("错误请求完整三维位置", {**valid, "request_full_3d_position": True, "requested_task_dimension": 3}),
        ("错误接入默认外层逆运动学", {**valid, "use_for_default_outer_ik": True}),
        ("错误请求生成电流步", {**valid, "generate_current_step": True}),
        ("错误超过安全半径", {**valid, "current_step_radius_A": 0.002}),
    ]
    rows = []
    for name, request in examples:
        decision = contract.validate_request(request)
        rows.append(
            {
                "example": name,
                "accepted": decision.accepted,
                "reasons": ";".join(decision.reasons) if decision.reasons else "无",
            }
        )
    return rows


def plot_contract(rows: list[dict[str, object]]) -> None:
    """Plot validation example outcomes."""
    configure_matplotlib_cjk()
    labels = [str(row["example"]) for row in rows]
    values = [1 if row["accepted"] else 0 for row in rows]
    colors = ["#2a9d8f" if value else "#e76f51" for value in values]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    ax.set_yticks([0, 1], ["拒绝", "允许"])
    ax.set_ylim(-0.1, 1.2)
    ax.set_title("两点磁控二维有效子空间接口门禁示例")
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        ax.text(i, values[i] + 0.04, "允许" if values[i] else "拒绝", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_effective_subspace_interface_contract_cn.png", dpi=200)
    plt.close(fig)


def write_record(contract: N2EffectiveSubspaceInterfaceContract, examples: list[dict[str, object]]) -> None:
    """Write the Chinese experiment record."""
    status = contract.evidence_status()
    example_table = [
        "| 示例 | 判定 | 原因 |",
        "|---|---|---|",
    ]
    for row in examples:
        example_table.append(
            f"| {row['example']} | {'允许' if row['accepted'] else '拒绝'} | {row['reasons']} |"
        )
    text = f"""# P2c150 两点磁控二维有效子空间接口契约

## 实验目的

P2c149 已给出两点磁控二维有效子空间候选证据。本轮将该证据转化为只读接口契约，明确允许范围、禁止范围、请求字段和失败回退规则，防止后续把二维局部诊断误用为完整三维逆运动学模型。

## 证据基础

- 来源实验：{status['source_experiment']}。
- 候选工作点数量：{status['candidate_count']}。
- 有效任务维数：{status['effective_subspace_dimension']}。
- 安全电流扰动半径：{status['safe_radius_A']} A。
- 完整三维位置是否可发布：{'是' if status['full_3d_position_publishable'] else '否'}。
- 是否允许默认接入外层逆运动学：{'是' if status['outer_ik_allowed'] else '否'}。
- 是否已经保存数值基向量：{'是' if status['numeric_basis_available'] else '否'}。

## 接口定义

该接口只允许表达“同一物理解分支、小扰动半径内的二维有效子空间位置灵敏度诊断”。这里的二维有效子空间，是局部灵敏度矩阵中前两个可靠主响应方向张成的空间；它不是固定世界坐标平面，也不是完整三维控制能力。

## 禁止事项

- 不得发布完整三维末端位置灵敏度。
- 不得发布完整末端位姿灵敏度。
- 不得跨工作点、跨分支或跨模型复用。
- 不得作为默认外层逆运动学模型。
- 在未保存数值基向量前，不得根据该接口生成电流增量。

## 门禁示例

{chr(10).join(example_table)}

## 实验结论

当前两点磁控结果已具备二维有效子空间的只读诊断接口候选价值，但还不具备数值控制接口价值。若后续要进入外层逆运动学，需要先补充每个工作点的数值主方向基、任务投影矩阵和电流步生成审计，并重新通过扰动重求解一致性门禁。

## 下一步计划

下一轮建议开展 P2c151：重新生成或补充保存每个工作点的数值子空间基向量，形成可复核的投影矩阵文件；随后只在二维子空间内测试小步任务闭环，不触碰完整三维外层逆运动学。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Generate P2c150 artifacts."""
    contract = N2EffectiveSubspaceInterfaceContract(ROOT)
    examples = validation_examples(contract)
    snapshot = contract.as_dict()
    snapshot["validation_examples"] = examples
    summary = {
        "experiment_id": "20260630113000_p2c150_n2_effective_subspace_interface_contract",
        "source_experiment": contract.summary["experiment_id"],
        "contract_schema_version": contract.schema_version,
        "allowed_scope": contract.allowed_scope(),
        "forbidden_scope": contract.forbidden_scope(),
        "local_2d_interface_candidate": True,
        "numeric_basis_available": False,
        "can_generate_current_step": False,
        "outer_ik_allowed": False,
        "next_required_artifact": "per-workpoint numeric effective-subspace basis",
    }
    (OUT_DIR / "interface_contract.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_DIR / "validation_examples.csv", examples)
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(
            {
                "failure_class": "数值子空间基缺失",
                "interpretation": "当前证据可支持二维有效子空间只读诊断，但不能生成电流步。",
                "next_action": "补充保存每个工作点的主方向基和投影矩阵。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_contract(examples)
    write_record(contract, examples)


if __name__ == "__main__":
    main()
