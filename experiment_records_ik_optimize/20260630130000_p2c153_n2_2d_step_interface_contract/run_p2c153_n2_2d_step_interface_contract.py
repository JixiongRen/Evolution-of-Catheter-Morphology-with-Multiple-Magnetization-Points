"""Generate the upgraded n2 2D step interface contract package."""

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
from n2_2d_step_interface_contract import N2TwoDimensionalStepInterfaceContract


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_examples(contract: N2TwoDimensionalStepInterfaceContract) -> list[dict[str, object]]:
    """Build representative validation examples."""
    workpoint = contract.allowed_workpoints()[0]
    valid = {
        "schema_version": contract.schema_version,
        "magnet_count": 2,
        "terminal_quantity": "position",
        "task_representation": "effective_subspace",
        "branch_conditioned": True,
        "workpoint": workpoint,
        "desired_tip_delta_m": [1.0e-10, 0.0, 0.0],
        "use_for_default_outer_ik": False,
        "request_full_3d_position": False,
        "request_full_pose": False,
    }
    examples = [
        ("合法二维小步请求", valid),
        ("错误请求完整三维位置", {**valid, "request_full_3d_position": True}),
        ("错误请求完整末端位姿", {**valid, "request_full_pose": True}),
        ("错误跨工作点", {**valid, "workpoint": "not_audited"}),
        ("错误接入默认外层逆运动学", {**valid, "use_for_default_outer_ik": True}),
        ("错误超过半径", {**valid, "desired_tip_delta_m": [1.0e-3, 0.0, 0.0]}),
    ]
    rows = []
    for name, request in examples:
        decision = contract.validate_step_request(request)
        rows.append(
            {
                "example": name,
                "accepted": decision.accepted,
                "current_step_norm_A": decision.current_step_norm_A,
                "projected_task_norm_m": decision.projected_task_norm_m,
                "reasons": ";".join(decision.reasons) if decision.reasons else "无",
            }
        )
    return rows


def plot_examples(rows: list[dict[str, object]]) -> None:
    """Plot validation outcomes."""
    configure_matplotlib_cjk()
    labels = [str(row["example"]) for row in rows]
    values = [1 if row["accepted"] else 0 for row in rows]
    colors = ["#2a9d8f" if value else "#e76f51" for value in values]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    ax.set_yticks([0, 1], ["拒绝", "允许"])
    ax.set_ylim(-0.1, 1.2)
    ax.set_title("两点磁控二维小步接口门禁")
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        ax.text(i, values[i] + 0.04, "允许" if values[i] else "拒绝", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_2d_step_interface_contract_cn.png", dpi=200)
    plt.close(fig)


def write_record(contract: N2TwoDimensionalStepInterfaceContract, rows: list[dict[str, object]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 示例 | 判定 | 电流步长/A | 原因 |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['example']} | {'允许' if row['accepted'] else '拒绝'} | "
            f"{float(row['current_step_norm_A']):.3e} | {row['reasons']} |"
        )
    text = f"""# P2c153 两点磁控二维小步接口契约升级

## 实验目的

P2c150 只允许二维有效子空间只读诊断，P2c151 补齐了数值基，P2c152 证明二维子空间内 0.001 A 小步闭环全部通过。本轮据此升级接口契约：允许生成二维有效子空间内的小电流诊断步，但仍禁止完整三维位置、完整末端位姿和默认外层逆运动学。

## 证据基础

- 数值基来源：P2c151。
- 闭环验证来源：P2c152。
- P2c152 通过样本：15 / 15。
- 最大局部线性化误差：{contract.closure['max_linear_rel_err']:.3e}。
- 最大中心线分支漂移：{contract.closure['max_centerline_change_m']:.3e} m。
- 最大载荷传播比例：{contract.closure['max_force_ratio']:.3f}。
- 安全电流半径：{contract.safe_radius_A()} A。

## 接口含义

该接口只在指定工作点、同一物理解分支、二维有效子空间内生成诊断小步。对任意末端位移需求，接口先投影到已审计的二维任务子空间，再由局部灵敏度矩阵的前两主方向计算电流增量；若电流步长超过审计半径，则拒绝请求。

## 禁止事项

- 不得把该接口解释为完整三维末端位置灵敏度。
- 不得用于完整末端位姿灵敏度。
- 不得跨工作点或跨分支复用。
- 不得作为默认外层逆运动学模型。
- 不得超过 0.001 A 审计半径做线性外推。

## 门禁示例

{chr(10).join(table)}

## 实验结论

两点磁控当前已经具备“二维有效子空间小步诊断步可生成”的候选接口。该结论仍是局部、分支条件、小扰动半径内的结论；它不能替代完整三维位置灵敏度，也不能直接作为工程闭环控制器。

## 下一步计划

下一轮建议做跨半径复核：在 0.0003 A、0.0005 A、0.0010 A、0.0015 A 下重复二维子空间闭环，确定诊断步半径边界。若 0.0015 A 失败，则给出保守半径；若通过，再考虑小规模外层迭代原型。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Generate P2c153 artifacts."""
    contract = N2TwoDimensionalStepInterfaceContract(ROOT)
    rows = build_examples(contract)
    snapshot = contract.as_dict()
    snapshot["validation_examples"] = rows
    summary = {
        "experiment_id": "20260630130000_p2c153_n2_2d_step_interface_contract",
        "schema_version": contract.schema_version,
        "allowed_workpoint_count": len(contract.allowed_workpoints()),
        "safe_radius_A": contract.safe_radius_A(),
        "can_generate_2d_diagnostic_step": contract.can_generate_step(),
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "full_pose_publishable": False,
        "source_basis_experiment": contract.basis["summary"]["experiment_id"],
        "source_closure_experiment": contract.closure["experiment_id"],
        "next_required_artifact": "cross-radius 2D subspace closure audit",
    }
    (OUT_DIR / "interface_contract.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_DIR / "validation_examples.csv", rows)
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    plot_examples(rows)
    write_record(contract, rows)


if __name__ == "__main__":
    main()
