"""Generate the single-step n2 2D diagnostic interface package."""

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
from n2_single_step_diagnostic_interface import N2SingleStepDiagnosticInterface


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_examples(contract: N2SingleStepDiagnosticInterface) -> list[dict[str, object]]:
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
        "step_count": 1,
        "reuse_local_subspace_after_step": False,
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
        ("错误连续复用固定子空间", {**valid, "step_count": 2, "reuse_local_subspace_after_step": True}),
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
    fig.savefig(OUT_DIR / "n2_single_step_diagnostic_interface_cn.png", dpi=200)
    plt.close(fig)


def write_record(contract: N2SingleStepDiagnosticInterface, rows: list[dict[str, object]]) -> None:
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
    cross = contract.cross_radius
    good = cross["by_radius"]["0.0010"]
    failed = cross["by_radius"]["0.0015"]
    multistep = contract.multistep
    text = f"""# P2c157 两点磁控二维单步诊断接口

## 实验目的

P2c155 已固定 0.0010 A 为二维诊断步保守半径。P2c156 进一步显示，连续两步复用同一局部二维子空间时，第二步全部因接口连续残差超限失败。本轮据此将接口收紧为“单步诊断接口”，禁止多步复用固定局部子空间。

## 证据基础

- 数值基来源：P2c151。
- 单半径闭环来源：P2c152。
- 跨半径闭环来源：P2c154。
- 多步原型来源：P2c156。
- 0.0010 A 通过样本：{good['pass_count']} / {good['tested_count']}。
- 0.0010 A 最大局部线性化误差：{good['max_linear_rel_err']:.3e}。
- 0.0010 A 最大中心线分支漂移：{good['max_centerline_change_m']:.3e} m。
- 0.0010 A 最大载荷传播比例：{good['max_force_ratio']:.3f}。
- 0.0015 A 通过样本：{failed['pass_count']} / {failed['tested_count']}。
- 0.0015 A 失败原因：接口连续残差超限。
- 保守电流半径：{contract.safe_radius_A():.4f} A。
- 第一步多步原型通过样本：{multistep['by_step']['1']['pass_count']} / {multistep['by_step']['1']['tested_count']}。
- 第二步多步原型通过样本：{multistep['by_step']['2']['pass_count']} / {multistep['by_step']['2']['tested_count']}。
- 第二步失败原因：接口连续残差超限。

## 接口含义

该接口只在指定工作点、同一物理解分支、二维有效子空间内生成单步诊断小步。对任意末端位移需求，接口先投影到已审计的二维任务子空间，再由局部灵敏度矩阵的前两主方向计算电流增量；若电流步长超过审计半径或请求连续复用固定局部子空间，则拒绝请求。

## 禁止事项

- 不得把该接口解释为完整三维末端位置灵敏度。
- 不得用于完整末端位姿灵敏度。
- 不得跨工作点或跨分支复用。
- 不得作为默认外层逆运动学模型。
- 不得超过 0.0010 A 审计半径做线性外推。
- 不得使用 0.0015 A 作为线性诊断步半径。
- 不得连续复用同一工作点的固定二维子空间。
- 不得把该接口直接用作多步外层逆运动学。

## 门禁示例

{chr(10).join(table)}

## 实验结论

两点磁控当前具备“二维有效子空间 0.0010 A 保守半径内的单步诊断步生成”候选接口。该结论仍是局部、分支条件、小扰动范围内的结论；它不能替代完整三维位置灵敏度，也不能直接作为工程闭环控制器或多步外层逆运动学。

## 下一步计划

下一轮建议转向两条路线之一：一是将两点磁控结论整理为阶段性工程边界报告；二是探索“每一步重新求解局部物理分支并重建灵敏度”的滚动诊断方案，但不得复用初始固定子空间。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Generate P2c157 artifacts."""
    contract = N2SingleStepDiagnosticInterface(ROOT)
    rows = build_examples(contract)
    snapshot = contract.as_dict()
    snapshot["validation_examples"] = rows
    summary = {
        "experiment_id": "20260630150000_p2c157_n2_single_step_interface",
        "schema_version": contract.schema_version,
        "allowed_workpoint_count": len(contract.allowed_workpoints()),
        "conservative_radius_A": contract.safe_radius_A(),
        "failed_radius_A": 0.0015,
        "failed_radius_reason": contract.cross_radius["failure_by_radius"].get("0.0015", {}),
        "can_generate_2d_diagnostic_step": contract.can_generate_step(),
        "multistep_allowed": False,
        "multistep_failure_reason": contract.multistep["failure_by_step"].get("2", {}),
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "full_pose_publishable": False,
        "source_basis_experiment": contract.basis["summary"]["experiment_id"],
        "source_single_radius_closure_experiment": contract.single_radius_closure["experiment_id"],
        "source_cross_radius_experiment": contract.cross_radius["experiment_id"],
        "source_multistep_experiment": contract.multistep["experiment_id"],
        "next_required_artifact": "n2 2D single-step engineering boundary synthesis",
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
