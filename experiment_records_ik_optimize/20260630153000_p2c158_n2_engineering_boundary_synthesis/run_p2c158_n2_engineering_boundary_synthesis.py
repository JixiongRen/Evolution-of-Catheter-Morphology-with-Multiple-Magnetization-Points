"""Synthesize the n2 single-step engineering boundary evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


SOURCES = {
    "P2c149": "experiment_records_ik_optimize/20260630110000_p2c149_n2_2d_effective_subspace_release_candidate/summary.json",
    "P2c151": "experiment_records_ik_optimize/20260630120000_p2c151_n2_numeric_subspace_basis/summary.json",
    "P2c152": "experiment_records_ik_optimize/20260630123000_p2c152_n2_2d_subspace_closure/summary.json",
    "P2c154": "experiment_records_ik_optimize/20260630133000_p2c154_n2_2d_cross_radius_closure/summary.json",
    "P2c156": "experiment_records_ik_optimize/20260630143000_p2c156_n2_2d_multistep_prototype/summary.json",
    "P2c157": "experiment_records_ik_optimize/20260630150000_p2c157_n2_single_step_interface/summary.json",
}


def read_json(rel_path: str) -> dict[str, Any]:
    """Read a JSON file relative to the repository root."""
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    if not rows:
        raise ValueError("no rows")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_decision_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the engineering decision table."""
    return [
        {
            "scope": "两点磁控二维有效子空间单步末端位置诊断",
            "decision": "可作为候选接口",
            "evidence": "P2c149-P2c157",
            "limits": "指定工作点、同一分支、0.0010 A 内、单步",
            "reason": "二维数值基存在，0.0010 A 单步闭环全部通过。",
        },
        {
            "scope": "两点磁控完整三维末端位置灵敏度",
            "decision": "不可发布",
            "evidence": "P2c149",
            "limits": "第三主方向弱信号，不可辨识",
            "reason": data["P2c149"]["reason"],
        },
        {
            "scope": "两点磁控完整末端位姿灵敏度",
            "decision": "不可发布",
            "evidence": "P2c149-P2c157",
            "limits": "本阶段只验证末端位置二维子空间",
            "reason": "姿态任务未通过同等物理一致性和局部线性化证据链。",
        },
        {
            "scope": "0.0015 A 二维诊断步",
            "decision": "不可用",
            "evidence": "P2c154",
            "limits": "接口连续残差超限",
            "reason": "0.0015 A 下 15/15 样本失败。",
        },
        {
            "scope": "连续复用固定二维子空间做多步外层 IK",
            "decision": "禁止",
            "evidence": "P2c156",
            "limits": "第二步接口连续残差超限",
            "reason": "第二步 20/20 样本失败，需每步重建局部分支和灵敏度。",
        },
    ]


def plot_boundary(data: dict[str, dict[str, Any]]) -> None:
    """Plot pass rates for the main evidence stages."""
    configure_matplotlib_cjk()
    labels = ["二维候选", "0.001A单步", "0.0015A", "第2步复用"]
    pass_rates = [
        data["P2c149"]["n2_2d_effective_subspace_candidate_count"] / data["P2c149"]["candidate_count"],
        data["P2c152"]["pass_count"] / data["P2c152"]["tested_count"],
        data["P2c154"]["by_radius"]["0.0015"]["pass_count"] / data["P2c154"]["by_radius"]["0.0015"]["tested_count"],
        data["P2c156"]["by_step"]["2"]["pass_count"] / data["P2c156"]["by_step"]["2"]["tested_count"],
    ]
    colors = ["#2a9d8f" if rate == 1.0 else "#e76f51" for rate in pass_rates]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(range(len(labels)), pass_rates, color=colors)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("通过率")
    ax.set_title("两点磁控二维单步灵敏度工程边界")
    ax.grid(axis="y", alpha=0.25)
    for i, rate in enumerate(pass_rates):
        ax.text(i, rate + 0.04, f"{rate:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_engineering_boundary_synthesis_cn.png", dpi=200)
    plt.close(fig)


def write_record(data: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese synthesis report."""
    decision_table = [
        "| 范围 | 判定 | 证据 | 限制 | 原因 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        decision_table.append(
            f"| {row['scope']} | {row['decision']} | {row['evidence']} | {row['limits']} | {row['reason']} |"
        )
    text = f"""# P2c158 两点磁控二维单步灵敏度工程边界综合

## 实验目的

本轮不重新求解物理模型，而是综合 P2c149 至 P2c157 的证据链，形成两点磁控导管当前可用与不可用范围的阶段性工程边界。重点是防止把“二维有效子空间单步诊断能力”误写成完整三维位置灵敏度、完整末端位姿灵敏度或多步外层逆运动学能力。

## 证据链

- P2c149：五个工作点形成二维有效子空间候选，但完整三维位置不可发布。
- P2c151：五个工作点均导出数值基，二维条件数约 {data['P2c151']['condition_2d_min']:.2f} 至 {data['P2c151']['condition_2d_max']:.2f}。
- P2c152：0.0010 A 单步二维闭环 15/15 通过，最大线性化误差 {data['P2c152']['max_linear_rel_err']:.3e}。
- P2c154：跨半径复核显示 0.0010 A 全通过，0.0015 A 因接口连续残差超限全部失败。
- P2c156：连续复用固定二维子空间时，第一步 20/20 通过，第二步 0/20 通过，失败机制为接口连续残差超限。
- P2c157：接口收紧为单步诊断接口，禁止多步复用固定局部子空间。

## 工程判定表

{chr(10).join(decision_table)}

## 数据分析

当前两点磁控结果已经越过“仅有末端拟合表象”的阶段。证据链同时覆盖了基线门禁、数值基、扰动重求解、一阶局部线性化、中心线分支漂移、载荷传播和接口连续残差。可用结论集中在二维有效子空间内，且只适用于 0.0010 A 保守半径内的单步诊断。

不可用边界同样明确。完整三维位置灵敏度不可发布，原因是第三主方向响应过弱，扰动重求解中不能稳定通过局部线性化门禁。0.0015 A 不可用，原因是接口连续残差超限。连续复用固定局部二维子空间不可用，原因是第二步接口连续残差超限，而不是中心线分支漂移或载荷传播失控。

## 实验结论

两点磁控当前可发布为“分支条件下、指定工作点、0.0010 A 保守半径内、二维有效子空间的单步末端位置诊断灵敏度候选”。该结论不能外推为完整三维末端位置灵敏度、完整末端位姿灵敏度，也不能直接接入多步外层逆运动学。

## 工程建议

- 可用：五个候选工作点上的二维有效子空间单步诊断。
- 谨慎：若要连续控制，必须每一步重新求解局部物理分支并重建灵敏度。
- 禁止：复用初始工作点固定二维子空间做多步 IK。
- 禁止：将该结果解释为完整三维位置或完整位姿灵敏度。
- 禁止：使用 0.0015 A 或更大半径作为线性诊断步。

## 下一步计划

建议转入两条路线之一。第一，整理单点磁控不可发布边界，补齐长期目标中的单点部分。第二，探索滚动诊断方案：每一步重新建立局部物理解分支和灵敏度，并以 P2c157 单步接口作为每步局部门禁。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Generate P2c158 synthesis artifacts."""
    data = {name: read_json(path) for name, path in SOURCES.items()}
    rows = build_decision_rows(data)
    summary = {
        "experiment_id": "20260630153000_p2c158_n2_engineering_boundary_synthesis",
        "object": "n2 two-magnet catheter",
        "publishable_scope": "branch-conditioned 2D effective-subspace single-step terminal-position diagnostic sensitivity",
        "conservative_radius_A": 0.001,
        "candidate_workpoint_count": data["P2c149"]["candidate_count"],
        "single_step_pass_rate": data["P2c152"]["pass_count"] / data["P2c152"]["tested_count"],
        "multistep_allowed": False,
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "full_pose_publishable": False,
        "dominant_failures": {
            "third_direction": "weak signal and linearization failure",
            "large_radius": data["P2c154"]["failure_by_radius"].get("0.0015", {}),
            "fixed_subspace_multistep": data["P2c156"]["failure_by_step"].get("2", {}),
        },
        "next_recommended_work": "n1 boundary synthesis or rolling local branch-and-sensitivity rebuild",
    }
    write_csv(OUT_DIR / "engineering_decision_rows.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"scope": row["scope"], "decision": row["decision"], "reason": row["reason"]}, ensure_ascii=False)
                for row in rows
                if row["decision"] in {"不可发布", "不可用", "禁止"}
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plot_boundary(data)
    write_record(data, rows)


if __name__ == "__main__":
    main()
