from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

SOURCES = {
    "single_step": "20260630150000_p2c157_n2_single_step_interface",
    "multistep_fixed": "20260630143000_p2c156_n2_2d_multistep_prototype",
    "boundary": "20260630170000_p2c160_n2_3d_position_boundary_synthesis",
}


def load_summary(key: str) -> dict:
    path = ROOT / "experiment_records_ik_optimize" / SOURCES[key] / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source_path"] = str(path.relative_to(ROOT))
    return data


def configure_chinese_font() -> None:
    candidates = [
        ROOT / "assets" / "fonts" / "SourceHanSerifCN-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    single = load_summary("single_step")
    fixed = load_summary("multistep_fixed")
    boundary = load_summary("boundary")

    protocol_steps = [
        {
            "step": 1,
            "name": "当前分支重求解",
            "required": "以上一步接受状态为初值，对当前电流重新求解物理分支。",
            "gate": "一阶最优性、载荷传播、接口连续、中心线分支漂移均需重新通过。",
        },
        {
            "step": 2,
            "name": "局部灵敏度重建",
            "required": "在当前分支重新构造末端位置灵敏度矩阵并做奇异值分解。",
            "gate": "只保留通过有效秩与条件数门禁的前两主方向；第三方向默认不可发布。",
        },
        {
            "step": 3,
            "name": "二维任务投影",
            "required": "将目标末端位移投影到当前二维有效子空间。",
            "gate": "投影外残差必须报告；不得伪装成完整三维跟踪。",
        },
        {
            "step": 4,
            "name": "保守电流步生成",
            "required": "按当前二维基计算电流增量，并限制在 0.001 A 半径内。",
            "gate": "超过半径则缩步或拒绝；禁止使用 0.0015 A 作为默认半径。",
        },
        {
            "step": 5,
            "name": "扰动重求解闭合",
            "required": "对新电流重新求解物理状态，比较预测位移与实际位移。",
            "gate": "局部线性化误差 <0.05，中心线漂移 <1e-3 m，载荷传播比例 <=1.5。",
        },
    ]

    decision_rows = [
        {
            "item": "固定二维子空间多步复用",
            "decision": "拒绝",
            "reason": "P2c156 第二步 0/20 通过，接口连续残差超限。",
        },
        {
            "item": "滚动局部分支重建",
            "decision": "允许进入原型验证",
            "reason": "每步重新求解分支并重建二维基，可直接针对固定子空间失败机制。",
        },
        {
            "item": "完整三维末端位置控制",
            "decision": "拒绝发布",
            "reason": "P2c160 第三主方向 0/5 通过，线性化误差超限。",
        },
        {
            "item": "二维有效子空间单步诊断",
            "decision": "保留",
            "reason": "P2c157 已收紧为单步诊断接口，半径 0.001 A。",
        },
    ]

    failure_events = [
        {
            "failure_class": "固定子空间复用失败",
            "evidence": fixed["_source_path"],
            "interpretation": "第二步不重新建模时，接口连续残差跨过门禁。",
            "repair_hypothesis": "每一步重新求解分支并重建局部二维有效子空间。",
        },
        {
            "failure_class": "完整三维有效秩不足",
            "evidence": boundary["_source_path"],
            "interpretation": "第三主方向扰动响应不能稳定线性化。",
            "repair_hypothesis": "滚动协议仍只发布二维子空间，不扩展三维声明。",
        },
    ]

    summary = {
        "experiment_id": "20260630173000_p2c161_n2_rolling_local_rebuild_protocol",
        "status": "rolling_local_rebuild_protocol_defined_not_yet_validated",
        "purpose": "define_n2_rolling_branch_and_sensitivity_rebuild_protocol",
        "fixed_subspace_multistep_allowed": False,
        "rolling_rebuild_prototype_allowed": True,
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "effective_subspace_dimension": 2,
        "current_step_radius_A": single.get("conservative_radius_A", 0.001),
        "required_gates": {
            "stationarity_norm": "<=1e-5",
            "centerline_branch_drift_m": "<1e-3",
            "linearization_error": "<0.05",
            "force_ratio": "<=1.5 preferred",
            "continuity_norm": "must not exceed audited interface threshold",
        },
        "source_summaries": {key: value["_source_path"] for key, value in {"single_step": single, "multistep_fixed": fixed, "boundary": boundary}.items()},
        "next_experiment": "P2c162 run one rolling two-step prototype on one n2 workpoint; compare against P2c156 fixed-subspace failure.",
    }

    write_json(EXP_DIR / "summary.json", summary)
    write_csv(EXP_DIR / "protocol_steps.csv", protocol_steps)
    write_csv(EXP_DIR / "engineering_decision_rows.csv", decision_rows)
    write_jsonl(EXP_DIR / "failure_events.jsonl", failure_events)

    protocol = {
        "name": "n2_rolling_local_branch_sensitivity_rebuild",
        "scope": "branch-conditioned 2D effective-subspace terminal-position sensitivity",
        "not_scope": ["full 3D terminal-position sensitivity", "full terminal-pose sensitivity", "default outer IK"],
        "loop": protocol_steps,
        "acceptance": summary["required_gates"],
        "stop_rules": [
            "If current step exceeds 0.001 A, shrink or reject.",
            "If third singular direction is required, reject full 3D task claim.",
            "If any gate fails, stop and classify failure; do not continue multistep scan.",
        ],
    }
    write_json(EXP_DIR / "rolling_protocol_contract.json", protocol)

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [row["name"] for row in protocol_steps]
    ax.plot(range(1, len(labels) + 1), [1] * len(labels), marker="o", linewidth=2, color="#4C78A8")
    ax.set_yticks([])
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
    ax.set_title("两点磁控 n2 滚动局部分支重建协议")
    for row in protocol_steps:
        ax.text(row["step"], 1.04, f"门禁{row['step']}", ha="center", fontsize=9)
    ax.text(
        0.5,
        -0.28,
        "每一步必须重新求解物理分支并重建二维有效子空间；禁止固定子空间多步复用。",
        transform=ax.transAxes,
        ha="center",
    )
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_rolling_local_rebuild_protocol_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c161 两点磁控 n2 滚动局部分支重建协议

## 实验目的

P2c156 已证明固定二维子空间不能直接用于多步外层迭代；第二步会因接口连续残差超限失败。P2c160 又证明完整三维末端位置灵敏度不可发布。本轮不扩大结论，而是定义下一阶段可验证的滚动局部协议：每一步重新求解当前物理分支，并重新构造二维有效子空间。

## 方法

协议包含五个步骤：当前分支重求解、局部灵敏度重建、二维任务投影、保守电流步生成、扰动重求解闭合。每一步均有硬门禁。该协议只允许二维有效子空间单步推进后重新建模，不允许固定初始二维子空间连续复用。

## 工程边界

- 允许：进入原型验证，检验滚动重建是否修复 P2c156 的第二步失败。
- 禁止：发布完整三维末端位置灵敏度。
- 禁止：发布完整末端位姿灵敏度。
- 禁止：接入默认外层 IK。
- 禁止：超过 0.001 A 半径外推。

## 关键门禁

- 一阶最优性残差原则上不超过 1e-5。
- 中心线分支漂移小于 1e-3 m。
- 局部线性化误差小于 0.05。
- 载荷传播比例优先不超过 1.5。
- 接口连续残差不得超过既有审计阈值。

## 下一步计划

P2c162 应选择一个 n2 工作点，运行两步滚动重建原型，并与 P2c156 固定子空间两步失败直接对照。若第二步仍失败，应归因到模型耦合或分支漂移；若第二步通过，才可扩大到更多工作点和路径。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630173000_p2c161_n2_rolling_local_rebuild_protocol/run_p2c161_n2_rolling_local_rebuild_protocol.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
