from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

SOURCES = {
    "P2c99": "20260628215500_p2c99_n1_nonpublishable_boundary_synthesis",
    "P2c112": "20260629040000_p2c112_n1_nonpublishable_boundary_recommendation",
    "P2c120": "20260629083000_p2c120_n1_publishability_boundary_package",
    "P2c126": "20260629140000_p2c126_n1_failure_mechanism_repair_map",
    "P2c127": "20260629150000_p2c127_n1_rigid_load_repair_interface_design",
    "P2c128": "20260629160000_p2c128_n1_readonly_rigid_load_observer",
    "P2c129": "20260629170000_p2c129_n1_explicit_rigid_load_residual",
    "P2c130": "20260629190000_p2c130_n1_task_reduction_conservative_workpoint",
    "P2c131": "20260629210000_p2c131_n1_configuration_boundary",
    "P2c132": "20260629230000_p2c132_n1_mid_branch_consistency",
    "P2c138": "20260630060000_p2c138_n1_effective_subspace_boundary_package",
    "P2c139": "20260630063000_p2c139_n1_mid_branch_recovery_probe",
    "P2c141": "20260630070000_p2c141_release_evidence_audit",
}


def load_source(name: str) -> dict:
    path = ROOT / "experiment_records_ik_optimize" / SOURCES[name] / "summary.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
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


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    data = {name: load_source(name) for name in SOURCES}

    p2c131_rows = data["P2c131"].get("final_rows", [])
    p2c132_branch_rows = data["P2c132"].get("branch_rows", [])
    p2c138_rows = data["P2c138"].get("boundary_rows", [])

    baseline_passes = [r for r in p2c131_rows if r.get("baseline_gate")]
    publishable_rows = [r for r in p2c138_rows if r.get("release_class") == "可发布"]
    branch_passes = [r for r in p2c132_branch_rows if r.get("branch_gate")]

    failure_events = [
        {
            "failure_class": "刚性磁体段载荷传播失配",
            "evidence": "P2c120/P2c126/P2c138",
            "interpretation": "单点磁体段的外载荷未能稳定传递到柔性段内力链，导致物理闭合证据不足。",
            "severity": "hard_stop",
        },
        {
            "failure_class": "条件数病态",
            "evidence": "P2c129/P2c130/P2c131/P2c132",
            "interpretation": "多个候选解的一阶线性系统对扰动放大过强，不能支持稳定灵敏度发布。",
            "severity": "hard_stop",
        },
        {
            "failure_class": "初始化敏感与多平衡竞争",
            "evidence": "P2c132/P2c139",
            "interpretation": "同一末端任务可由不同中心线分支拟合，末端误差小不代表局部导数唯一。",
            "severity": "hard_stop",
        },
        {
            "failure_class": "任务降维仍未闭合",
            "evidence": "P2c130",
            "interpretation": "即使退到有效一维方向，保守工作点仍未稳定通过基线门禁。",
            "severity": "hard_stop",
        },
        {
            "failure_class": "证据不足",
            "evidence": "P2c131/P2c139",
            "interpretation": "中段磁体位置出现孤立基线候选，但分支一致性不足，不能作为发布证据。",
            "severity": "diagnostic_only",
        },
    ]

    decision_rows = [
        {
            "scope": "单点磁控完整三维末端位置灵敏度",
            "decision": "不可发布",
            "allowed_use": "失败机制分析、模型修复对照",
            "forbidden_use": "外层 IK、扰动扫描、控制律设计",
            "primary_reason": "分支不唯一与条件数病态",
        },
        {
            "scope": "单点磁控有效子空间末端位置灵敏度",
            "decision": "当前模型不可发布",
            "allowed_use": "诊断性低维候选搜索",
            "forbidden_use": "发布局部导数、默认工程接口",
            "primary_reason": "降维后仍存在基线失败与分支不稳定",
        },
        {
            "scope": "单点磁控中段磁体候选工作点",
            "decision": "仅可诊断",
            "allowed_use": "分支恢复、载荷传播修复验证",
            "forbidden_use": "灵敏度发布、外层迭代",
            "primary_reason": "孤立基线通过，但跨初始化中心线漂移超限",
        },
        {
            "scope": "单点磁控外层 IK 接入",
            "decision": "禁止",
            "allowed_use": "无",
            "forbidden_use": "自动控制、规划、批量扰动求导",
            "primary_reason": "任一硬门禁失败即不得接入外层 IK",
        },
    ]

    summary = {
        "experiment_id": "20260630160000_p2c159_n1_engineering_boundary_synthesis",
        "status": "n1_current_model_not_publishable",
        "n1_effective_subspace_publishable": False,
        "n1_full_3d_position_publishable": False,
        "n1_full_pose_publishable": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "baseline_pass_count_from_p2c131": len(baseline_passes),
        "publishable_row_count_from_p2c138": len(publishable_rows),
        "branch_pair_pass_count_from_p2c132": len(branch_passes),
        "dominant_failure_classes": [
            "刚性磁体段载荷传播失配",
            "条件数病态",
            "初始化敏感与多平衡竞争",
            "任务降维仍未闭合",
            "证据不足",
        ],
        "engineering_boundary": {
            "allowed": [
                "模型失配分析",
                "分支恢复方法验证",
                "刚性磁体段载荷传播模型修复对照",
                "工作空间不可发布边界说明",
            ],
            "forbidden": [
                "发布单点磁控末端位置灵敏度",
                "发布单点磁控完整位姿灵敏度",
                "接入外层 IK",
                "批量扰动扫描作为控制律依据",
            ],
        },
        "source_summaries": {k: v["_source_path"] for k, v in data.items()},
    }

    write_json(EXP_DIR / "summary.json", summary)
    write_csv(EXP_DIR / "engineering_decision_rows.csv", decision_rows)
    write_jsonl(EXP_DIR / "failure_events.jsonl", failure_events)

    configure_chinese_font()
    labels = ["基线候选通过", "可发布条目", "分支对通过"]
    values = [len(baseline_passes), len(publishable_rows), len(branch_passes)]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("数量")
    ax.set_title("单点磁控 n1 工程边界综合")
    ax.set_ylim(0, max(values + [1]) + 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")
    ax.text(
        0.5,
        -0.22,
        "结论：当前模型下无可发布 n1 灵敏度；中段孤立候选仅可用于诊断。",
        ha="center",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n1_engineering_boundary_synthesis_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c159 单点磁控 n1 工程边界综合

## 实验目的

本轮不新增求解器实验，而是对既有单点磁控 n1 证据链进行工程边界综合。目标是判断当前模型是否已经支持可发布的末端位置灵敏度，或是否应明确标记为不可发布边界。

## 证据来源

本报告读取 `{len(SOURCES)}` 组既有实验摘要。核心证据包括 P2c120 的不可发布边界包、P2c126 的失败机制图谱、P2c129 的显式刚性载荷残差实验、P2c130 的任务降维实验、P2c131 的配置边界搜索、P2c132 与 P2c139 的分支一致性复核，以及 P2c141 的阶段性发布审计。

## 主要结果

- P2c131 中仅有 `{len(baseline_passes)}` 个孤立基线候选通过。
- P2c138 中可发布条目数为 `{len(publishable_rows)}`。
- P2c132 中跨初始化分支对通过数为 `{len(branch_passes)}`。
- P2c139 表明延续路径可以形成分支条件下的一致解，但与直接参考路径仍存在明显中心线分支差异，因此不能视为全局唯一灵敏度。

## 失败机制

1. 刚性磁体段载荷传播失配仍是主导问题。磁体段外载荷未能稳定进入柔性段内力链，导致物理闭合证据不足。
2. 条件数病态在多组候选中反复出现。即使末端位置误差较小，扰动导数仍可能被病态线性系统放大。
3. 初始化敏感和多平衡竞争仍未消除。同一末端位置可由不同中心线分支拟合，说明末端拟合不能证明灵敏度唯一。
4. 任务降维未能提供稳定发布口径。有效一维方向也未稳定通过基线门禁。

## 工程结论

当前模型下，单点磁控 n1 不能发布末端位置灵敏度，不能发布完整末端位姿灵敏度，也不能接入外层 IK。中段磁体候选工作点只能作为模型修复和分支恢复诊断对象，不能作为控制接口。

## 下一步计划

优先做刚性磁体段载荷传播模型修复，明确磁体段外力与柔性段内力链的耦合方式。修复前不再扩大 n1 扰动扫描。若继续 n1，应只运行最小基线门禁与分支复核；若推进工程应用，应优先使用当前已有两点磁控 n2 的有效二维子空间诊断接口。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630160000_p2c159_n1_engineering_boundary_synthesis/run_p2c159_n1_engineering_boundary_synthesis.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
