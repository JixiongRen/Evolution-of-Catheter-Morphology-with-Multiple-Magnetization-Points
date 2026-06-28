"""Generate the P2c81 n2 engineering-boundary synthesis report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def _set_chinese_font() -> None:
    """Use an installed CJK font when available."""
    font_files = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for font_file in font_files:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return
    candidates = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a CSV file."""
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gate_rows() -> list[dict[str, Any]]:
    """Return manually curated gate evidence from P2c72-P2c80."""
    return [
        {
            "stage": "P2c72",
            "claim": "16维有效状态子空间可稳定重求解",
            "gate": "扰动重求解",
            "value": "3/3 diagnostic, 0/3 publish",
            "status": "诊断通过",
            "failure_class": "局部线性化未达发布门禁",
            "evidence": "relative error 0.106-0.196 > 0.05",
        },
        {
            "stage": "P2c73",
            "claim": "误差不是扰动尺度过大导致",
            "gate": "扰动尺度平台",
            "value": "eps=0.0025/0.005/0.010",
            "status": "诊断通过",
            "failure_class": "隐式灵敏度公式不一致",
            "evidence": "relative error nearly flat across eps",
        },
        {
            "stage": "P2c74",
            "claim": "基础自动微分分量基本正确",
            "gate": "导数分量有限差分",
            "value": "residual dI <=1.6e-11; tip dq pass",
            "status": "通过",
            "failure_class": "非分量错误",
            "evidence": "residual-current and tip-state derivatives pass",
        },
        {
            "stage": "P2c75",
            "claim": "误差来自状态增量方程",
            "gate": "实际状态增量对照",
            "value": "state error 0.0275-0.0988",
            "status": "诊断通过",
            "failure_class": "状态灵敏度方程缺项",
            "evidence": "actual dq explains tip FD, old implicit dq does not",
        },
        {
            "stage": "P2c76",
            "claim": "完整一阶最优性灵敏度修复线性化",
            "gate": "扰动重求解一致性",
            "value": "tip error 5e-5-5.8e-4; condition 2.432e13",
            "status": "不可发布",
            "failure_class": "条件数病态",
            "evidence": "linearity passes but condition exceeds 1e9",
        },
        {
            "stage": "P2c77",
            "claim": "简单截断或阻尼不能稳定化",
            "gate": "稳定化审计",
            "value": "0/54 pass",
            "status": "不可发布",
            "failure_class": "病态方向承载真实任务灵敏度",
            "evidence": "regularization lowers condition but destroys tip derivative",
        },
        {
            "stage": "P2c78",
            "claim": "末端三维位置任务空间可线性化",
            "gate": "任务空间模态扰动",
            "value": "3/3 modes pass; task condition 322.8",
            "status": "候选通过",
            "failure_class": "第三方向弱可控",
            "evidence": "singular values 1.516e-06, 8.595e-08, 4.697e-09",
        },
        {
            "stage": "P2c79",
            "claim": "受信任分支内随机方向稳定",
            "gate": "分支与随机方向",
            "value": "trusted 9/9 pass; zero/half fail",
            "status": "分支条件通过",
            "failure_class": "初始化敏感与多平衡竞争",
            "evidence": "zero/half branch sensitivity differs by 56.98 and 928.5",
        },
        {
            "stage": "P2c80",
            "claim": "电流分段加载不能恢复可信分支",
            "gate": "continuation 分支复核",
            "value": "zero/half continuation both fail",
            "status": "不可发布为初始化无关结果",
            "failure_class": "初始化敏感与物理分支不可达",
            "evidence": "half terminal close but sensitivity differs by 15.57",
        },
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the machine-readable synthesis summary."""
    return {
        "experiment": "P2c81_n2_engineering_boundary_synthesis",
        "source_range": "P2c72-P2c80",
        "case": "n2_z_pair_rand12_alpha_0.10_force_4_reg_3e-4",
        "current_publish_status": "branch_conditional_local_terminal_position_sensitivity_only",
        "publishable_scope": [
            "受信任分支内的局部三维末端位置灵敏度",
            "第三任务方向应标注为弱可控方向",
        ],
        "not_publishable_scope": [
            "初始化无关灵敏度",
            "零初值或半初值可达的稳健灵敏度",
            "完整末端位姿灵敏度",
            "外层 IK 生产接入",
        ],
        "hard_gate_status": {
            "stationarity": "trusted branch pass; alternative branches fail",
            "condition": "task-space pass; internal optimality system fail",
            "branch_consistency": "fail outside trusted branch",
            "local_linearity": "pass inside trusted task-space modes",
            "load_propagation": "trusted branch pass; zero/half fail",
            "perturb_reoptimization": "pass inside trusted branch",
        },
        "main_failure_mechanism": [
            "初始化敏感",
            "多平衡分支竞争",
            "内部最优性线性系统病态",
            "弱可控第三任务方向",
        ],
        "recommended_next_work": [
            "先形成受信任分支工程边界和使用条件",
            "若要扩大适用范围，优先做目标点延拓或更细分段加载，而不是继续无约束扫初值",
            "单点磁控应独立做有效子空间末端位置灵敏度，不应套用 n2 结论",
        ],
        "rows": rows,
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot the evidence chain status."""
    _set_chinese_font()
    status_score = {
        "通过": 2.0,
        "候选通过": 1.6,
        "分支条件通过": 1.2,
        "诊断通过": 0.6,
        "不可发布": -1.0,
        "不可发布为初始化无关结果": -1.0,
    }
    colors = []
    scores = []
    labels = []
    display_status = {
        "不可发布为初始化无关结果": "不可发布\n初始化相关",
        "分支条件通过": "分支条件\n通过",
    }
    for row in rows:
        score = status_score[row["status"]]
        scores.append(score)
        labels.append(row["stage"])
        colors.append("#2a9d8f" if score > 1 else "#e9c46a" if score > 0 else "#e76f51")
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(labels, scores, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("证据状态评分")
    ax.set_title("n2 两点磁控灵敏度证据链边界")
    ax.set_ylim(-1.4, 2.3)
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        label = display_status.get(row["status"], row["status"])
        ax.text(i, scores[i] + (0.08 if scores[i] >= 0 else -0.08), label, ha="center", va="bottom" if scores[i] >= 0 else "top", fontsize=8, rotation=25)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.26)
    fig.savefig(OUT_DIR / "n2_evidence_boundary_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render the Chinese experiment record."""
    table = [
        "| 阶段 | 核心判断 | 门禁 | 结果 | 状态 | 失败或风险分类 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['stage']} | {row['claim']} | {row['gate']} | {row['value']} | {row['status']} | {row['failure_class']} |"
        )
    return "\n".join(
        [
            "# P2c81 n2 工程边界与可信分支综合报告",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解导管平衡方程，而是对 P2c72-P2c80 已提交证据进行边界综合。目标是明确两点磁控当前能发布什么、不能发布什么，以及后续应当优先修复哪类问题。",
            "",
            "## 证据来源",
            "",
            "- P2c72-P2c73：有效状态子空间扰动重求解和扰动尺度平台审计。",
            "- P2c74-P2c77：导数分量、状态增量、完整一阶最优性灵敏度和稳定化审计。",
            "- P2c78-P2c80：末端任务空间降维、多初值分支复核和电流分段加载复核。",
            "",
            "## 门禁综合",
            "",
            *table,
            "",
            "## 当前可发布口径",
            "",
            "当前只支持非常限定的表述：在 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4` 工况下，受信任分支内的局部三维末端位置灵敏度通过任务空间线性化与扰动重求解复核。第三个任务方向奇异值很小，应解释为弱可控方向。",
            "",
            "该结论不能扩展为初始化无关的稳健灵敏度。零初值、半初值以及当前电流分段加载均不能稳定回到同一物理解分支。末端位置接近也不能替代一阶最优性、载荷传播和灵敏度一致性门禁。",
            "",
            "## 不可发布范围",
            "",
            "- 不能发布完整末端位姿灵敏度。",
            "- 不能发布零初值或半初值可达的稳健灵敏度。",
            "- 不能接入外层 IK 作为默认生产灵敏度。",
            "- 不能把内部最优性系统的原始状态灵敏度解释为稳定物理量；只能解释任务空间投影后的局部导数。",
            "",
            "## 失败机制",
            "",
            "主要失败机制不是末端拟合误差，而是初始化敏感和多平衡分支竞争。P2c80 中半初值路径末端位置已接近受信任分支，但一阶最优性和灵敏度差异仍严重失控，说明当前模型存在多个几何相近但物理载荷链不同的局部解。",
            "",
            "内部完整一阶最优性线性系统条件数约 `2.432e+13`，说明状态层面存在严重病态。P2c78 表明末端三维位置任务空间本身条件数约 `3.228e+02`，因此当前较合理的研究对象是任务空间局部导数，而不是完整状态导数。",
            "",
            "## 工程建议",
            "",
            "1. 两点磁控近期只推进三维末端位置任务；完整姿态只保留诊断。",
            "2. 发布前必须写明“受信任分支条件”。若需要工程使用，应增加目标点延拓或更细电流加载，使至少两条合理路径回到同一物理解分支。",
            "3. 若分支一致性仍失败，应转入模型闭合修复，重点检查刚性磁体段载荷传播、接口约束和冗余自由变量是否替代了真实物理载荷链。",
            "4. 单点磁控不能直接套用本结论，应独立做有效子空间末端位置灵敏度审计。",
            "",
            "## 下一步计划",
            "",
            "下一轮优先建议做 `P2c82_n1_effective_position_subspace_boundary`，把单点磁控按同样门禁体系重做有效子空间审计。若继续两点磁控，则应做目标点延拓而不是继续普通初值扫参。",
            "",
            "## 产物",
            "",
            "- `summary.json`：机器可读边界结论。",
            "- `summary.csv`：门禁证据表。",
            "- `failure_events.jsonl`：失败分类。",
            "- `n2_evidence_boundary_cn.png`：中文证据链图。",
            "",
        ]
    )


def main() -> None:
    rows = _gate_rows()
    summary = _summary(rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [
        {
            "stage": row["stage"],
            "failure_class": row["failure_class"],
            "evidence": row["evidence"],
        }
        for row in rows
        if row["status"] not in {"通过", "候选通过", "分支条件通过"}
    ]
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
