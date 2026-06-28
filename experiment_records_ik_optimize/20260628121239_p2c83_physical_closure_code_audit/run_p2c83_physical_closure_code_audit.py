"""Generate the P2c83 physical-closure code audit report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent


def _set_chinese_font() -> None:
    """Use an installed CJK font when available."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as CSV."""
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, Any]]:
    """Return curated physical-closure evidence."""
    return [
        {
            "item": "当前刚段传播公式",
            "source": "forward_kinematics/segments_nondim.py:89; inverse_kinematics/ik_terminal_shooting_jax.py:150",
            "finding": "刚段姿态保持常值，外载只更新远端内力和内矩",
            "status": "高风险",
            "implication": "打靶模型中磁体段外载对姿态和曲率的反馈主要通过后续柔段内力链间接体现",
        },
        {
            "item": "磁载荷作用点",
            "source": "forward_kinematics/external_wrench_nondim.py:139; P2c10d",
            "finding": "生产路径以刚段近端状态调用磁载荷；P2c10d 中心作用点原型未改善断裂",
            "status": "已否定简单修复",
            "implication": "单纯把磁载荷移到刚段中心并加入力矩臂不足以恢复可信灵敏度",
        },
        {
            "item": "硬连续 KKT",
            "source": "P2c9d, P2c10r",
            "finding": "直接将接口连续作为硬约束会导致 KKT 条件数约 1e16-1e17，分支不合并",
            "status": "已否定直接路线",
            "implication": "硬约束方向物理合理，但现有变量和尺度不能直接形成可解驻点系统",
        },
        {
            "item": "6维基端载荷连续打靶",
            "source": "P2c10s, P2c67",
            "finding": "自由度压缩后任务和边界难以同时满足",
            "status": "已否定单独路线",
            "implication": "刚性磁体局部载荷不能只由基端载荷一次性表达",
        },
        {
            "item": "自由界面松弛",
            "source": "P2c10t",
            "finding": "几何拟合显著改善，但一阶平稳性、条件数和分支一致性失败",
            "status": "诊断有效",
            "implication": "界面处确有缺失自由度，但自由松弛会生成非唯一内部载荷路径",
        },
        {
            "item": "磁力/磁矩主方向松弛",
            "source": "P2c10u",
            "finding": "限制到合力或合矩方向后拟合能力下降，门禁仍失败",
            "status": "已否定简单方向",
            "implication": "缺失自由度可能位于力臂诱导方向、相邻柔段曲率方向或分块残差冲突方向",
        },
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build machine-readable summary."""
    return {
        "experiment": "P2c83_physical_closure_code_audit",
        "type": "code_and_evidence_synthesis",
        "main_conclusion": "当前问题不应继续归因为单一权重、初值或磁载作用点；更可能是刚性磁体段外载与相邻柔性段边界响应之间的模型闭合不足。",
        "do_not_repeat": [
            "单纯软约束权重扫描",
            "直接硬连续 KKT",
            "纯 6 维基端载荷打靶",
            "自由界面松弛后直接发布灵敏度",
            "只沿磁力或磁矩主方向加入松弛",
        ],
        "next_minimal_experiment": {
            "name": "P2c84_block_stationarity_and_load_closure_audit",
            "goal": "对代表性 n1/n2 近门分支做残差分块梯度贡献和接口载荷平衡逐项审计，判断失败来自尺度冲突、边界约束冲突还是刚段载荷方程缺项。",
            "cases": [
                "n1_tip_z_rand15_alpha_0.10 soft near-gate",
                "n2_z_pair_rand12_alpha_0.10 trusted branch",
            ],
            "required_outputs": [
                "各残差块梯度范数",
                "残差块梯度夹角",
                "刚段近端/中心/远端外载平衡项",
                "接口内力内矩跳变分解",
                "模型修复优先级",
            ],
        },
        "rows": rows,
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot route status."""
    _set_chinese_font()
    score = {
        "高风险": -0.7,
        "已否定简单修复": -1.0,
        "已否定直接路线": -1.0,
        "已否定单独路线": -1.0,
        "诊断有效": 0.5,
        "已否定简单方向": -1.0,
    }
    colors = ["#e76f51" if score[row["status"]] < 0 else "#e9c46a" for row in rows]
    labels = [f"{i + 1}" for i in range(len(rows))]
    values = [score[row["status"]] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylim(-1.25, 0.8)
    ax.set_title("物理闭合候选路线审计")
    ax.set_ylabel("证据状态评分")
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        ax.text(i, values[i] + (0.05 if values[i] >= 0 else -0.05), row["status"], ha="center", va="bottom" if values[i] >= 0 else "top", fontsize=9, rotation=25)
    caption = "\n".join([f"{i + 1}. {row['item']}" for i, row in enumerate(rows)])
    fig.text(0.08, 0.02, caption, fontsize=9, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.34)
    fig.savefig(OUT_DIR / "physical_closure_route_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    """Render Chinese audit report."""
    table = [
        "| 对象 | 证据来源 | 发现 | 状态 | 对灵敏度研究的含义 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(f"| {row['item']} | {row['source']} | {row['finding']} | {row['status']} | {row['implication']} |")
    return "\n".join(
        [
            "# P2c83 物理闭合代码审计与下一步修复入口",
            "",
            "## 实验目的",
            "",
            "P2c81 与 P2c82 已分别给出两点和单点磁控的发布边界。本轮不再继续扰动灵敏度扫描，而是审计刚性磁体段载荷传播相关代码和历史反事实实验，明确下一步模型修复入口。",
            "",
            "## 代码级事实",
            "",
            "当前刚性段传播的核心形式为",
            "",
            "\\[",
            "p(\\sigma)=p_p+\\sigma R_p v^*,\\qquad Q(\\sigma)=Q_p,",
            "\\]",
            "",
            "\\[",
            "f(\\sigma)=f_p-\\frac{\\sigma}{L}F_{ext},",
            "\\]",
            "",
            "\\[",
            "\\tau(\\sigma)=\\tau_p-\\frac{\\sigma}{L}T_{ext}-r(\\sigma)\\times f_p+\\frac{\\sigma}{2L}r(\\sigma)\\times F_{ext}.",
            "\\]",
            "",
            "也就是说，刚段姿态在段内保持常值，外载主要改变传递到下一柔段的内力和内矩。该处理与刚性段不可弯曲的基本假设一致，但对 IK 打靶灵敏度而言，关键问题是外载如何通过接口边界条件稳定地传递给相邻柔性段。",
            "",
            "## 证据综合",
            "",
            *table,
            "",
            "## 当前判断",
            "",
            "当前失败不应继续解释为单一权重、初值或磁载作用点问题。已有实验已经否定了多条简单路线：直接硬连续 KKT、纯 6 维基端载荷、自由界面松弛、磁力/磁矩主方向松弛，以及单纯把磁载荷作用点移到刚段中心。",
            "",
            "更合理的判断是：刚性磁体段外载与相邻柔性段边界响应之间存在模型闭合不足。软约束模型能借助独立分段内力拟合末端；硬闭合模型又难以同时满足任务、边界和一阶平稳性。这正是 P2c81/P2c82 中“末端拟合成功但灵敏度不可发布”的共同根源。",
            "",
            "## 下一步最小实验",
            "",
            "建议下一轮开展 `P2c84_block_stationarity_and_load_closure_audit`。该实验不应先改模型，而应对代表性近门分支做分块最优性和载荷闭合逐项审计：",
            "",
            "1. 对 n1 软约束近门样本和 n2 受信任分支分别计算末端位置、边界、接口连续、正则项的梯度贡献。",
            "2. 计算残差块梯度夹角，判断一阶平稳性失败是尺度问题、目标冲突还是模型不可兼容。",
            "3. 沿每个刚段记录近端、中心、远端的力和力矩平衡项，分解接口内力和内矩跳变。",
            "4. 若边界项与末端项强冲突，优先修正任务/边界尺度；若接口载荷项主导，优先重构刚段到柔段的载荷传播方程；若多块均衡但条件数高，转入近失稳或多平衡分析。",
            "",
            "## 工程建议",
            "",
            "- 不再做简单权重扫参。",
            "- 不再用末端误差小作为进入灵敏度审计的理由。",
            "- 不把自由松弛变量作为发布模型；松弛只能用于诊断缺失自由度。",
            "- 下一步必须先证明哪一个残差块或哪一条力学平衡关系导致门禁失败，再做模型修复。",
            "",
            "## 产物",
            "",
            "- `summary.json`：机器可读审计结论。",
            "- `summary.csv`：路线证据表。",
            "- `failure_events.jsonl`：被否定路线与风险分类。",
            "- `physical_closure_route_audit_cn.png`：中文路线图。",
            "",
        ]
    )


def main() -> None:
    rows = _rows()
    summary = _summary(rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [
        {
            "item": row["item"],
            "status": row["status"],
            "failure_or_risk": row["implication"],
        }
        for row in rows
        if row["status"] != "诊断有效"
    ]
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
