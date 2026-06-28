"""Generate the P2c82 n1 effective-subspace boundary report."""

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
            break
    plt.rcParams["axes.unicode_minus"] = False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a CSV file."""
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, Any]]:
    """Return curated n1 evidence from P2c56 and P2c64-P2c68."""
    return [
        {
            "stage": "P2c56",
            "claim": "单点近似响应呈二维有效子空间",
            "gate": "原始基线门禁",
            "value": "rank=2; stationarity=2.447; condition=4.495e9; force_ratio=17.71",
            "status": "不可发布",
            "failure_class": "一阶最优性退化、条件数病态、载荷传播失配",
            "evidence": "二维结构存在，但物理基线严重失败",
        },
        {
            "stage": "P2c64",
            "claim": "力连续与正则可显著缓解数值病态",
            "gate": "软约束局部扫描",
            "value": "best stationarity=7.955e-7; condition=7.070e6; force_ratio=2.777",
            "status": "近门诊断",
            "failure_class": "接口力连续性失配",
            "evidence": "一阶最优性和条件数过门，但载荷传播仍失败",
        },
        {
            "stage": "P2c65",
            "claim": "局部调参不能突破载荷传播瓶颈",
            "gate": "近门局部细化",
            "value": "best stationarity=1.065e-6; condition=8.113e6; force_ratio=2.769",
            "status": "近门诊断",
            "failure_class": "接口力连续性失配",
            "evidence": "多组样本停留在 force_ratio 2.73-2.78",
        },
        {
            "stage": "P2c66",
            "claim": "载荷门禁定义一致，失败不是统计口径错误",
            "gate": "门禁一致性审计",
            "value": "publish 0/18; diagnostic 4/18; best force_ratio=2.734",
            "status": "诊断通过",
            "failure_class": "载荷传播失配",
            "evidence": "发布门禁无样本通过，宽松诊断门禁仅供归因",
        },
        {
            "stage": "P2c67",
            "claim": "硬载荷传播闭合后任务和边界难以相容",
            "gate": "硬物理闭合参数化",
            "value": "best tip_error=5.93e-3; stationarity=1.32e-3; condition=1.212e9",
            "status": "不可发布",
            "failure_class": "硬闭合下任务不可达、一阶最优性退化",
            "evidence": "消除自由载荷跳变后末端误差和边界残差明显退化",
        },
        {
            "stage": "P2c68",
            "claim": "单点当前模型不应进入扰动灵敏度审计",
            "gate": "阶段综合",
            "value": "current_model_not_publishable",
            "status": "不可发布",
            "failure_class": "模型耦合缺失或有效物理自由度不足",
            "evidence": "软约束近门依赖独立分段内力，硬闭合后任务不可兼容",
        },
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build machine-readable summary."""
    return {
        "experiment": "P2c82_n1_effective_subspace_boundary",
        "source_range": "P2c56,P2c64-P2c68",
        "case": "n1_tip_z_rand15_alpha_0.10_position_only",
        "current_publish_status": "not_publishable_under_current_model",
        "diagnostic_scope": [
            "二维有效子空间是诊断线索",
            "软约束近门样本可用于载荷传播失败归因",
        ],
        "not_publishable_scope": [
            "单点有效子空间末端位置灵敏度",
            "单点完整三维末端位置灵敏度",
            "外层 IK 默认灵敏度",
        ],
        "hard_gate_status": {
            "stationarity": "soft near-gate pass; hard-closure fail",
            "condition": "soft near-gate pass; hard-closure fail",
            "load_propagation": "fail under soft near-gate",
            "task_reachability": "fail or degrade under hard load propagation",
            "local_linearity": "not audited because baseline gates fail",
        },
        "main_failure_mechanism": [
            "载荷传播失配",
            "硬物理闭合后任务不可达",
            "独立分段内力自由度可能替代真实载荷链",
            "有效物理自由度不足或模型耦合缺失",
        ],
        "recommended_next_work": [
            "停止 n1 扰动灵敏度扫描，先修复物理闭合模型",
            "优先审查刚性磁体段载荷传播公式、边界残差尺度和接口约束定义",
            "若模型修复后仍失败，可形成当前单点配置不可发布结论",
        ],
        "rows": rows,
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot n1 boundary status."""
    _set_chinese_font()
    score = {
        "诊断通过": 1.0,
        "近门诊断": 0.5,
        "不可发布": -1.0,
    }
    colors = {"诊断通过": "#e9c46a", "近门诊断": "#f4a261", "不可发布": "#e76f51"}
    labels = [row["stage"] for row in rows]
    values = [score[row["status"]] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(labels, values, color=[colors[row["status"]] for row in rows])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("n1 单点磁控有效子空间证据边界")
    ax.set_ylabel("证据状态评分")
    ax.set_ylim(-1.35, 1.25)
    ax.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        ax.text(i, values[i] + (0.06 if values[i] >= 0 else -0.06), row["status"], ha="center", va="bottom" if values[i] >= 0 else "top", fontsize=9, rotation=20)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.22)
    fig.savefig(OUT_DIR / "n1_effective_subspace_boundary_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    """Render the Chinese report."""
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
            "# P2c82 n1 单点有效子空间发布边界报告",
            "",
            "## 实验目的",
            "",
            "本轮综合 P2c56 与 P2c64-P2c68 的单点磁控证据，判断 `n1_tip_z_rand15, alpha=0.10` 是否已经具备有效子空间末端位置灵敏度发布条件。本轮不重新求解，只做门禁归纳和工程边界判定。",
            "",
            "## 门禁综合",
            "",
            *table,
            "",
            "## 当前判定",
            "",
            "单点磁控当前不能发布有效子空间末端位置灵敏度。P2c56 显示近似响应具有二维有效秩，但原始物理基线的一阶最优性、条件数和载荷传播均严重失败。P2c64-P2c65 通过力连续与正则项把一阶最优性和条件数推进到近门或过门水平，但接口连续残差和相对刚段外力跳变仍稳定停留在失败区间。",
            "",
            "P2c66 说明这不是门禁统计口径错误。P2c67 进一步表明，若把刚性段载荷传播写成硬物理闭合，任务位置误差、终端边界残差、一阶最优性和条件数都会明显退化。这说明软约束近门解很可能依赖独立分段内力自由度，而不是由稳定物理载荷链唯一确定。",
            "",
            "## 可用与不可用范围",
            "",
            "- 可用：二维有效子空间可作为诊断线索，用于判断单点磁控存在约两个主要可控方向。",
            "- 可用：软约束近门样本可用于分析载荷传播失配和物理闭合缺口。",
            "- 不可用：不能发布单点有效子空间灵敏度。",
            "- 不可用：不能接入外层 IK 默认灵敏度。",
            "- 不可用：不能把末端位置误差小解释为物理可信解。",
            "",
            "## 失败机制",
            "",
            "当前主导失败机制是载荷传播失配与硬物理闭合下任务不可兼容。数学上，软约束模型增加了独立分段内力自由度，使末端位置残差可以被拟合；但这些自由度没有被刚性磁体段的真实载荷传播充分约束，因此无法保证局部灵敏度对应同一个物理解分支。",
            "",
            "若后续要推进单点磁控，应先修正模型闭合，而不是直接进入扰动重求解。优先检查对象包括：刚性磁体段外力和外矩传播公式、终端边界残差尺度、接口载荷连续的无量纲化、以及分段内力变量是否存在冗余自由度。",
            "",
            "## 下一步计划",
            "",
            "1. 若继续 n1，应设计模型修复实验，而非继续权重扫参。",
            "2. 最小修复方向：重新推导刚性磁体段载荷传播与接口平衡方程，并在相同工况下复测 P2c64 的近门样本。",
            "3. 若模型修复后仍不能同时满足任务、边界和载荷传播门禁，则形成该单点配置当前模型下不可发布结论。",
            "",
            "## 产物",
            "",
            "- `summary.json`：机器可读边界结论。",
            "- `summary.csv`：门禁证据表。",
            "- `failure_events.jsonl`：失败分类。",
            "- `n1_effective_subspace_boundary_cn.png`：中文证据链图。",
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
            "stage": row["stage"],
            "failure_class": row["failure_class"],
            "evidence": row["evidence"],
        }
        for row in rows
        if row["status"] != "诊断通过"
    ]
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
