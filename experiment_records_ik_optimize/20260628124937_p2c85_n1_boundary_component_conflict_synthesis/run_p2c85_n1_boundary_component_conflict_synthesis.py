"""Generate the P2c85 n1 boundary-component conflict synthesis."""

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


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C84_DIR = ROOT / "experiment_records_ik_optimize" / "20260628121600_p2c84_block_stationarity_load_closure_audit"


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


def _read_p2c84_rows() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the current near-gate n1 rows from P2c84."""
    with (P2C84_DIR / "gate_rows.csv").open(encoding="utf-8") as f:
        gate_rows = list(csv.DictReader(f))
    with (P2C84_DIR / "block_gradient_rows.csv").open(encoding="utf-8") as f:
        block_rows = list(csv.DictReader(f))
    with (P2C84_DIR / "block_gradient_cosines.csv").open(encoding="utf-8") as f:
        cosine_rows = list(csv.DictReader(f))
    for row in gate_rows + block_rows + cosine_rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    n1_gate = next(row for row in gate_rows if row["label"] == "n1_soft_near_gate")
    n1_blocks = [row for row in block_rows if row["label"] == "n1_soft_near_gate"]
    n1_cosines = [row for row in cosine_rows if row["label"] == "n1_soft_near_gate"]
    return n1_gate, n1_blocks, n1_cosines


def _evidence_rows(n1_gate: dict[str, Any], n1_blocks: list[dict[str, Any]], n1_cosines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build synthesis evidence rows."""
    strongest_conflict = min(n1_cosines, key=lambda row: row["gradient_cosine"])
    dominant_block = max(n1_blocks, key=lambda row: row["gradient_norm"])
    return [
        {
            "source": "P2c43",
            "evidence": "边界权重 0.02-1.00 均未产生通过点，n1 最优权重仍为 0.25",
            "interpretation": "失败不是单纯边界权重误设",
            "status": "排除简单权重修复",
        },
        {
            "source": "P2c44",
            "evidence": "无边界、仅力、仅矩、力矩全闭合均未过门；仅矩和全闭合降低内矩残差但载荷比仍约 2.49",
            "interpretation": "失败不是删除某个边界分量即可解决",
            "status": "排除边界删减修复",
        },
        {
            "source": "P2c84",
            "evidence": f"n1 一阶最优性 {n1_gate['stationarity_norm']:.3e}、条件数 {n1_gate['jacobian_condition']:.3e} 已过门，但载荷比 {n1_gate['max_interface_force_jump_over_rigid_force']:.3e} 失败",
            "interpretation": "当前 n1 已不是整体数值不可用，而是载荷传播闭合失败",
            "status": "定位载荷传播门禁",
        },
        {
            "source": "P2c84",
            "evidence": f"最大梯度贡献来自 {dominant_block['block']}，贡献 {dominant_block['gradient_norm']:.3e}",
            "interpretation": "末端自由边界仍是近门分支一阶结构中的主要因素",
            "status": "定位边界贡献",
        },
        {
            "source": "P2c84",
            "evidence": f"最强梯度冲突为 {strongest_conflict['block_a']} 与 {strongest_conflict['block_b']}，夹角余弦 {strongest_conflict['gradient_cosine']:.3e}",
            "interpretation": "任务位置与自由边界存在直接相容性张力",
            "status": "定位冲突方向",
        },
    ]


def _summary(n1_gate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build machine-readable summary."""
    return {
        "experiment": "P2c85_n1_boundary_component_conflict_synthesis",
        "case": "n1_tip_z_rand15_alpha_0.10",
        "current_status": "not_publishable_but_near_gate",
        "gate_status": {
            "stationarity": n1_gate["stationarity_norm"],
            "condition": n1_gate["jacobian_condition"],
            "continuity": n1_gate["continuity_norm"],
            "load_ratio": n1_gate["max_interface_force_jump_over_rigid_force"],
            "position_error_m": n1_gate["position_error_m"],
        },
        "main_conclusion": "n1 当前最小修复不应是继续调边界权重或删除边界分量，而应处理末端位置、自由边界和刚段接口载荷传播之间的相容性。",
        "next_minimal_experiment": "P2c86_n1_terminal_boundary_relaxation_vs_load_balance",
        "recommended_actions": [
            "保留 position-only 成功目标，但把边界闭合从硬指标改为诊断变量进行反事实比较",
            "分别测试末端边界弱化、接口载荷重标定、边界-接口联合白化三种最小修复",
            "所有修复仍用原物理门禁评价，不因边界放松直接发布灵敏度",
        ],
        "rows": rows,
    }


def _plot(n1_gate: dict[str, Any], n1_blocks: list[dict[str, Any]], n1_cosines: list[dict[str, Any]]) -> None:
    """Plot current n1 conflict evidence."""
    _set_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    gate_labels = ["一阶", "条件", "连续", "载荷比"]
    gate_values = [
        n1_gate["stationarity_norm"] / 1.0e-5,
        n1_gate["jacobian_condition"] / 1.0e7,
        n1_gate["continuity_norm"] / 1.0e-3,
        n1_gate["max_interface_force_jump_over_rigid_force"] / 1.5,
    ]
    axes[0].bar(gate_labels, gate_values, color=["#2a9d8f" if value <= 1 else "#e76f51" for value in gate_values])
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("门禁倍率")
    axes[0].grid(axis="y", alpha=0.25)
    block_labels = [row["block"] for row in n1_blocks]
    block_values = [row["gradient_norm"] for row in n1_blocks]
    axes[1].bar(block_labels, block_values, color="#457b9d")
    axes[1].set_yscale("log")
    axes[1].set_title("残差块一阶贡献")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    pair_labels = [f"{row['block_a']}\n{row['block_b']}" for row in n1_cosines]
    pair_values = [row["gradient_cosine"] for row in n1_cosines]
    axes[2].bar(pair_labels, pair_values, color=["#e76f51" if value < -0.5 else "#e9c46a" for value in pair_values])
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_title("残差块梯度夹角余弦")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c85 n1 边界分量冲突综合")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_boundary_component_conflict_cn.png", dpi=180)
    plt.close(fig)


def _record(n1_gate: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Render Chinese report."""
    table = [
        "| 来源 | 证据 | 解释 | 状态 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(f"| {row['source']} | {row['evidence']} | {row['interpretation']} | {row['status']} |")
    return "\n".join(
        [
            "# P2c85 n1 边界分量冲突综合记录",
            "",
            "## 实验目的",
            "",
            "P2c84 显示单点 n1 近门样本的一阶最优性和条件数已过门，但载荷传播失败，且末端位置与末端边界梯度明显冲突。本轮综合 P2c43、P2c44 与 P2c84，判断下一步是否应继续调边界权重或删除边界分量。",
            "",
            "## 当前门禁",
            "",
            f"- 一阶最优性：`{n1_gate['stationarity_norm']:.3e}`",
            f"- 条件数：`{n1_gate['jacobian_condition']:.3e}`",
            f"- 接口连续：`{n1_gate['continuity_norm']:.3e}`",
            f"- 载荷比：`{n1_gate['max_interface_force_jump_over_rigid_force']:.3e}`",
            f"- 末端位置误差：`{n1_gate['position_error_m']:.3e} m`",
            "",
            "## 证据综合",
            "",
            *table,
            "",
            "## 结论",
            "",
            "n1 当前不是缺少可控方向，也不是简单边界权重误设。已有证据显示：降低或拆分末端边界不能使门禁通过；近门分支的末端位置和自由边界之间存在直接梯度冲突；同时接口载荷传播仍明显超限。因此，继续做边界权重扫描或删除边界分量，不能形成可发布灵敏度。",
            "",
            "下一步最小修复应比较三类反事实：末端边界弱化、接口载荷重标定、边界-接口联合白化。注意，这些只能用于定位模型闭合缺口；即使某个反事实降低一阶冲突，也必须回到原物理门禁评价后才能讨论灵敏度发布。",
            "",
            "## 下一步计划",
            "",
            "1. 做 `P2c86_n1_terminal_boundary_relaxation_vs_load_balance`，只覆盖 n1 近门工况。",
            "2. 对比原模型、边界弱化、接口载荷重标定、边界-接口联合白化四组。",
            "3. 若某组降低冲突但原物理门禁仍失败，记为模型修复线索，不进入扰动灵敏度。",
            "",
        ]
    )


def main() -> None:
    n1_gate, n1_blocks, n1_cosines = _read_p2c84_rows()
    rows = _evidence_rows(n1_gate, n1_blocks, n1_cosines)
    summary = _summary(n1_gate, rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if row["status"].startswith("排除"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _plot(n1_gate, n1_blocks, n1_cosines)
    (OUT_DIR / "record.md").write_text(_record(n1_gate, rows), encoding="utf-8")


if __name__ == "__main__":
    main()
