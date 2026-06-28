"""Synthesize the n2 branch-conditioned sensitivity evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C71_DIR = ROOT / "experiment_records_ik_optimize" / "20260627233000_p2c71_n2_effective_subspace_audit"
P2C79_DIR = ROOT / "experiment_records_ik_optimize" / "20260628020500_p2c79_branch_random_validation"
P2C80_DIR = ROOT / "experiment_records_ik_optimize" / "20260628024500_p2c80_continuation_branch_validation"
P2C84_DIR = ROOT / "experiment_records_ik_optimize" / "20260628121600_p2c84_block_stationarity_load_closure_audit"


def _set_chinese_font() -> None:
    """Use a local CJK font for plots."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows and convert numeric fields."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build synthesis rows from previous n2 experiments."""
    p2c71 = _read_json(P2C71_DIR / "summary.json")
    p2c79 = _read_json(P2C79_DIR / "summary.json")
    p2c80 = _read_json(P2C80_DIR / "summary.json")
    gate_rows = _read_csv(P2C84_DIR / "gate_rows.csv")
    n2_gate = next(row for row in gate_rows if row["label"] == "n2_trusted_near_gate")
    best_subspace = next(row for row in p2c71["subspace_rows"] if row["removed_weak_directions"] == 2)
    trusted_audits = [row for row in p2c79["audit_rows"] if row["branch"] == "trusted_seed"]
    max_trusted_error = max(row["vector_rel_err"] for row in trusted_audits)
    trusted_pass = sum(bool(row["status"] == "publishable") for row in trusted_audits)
    branch_rows = p2c79["branch_rows"]
    continuation_rows = p2c80["branch_rows"]
    synthesis_rows = [
        {
            "evidence": "基线物理门禁",
            "status": "诊断通过但非全局发布",
            "metric_1": "一阶最优性",
            "value_1": n2_gate["stationarity_norm"],
            "metric_2": "载荷传播比例",
            "value_2": n2_gate["max_interface_force_jump_over_rigid_force"],
            "metric_3": "物理条件数",
            "value_3": n2_gate["jacobian_condition"],
            "interpretation": "物理残差链基本可信，但全空间条件数超过 1e7，不能发布全空间灵敏度。",
        },
        {
            "evidence": "有效子空间",
            "status": "可进入扰动审计",
            "metric_1": "移除弱方向数",
            "value_1": best_subspace["removed_weak_directions"],
            "metric_2": "子空间条件数",
            "value_2": best_subspace["condition"],
            "metric_3": "末端位置秩",
            "value_3": best_subspace["terminal_position_rank"],
            "interpretation": "移除两个最弱内矩方向后，三维末端位置秩仍为 3，说明任务空间不是主要秩亏来源。",
        },
        {
            "evidence": "受信任分支扰动重求解",
            "status": "分支内通过",
            "metric_1": "通过方向数",
            "value_1": trusted_pass,
            "metric_2": "总方向数",
            "value_2": len(trusted_audits),
            "metric_3": "最大线性化误差",
            "value_3": max_trusted_error,
            "interpretation": "在受信任分支内，三维末端位置导数通过多方向扰动重求解。",
        },
        {
            "evidence": "多初值分支复核",
            "status": "未通过",
            "metric_1": "zero_seed 载荷比",
            "value_1": next(row for row in branch_rows if row["branch"] == "zero_seed")["force_ratio"],
            "metric_2": "half_seed 一阶最优性",
            "value_2": next(row for row in branch_rows if row["branch"] == "half_seed")["stationarity_norm"],
            "metric_3": "分支一致性",
            "value_3": 0.0,
            "interpretation": "普通零初值和半初值均不能回到同一物理解，说明初始化敏感和多平衡分支风险仍未解除。",
        },
        {
            "evidence": "电流延拓复核",
            "status": "未通过",
            "metric_1": "zero continuation 载荷比",
            "value_1": next(row for row in continuation_rows if row["branch"] == "continuation_zero_seed")["force_ratio"],
            "metric_2": "half continuation 一阶最优性",
            "value_2": next(row for row in continuation_rows if row["branch"] == "continuation_half_seed")["stationarity_norm"],
            "metric_3": "延拓分支一致性",
            "value_3": 0.0,
            "interpretation": "分段加载不能稳定回到受信任分支，末端位置接近也不能证明物理分支一致。",
        },
    ]
    summary = {
        "p2c71": p2c71,
        "p2c79": p2c79,
        "p2c80": p2c80,
        "n2_gate": n2_gate,
        "trusted_audits": trusted_audits,
        "branch_rows": branch_rows,
        "continuation_rows": continuation_rows,
        "synthesis_rows": synthesis_rows,
    }
    return synthesis_rows, summary


def _plot(summary: dict[str, Any]) -> None:
    """Plot n2 evidence boundaries."""
    _set_chinese_font()
    sci_formatter = mticker.FuncFormatter(lambda value, _pos: f"{value:.0e}" if value > 0 else "0")
    trusted = summary["trusted_audits"]
    branches = summary["branch_rows"]
    cont = summary["continuation_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar([row["direction"] for row in trusted], [row["vector_rel_err"] for row in trusted], color="#2a9d8f")
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("受信任分支线性化误差")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar([row["branch"] for row in branches], [row["force_ratio"] for row in branches], color="#e76f51")
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(sci_formatter)
    axes[1].set_title("多初值载荷传播")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar([row["branch"] for row in cont], [row["stationarity_norm"] for row in cont], color="#457b9d")
    axes[2].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(sci_formatter)
    axes[2].set_title("延拓路径一阶最优性")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c88 两点磁控分支条件证据综合")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_branch_condition_synthesis_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render the Chinese report."""
    table = ["| 证据 | 状态 | 指标1 | 数值1 | 指标2 | 数值2 | 指标3 | 数值3 | 解释 |", "| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- |"]
    for row in rows:
        table.append(
            f"| {row['evidence']} | {row['status']} | {row['metric_1']} | {row['value_1']:.3e} | "
            f"{row['metric_2']} | {row['value_2']:.3e} | {row['metric_3']} | {row['value_3']:.3e} | {row['interpretation']} |"
        )
    trusted = summary["trusted_audits"]
    max_error = max(row["vector_rel_err"] for row in trusted)
    min_error = min(row["vector_rel_err"] for row in trusted)
    gate = summary["n2_gate"]
    return "\n".join(
        [
            "# P2c88 n2 分支条件灵敏度证据综合",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是综合 P2c71、P2c79、P2c80 与 P2c84 的现有证据，明确两点磁控 n2 当前是否已经满足长期目标中的“稳健三维末端位置灵敏度”，并给出下一步最小推进方向。",
            "",
            "## 证据表",
            "",
            *table,
            "",
            "## 判定",
            "",
            "n2 当前尚未达到初始化无关的稳健灵敏度目标，但已经具备一个较强的受信任分支条件结果。该分支同时满足一阶最优性、载荷传播、接口连续和扰动重求解一致性；受信任分支内 9 个方向均通过，线性化误差范围为 "
            f"`{min_error:.3e}` 至 `{max_error:.3e}`。",
            "",
            f"阻断项不是末端三维位置任务本身。P2c71 显示移除两个最弱内矩方向后，子空间条件数为 `{summary['synthesis_rows'][1]['value_2']:.3e}`，末端位置秩仍为 3。当前主要风险是分支可达性：zero、half 与 continuation 路径不能稳定回到 trusted 分支。因此，该结果只能表述为“受信任分支条件下的局部三维末端位置灵敏度候选”，不能表述为一般初值下可复现的稳健灵敏度。",
            "",
            "## 工程边界",
            "",
            "- 可用：受信任分支、position-only 任务、局部小电流扰动、分支条件导数诊断。",
            "- 不可用：零初值、半初值或简单电流延拓得到的一般求解结果；这些路径会落入不同物理解或载荷传播失败分支。",
            "- 暂不发布：完整末端位姿灵敏度、初始化无关三维末端位置灵敏度。",
            "",
            "## 下一步计划",
            "",
            "1. 优先做目标点延拓，而不是继续普通初值扫参；目标是证明 trusted 分支可由连续任务路径稳定到达。",
            "2. 若目标点延拓仍失败，则把 n2 明确归因为多平衡分支竞争，只发布受信任分支条件导数。",
            "3. 若目标点延拓成功，再做 eps=0.005 与 eps=0.010 的扰动重求解复核，并记录完整效率基线。",
            "",
            "## 关键原始门禁",
            "",
            f"- 一阶最优性：`{gate['stationarity_norm']:.3e}`",
            f"- 物理条件数：`{gate['jacobian_condition']:.3e}`",
            f"- 接口连续：`{gate['continuity_norm']:.3e}`",
            f"- 载荷传播比例：`{gate['max_interface_force_jump_over_rigid_force']:.3e}`",
            f"- 末端位置误差：`{gate['position_error_m']:.3e} m`",
            "",
        ]
    )


def main() -> None:
    rows, summary = _build_rows()
    _write_csv(OUT_DIR / "summary.csv", rows)
    payload = {
        "experiment": OUT_DIR.name,
        "status": "trusted_branch_conditioned_candidate_not_global_publishable",
        "main_conclusion": "n2 受信任分支内三维末端位置灵敏度较强，但初始化无关和延拓可达性未通过。",
        "synthesis_rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if "未通过" in row["status"] or "非全局" in row["status"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _plot(summary)
    (OUT_DIR / "record.md").write_text(_record(rows, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
