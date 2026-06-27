"""Audit n1 load-gate consistency from existing force-closure scans."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C64 = ROOT / "experiment_records_ik_optimize" / "20260627223000_p2c64_n1_force_reg_physical_closure"
P2C65 = ROOT / "experiment_records_ik_optimize" / "20260627231500_p2c65_n1_near_gate_local_refine"


def _read_rows(path: Path, source: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            out: dict[str, Any] = {"source": source}
            for key, value in row.items():
                if value in {"True", "False"}:
                    out[key] = value == "True"
                    continue
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = value
            rows.append(out)
    return rows


def _final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["source"]),
            float(row["force_continuity_scale"]),
            float(row["reg_scale"]),
            float(row["torque_continuity_scale"]),
        )
        if key not in grouped or float(row["stage"]) > float(grouped[key]["stage"]):
            grouped[key] = row
    return list(grouped.values())


def _score(row: dict[str, Any]) -> float:
    terms = [
        max(float(row["stationarity_norm"]) / 1.0e-5 - 1.0, 0.0),
        max(float(row["jacobian_condition"]) / 1.0e7 - 1.0, 0.0),
        max(float(row["continuity_norm"]) / 1.0e-3 - 1.0, 0.0),
        max(float(row["max_force_ratio_to_rigid_load"]) / 1.5 - 1.0, 0.0),
    ]
    return float(math.sqrt(sum(x * x for x in terms)))


def _classify(row: dict[str, Any]) -> dict[str, Any]:
    stationarity = float(row["stationarity_norm"])
    condition = float(row["jacobian_condition"])
    continuity = float(row["continuity_norm"])
    rigid_ratio = float(row["max_force_ratio_to_rigid_load"])
    internal_ratio = float(row["max_force_ratio_to_internal_force"])
    abs_jump = float(row["max_force_jump_abs_N"])
    publish_gate = bool(
        stationarity <= 1.0e-5
        and condition < 1.0e7
        and continuity <= 1.0e-3
        and rigid_ratio <= 1.5
    )
    diagnostic_gate = bool(
        stationarity <= 1.0e-5
        and condition < 1.0e7
        and continuity <= 1.5e-3
        and internal_ratio <= 1.5
        and abs_jump <= 1.5e-3
    )
    failures = []
    if stationarity > 1.0e-5:
        failures.append("一阶最优性退化")
    if condition >= 1.0e9:
        failures.append("条件数严重病态")
    elif condition >= 1.0e7:
        failures.append("条件数偏高")
    if continuity > 1.0e-3:
        failures.append("接口连续误差超门限")
    if rigid_ratio > 2.0:
        failures.append("刚性磁体段载荷传播失配")
    elif rigid_ratio > 1.5:
        failures.append("相对刚段外力偏高")
    if internal_ratio > 1.5:
        failures.append("相对接口内力偏高")
    if abs_jump > 1.5e-3:
        failures.append("绝对力跳变偏高")
    row = dict(row)
    row["publish_gate"] = publish_gate
    row["diagnostic_gate"] = diagnostic_gate
    row["gate_violation_score"] = _score(row)
    row["failure_modes"] = ";".join(failures) if failures else "通过"
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _plot(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (r["source"], r["gate_violation_score"]))
    labels = [
        f"{r['source']}\nF{r['force_continuity_scale']:.0f},R{r['reg_scale']:.0e}"
        for r in rows
    ]
    x = range(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(x, [r["stationarity_norm"] for r in rows], color="#2f5597")
    axes[1].bar(x, [r["jacobian_condition"] for r in rows], color="#70ad47")
    axes[2].bar(x, [r["continuity_norm"] for r in rows], color="#ed7d31")
    axes[3].bar(x, [r["max_force_ratio_to_rigid_load"] for r in rows], color="#a64d79")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(1.0e-3, color="black", linestyle="--", linewidth=1)
    axes[3].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("一阶最优性")
    axes[1].set_ylabel("条件数")
    axes[2].set_ylabel("接口连续")
    axes[3].set_ylabel("相对刚段外力")
    axes[3].set_xticks(list(x))
    axes[3].set_xticklabels(labels, rotation=45, ha="right")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("P2c66 单点载荷门禁一致性审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_load_gate_consistency_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    best = min(rows, key=lambda r: r["gate_violation_score"])
    best_diagnostic = summary.get("best_diagnostic")
    return "\n".join(
        [
            "# P2c66 单点载荷门禁一致性审计记录",
            "",
            "## 实验目的",
            "",
            "基于 P2c64 与 P2c65 已完成结果，复核单点磁控近门样本的载荷传播门禁是否存在定义不一致。该轮不重新求解，只做证据整理与门禁分层。",
            "",
            "## 实验方案",
            "",
            "读取两轮实验的最终阶段表格，重新计算发布门禁和诊断门禁。发布门禁保持原硬标准；诊断门禁允许接口连续误差、绝对力跳变和相对接口内力处于较宽范围，但仍不允许一阶最优性和条件数失败。",
            "",
            "## 关键结果",
            "",
            f"- 纳入最终阶段样本数：`{len(rows)}`",
            f"- 发布门禁通过数：`{summary['publish_pass_count']}`",
            f"- 诊断门禁通过数：`{summary['diagnostic_pass_count']}`",
            f"- 最低违背评分样本：`{best['source']}`，力权重 `{best['force_continuity_scale']:.0f}`，正则 `{best['reg_scale']:.0e}`",
            f"- 该样本一阶最优性：`{best['stationarity_norm']:.3e}`",
            f"- 该样本条件数：`{best['jacobian_condition']:.3e}`",
            f"- 该样本接口连续误差：`{best['continuity_norm']:.3e}`",
            f"- 该样本绝对力跳变：`{best['max_force_jump_abs_N']:.3e} N`",
            f"- 该样本相对刚段外力跳变：`{best['max_force_ratio_to_rigid_load']:.3e}`",
            f"- 该样本相对接口内力跳变：`{best['max_force_ratio_to_internal_force']:.3e}`",
            (
                f"- 最优诊断样本：`{best_diagnostic['source']}`，力权重 `{best_diagnostic['force_continuity_scale']:.0f}`，正则 `{best_diagnostic['reg_scale']:.0e}`，"
                f"接口连续误差 `{best_diagnostic['continuity_norm']:.3e}`，相对刚段外力跳变 `{best_diagnostic['max_force_ratio_to_rigid_load']:.3e}`"
                if best_diagnostic
                else "- 最优诊断样本：无"
            ),
            "",
            "## 结论",
            "",
            "当前单点样本仍没有达到发布级载荷传播门禁。最低发布违背评分样本的一阶最优性和条件数已经可接受，但接口连续误差略高于 `1e-3`，相对刚段外力跳变仍大于 `2`。部分样本满足较宽的诊断门禁，说明它们可用于失败机理分析，但这些样本的相对刚段外力跳变同样偏高，不能作为可发布灵敏度基线。",
            "",
            "该证据支持停止继续做力权重和正则权重局部扫参。下一步应转向模型闭合修复：优先测试把相邻段内力通过刚段传播显式消元，或把接口力连续改写为硬约束参数化，而不是继续用软惩罚约束自由松弛变量。",
            "",
            "## 下一步计划",
            "",
            "1. 设计 P2c67：单点硬载荷传播参数化，使后一柔段初始内力由前一段传播结果决定，只保留必要独立变量。",
            "2. 对同一 `n1_tip_z_rand15, alpha=0.10` 运行最小求解，比较发布门禁、诊断门禁和末端误差。",
            "3. 若硬参数化仍失败，则归因为当前模型/任务组合缺少可发布单点灵敏度证据。",
            "",
        ]
    )


def main() -> None:
    rows = []
    rows.extend(_read_rows(P2C64 / "n1_force_reg_rows.csv", "P2c64"))
    rows.extend(_read_rows(P2C65 / "n1_force_reg_rows.csv", "P2c65"))
    final = [_classify(row) for row in _final_rows(rows)]
    final = sorted(final, key=lambda r: r["gate_violation_score"])
    diagnostic = [row for row in final if row["diagnostic_gate"]]
    summary = {
        "experiment": "20260627224102_p2c66_n1_load_gate_consistency_audit",
        "source_dirs": [str(P2C64), str(P2C65)],
        "final_sample_count": len(final),
        "publish_pass_count": sum(1 for row in final if row["publish_gate"]),
        "diagnostic_pass_count": sum(1 for row in final if row["diagnostic_gate"]),
        "best": final[0],
        "best_diagnostic": min(diagnostic, key=lambda r: r["gate_violation_score"]) if diagnostic else None,
    }
    _write_csv(OUT_DIR / "gate_audit_rows.csv", final)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.csv").write_text(
        "metric,value\n"
        f"final_sample_count,{summary['final_sample_count']}\n"
        f"publish_pass_count,{summary['publish_pass_count']}\n"
        f"diagnostic_pass_count,{summary['diagnostic_pass_count']}\n",
        encoding="utf-8",
    )
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    for row in final:
        if not row["publish_gate"]:
            with (OUT_DIR / "failure_events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"source": row["source"], "failure_modes": row["failure_modes"], "row": row}, ensure_ascii=False) + "\n")
    _plot(final)
    (OUT_DIR / "record.md").write_text(_record(final, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
