"""Build n1 nonpublishable boundary and engineering recommendations."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
for path in (ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt

import plotting_fonts


OUT_DIR = Path(__file__).resolve().parent
P2C99 = EXPERIMENT_ROOT / "20260628215500_p2c99_n1_nonpublishable_boundary_synthesis" / "summary.json"
P2C108 = EXPERIMENT_ROOT / "20260629020000_p2c108_sensitivity_boundary_milestone" / "summary.json"
P2C110 = EXPERIMENT_ROOT / "20260629030000_p2c110_n1_hard_force_scaled_continuity" / "summary.json"
P2C111 = EXPERIMENT_ROOT / "20260629033000_p2c111_n1_beta_load_transfer" / "summary.json"


def _read(path: Path) -> dict[str, Any]:
    """Read JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evidence_rows(p2c99: dict[str, Any], p2c108: dict[str, Any], p2c110: dict[str, Any], p2c111: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact evidence rows."""
    p2c110_best = min(p2c110["final_rows"], key=lambda row: row["max_interface_force_jump_over_rigid_force"])
    p2c111_best_pos = min(p2c111["final_rows"], key=lambda row: row["position_error_m"])
    p2c111_best_beta = min(p2c111["final_rows"], key=lambda row: row["beta_deviation_from_one"])
    p2c108_n1 = next(row for row in p2c108["rows"] if row["object_task"] == "n1 有效子空间位置")
    return [
        {
            "source": "P2c99",
            "test": "历史证据综合",
            "position_error_m": None,
            "stationarity_norm": 1.065e-6,
            "condition_number": 8.113e6,
            "force_ratio": 2.769,
            "beta_deviation": None,
            "publishable": False,
            "failure_class": "刚性磁体段载荷传播失配；模型耦合缺失",
        },
        {
            "source": "P2c108",
            "test": "阶段边界",
            "position_error_m": None,
            "stationarity_norm": p2c108_n1["stationarity_norm"],
            "condition_number": p2c108_n1["condition_number"],
            "force_ratio": p2c108_n1["force_ratio"],
            "beta_deviation": None,
            "publishable": False,
            "failure_class": p2c108_n1["failure_class"],
        },
        {
            "source": "P2c110",
            "test": "硬力连续归一化",
            "position_error_m": p2c110_best["position_error_m"],
            "stationarity_norm": p2c110_best["stationarity_norm"],
            "condition_number": p2c110_best["jacobian_condition"],
            "force_ratio": p2c110_best["max_interface_force_jump_over_rigid_force"],
            "beta_deviation": None,
            "publishable": bool(p2c110_best["publish_gate"]),
            "failure_class": p2c110_best["failure_class"],
        },
        {
            "source": "P2c111",
            "test": "beta 最小末端误差",
            "position_error_m": p2c111_best_pos["position_error_m"],
            "stationarity_norm": p2c111_best_pos["stationarity_norm"],
            "condition_number": p2c111_best_pos["jacobian_condition"],
            "force_ratio": p2c111_best_pos["max_interface_force_jump_over_rigid_force"],
            "beta_deviation": p2c111_best_pos["beta_deviation_from_one"],
            "publishable": bool(p2c111_best_pos["publish_gate"]),
            "failure_class": p2c111_best_pos["failure_class"],
        },
        {
            "source": "P2c111",
            "test": "beta 最接近物理闭合",
            "position_error_m": p2c111_best_beta["position_error_m"],
            "stationarity_norm": p2c111_best_beta["stationarity_norm"],
            "condition_number": p2c111_best_beta["jacobian_condition"],
            "force_ratio": p2c111_best_beta["max_interface_force_jump_over_rigid_force"],
            "beta_deviation": p2c111_best_beta["beta_deviation_from_one"],
            "publishable": bool(p2c111_best_beta["publish_gate"]),
            "failure_class": p2c111_best_beta["failure_class"],
        },
    ]


def _recommendation_rows() -> list[dict[str, Any]]:
    """Build engineering recommendation rows."""
    return [
        {
            "topic": "n1 当前工作点",
            "recommendation": "不可发布，不接入外层 IK",
            "reason": "物理闭合与目标拟合不能同时满足硬门禁",
        },
        {
            "topic": "n1 扰动灵敏度扫描",
            "recommendation": "停止",
            "reason": "基线门禁失败，扰动导数没有物理可信基础",
        },
        {
            "topic": "n1 可继续方向",
            "recommendation": "只保留为模型失配与工作空间边界案例",
            "reason": "硬力连续、beta 反事实均未形成可发布分支",
        },
        {
            "topic": "目标选择",
            "recommendation": "后续应选择更保守目标或减小单点磁控任务维度",
            "reason": "当前目标在严格力链闭合下末端误差约 0.46-0.61 mm",
        },
        {
            "topic": "优先主线",
            "recommendation": "转回 n2 分支条件三维位置灵敏度和显式实验接入",
            "reason": "n2 已有门禁证据和契约保护接口候选",
        },
    ]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot n1 boundary evidence."""
    plotting_fonts.configure_matplotlib_cjk()
    labels = [f"{row['source']}\\n{row['test']}" for row in rows]
    force = [row["force_ratio"] if row["force_ratio"] is not None else 0.0 for row in rows]
    pos_um = [row["position_error_m"] * 1e6 if row["position_error_m"] is not None else 0.0 for row in rows]
    cond = [row["condition_number"] if row["condition_number"] is not None else 0.0 for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(labels, force, color="#e76f51")
    axes[0].axhline(1.5, color="black", linestyle="--", label="载荷门禁")
    axes[0].set_ylabel("载荷传播比例")
    axes[0].set_title("载荷闭合")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, pos_um, color="#2a9d8f")
    axes[1].set_ylabel("末端误差/微米")
    axes[1].set_title("末端目标代价")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(labels, cond, color="#457b9d")
    axes[2].axhline(1.0e7, color="black", linestyle="--", label="条件数门禁")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("条件数")
    axes[2].set_title("数值病态")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c112 n1 不可发布边界")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_nonpublishable_boundary_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese boundary report."""
    evidence_table = ["| 来源 | 测试 | 末端误差/m | 一阶最优性 | 条件数 | 载荷比 | beta偏离 | 判定 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["evidence_rows"]:
        pos = f"{row['position_error_m']:.3e}" if row["position_error_m"] is not None else "无"
        sta = f"{row['stationarity_norm']:.3e}" if row["stationarity_norm"] is not None else "无"
        cond = f"{row['condition_number']:.3e}" if row["condition_number"] is not None else "无"
        force = f"{row['force_ratio']:.3e}" if row["force_ratio"] is not None else "无"
        beta = f"{row['beta_deviation']:.3e}" if row["beta_deviation"] is not None else "无"
        evidence_table.append(f"| {row['source']} | {row['test']} | {pos} | {sta} | {cond} | {force} | {beta} | {row['failure_class']} |")
    rec_table = ["| 主题 | 建议 | 原因 |", "| --- | --- | --- |"]
    for row in summary["recommendation_rows"]:
        rec_table.append(f"| {row['topic']} | {row['recommendation']} | {row['reason']} |")
    return "\n".join(
        [
            "# P2c112 n1 不可发布边界与工程建议",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是综合 P2c99、P2c108、P2c110 和 P2c111，为单点磁控 n1 给出可追溯的不可发布边界。目标是停止无效扰动扫描，并明确后续工程策略。",
            "",
            "## 证据表",
            "",
            *evidence_table,
            "",
            "## 判定",
            "",
            "n1 当前工作点不能发布有效子空间末端位置灵敏度。证据不是单一失败，而是三类约束共同指向同一结论：原模型下载荷传播比例约 `2.769` 超门禁；硬力连续可以压低力跳变但使末端误差升至约 `0.6 mm` 且其他门禁不过；beta 反事实若改善末端误差则需要明显放弃刚段外力传递，若保持物理闭合则仍不过一阶最优性和条件数门禁。",
            "",
            "因此，当前 n1 目标应归类为“物理闭合下不可发布或模型缺失边界”，而不是“可通过继续调权重获得的灵敏度候选”。",
            "",
            "## 工程建议",
            "",
            *rec_table,
            "",
            "## 后续规则",
            "",
            "1. n1 在出现新的物理模型前，不再做扰动重求解灵敏度扫描。",
            "2. 若继续研究 n1，应先换更保守目标或降低任务维度，再重新走基线门禁。",
            "3. n1 不能接入外层 IK；相关脚本只能作为诊断和边界分析工具。",
            "4. 当前主线应转回 n2：在分支条件接口候选上做显式外层 IK 接入实验，但默认关闭。",
            "",
            "## 下一步计划",
            "",
            "1. 设计 n2 显式实验开关，调用 P2c107 只读接口候选。",
            "2. 复核该接口在外层 IK 中只作为局部线性建议，不作为默认全局模型。",
            "3. 继续记录 n1 为不可发布边界，除非后续有新物理模型。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    evidence_rows = _evidence_rows(_read(P2C99), _read(P2C108), _read(P2C110), _read(P2C111))
    recommendation_rows = _recommendation_rows()
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in evidence_rows
        if not row["publishable"]
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "evidence_rows": evidence_rows,
        "recommendation_rows": recommendation_rows,
        "decision": "n1_nonpublishable_boundary",
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "recommended_mainline": "n2_guarded_position_sensitivity_explicit_outer_ik_probe",
    }
    _write_csv(OUT_DIR / "evidence_rows.csv", evidence_rows)
    _write_csv(OUT_DIR / "recommendation_rows.csv", recommendation_rows)
    _write_csv(OUT_DIR / "summary.csv", evidence_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(evidence_rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "outer_ik_allowed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
