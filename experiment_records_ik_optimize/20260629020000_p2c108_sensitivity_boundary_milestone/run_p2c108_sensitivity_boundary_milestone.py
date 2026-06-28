"""Build the current IK sensitivity boundary milestone report."""

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
P2C100 = EXPERIMENT_ROOT / "20260628222500_p2c100_ik_sensitivity_milestone_synthesis" / "summary.json"
P2C107 = EXPERIMENT_ROOT / "20260629013000_p2c107_n2_guarded_interface_candidate" / "summary.json"


def _read(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot the current publishability boundary."""
    plotting_fonts.configure_matplotlib_cjk()
    labels = [row["object_task"] for row in rows]
    publishable = [1 if row["decision_level"] == "可作为分支条件候选" else 0 for row in rows]
    force_margin = [row["force_ratio"] / 1.5 if row["force_ratio"] is not None else 0.0 for row in rows]
    linear_margin = [
        row["linearization_rel_err"] / 0.05 if row["linearization_rel_err"] is not None else 0.0
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#2a9d8f" if value else "#e76f51" for value in publishable]
    axes[0].bar(labels, publishable, color=colors)
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].set_ylabel("是否可作为候选")
    axes[0].set_title("当前发布边界")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, force_margin, color="#457b9d")
    axes[1].axhline(1.0, color="#e76f51", linestyle="--", label="门禁")
    axes[1].set_ylabel("载荷传播比例/门禁")
    axes[1].set_title("物理闭合门禁")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(labels, linear_margin, color="#2a9d8f")
    axes[2].axhline(1.0, color="#e76f51", linestyle="--", label="门禁")
    axes[2].set_ylabel("线性化误差/门禁")
    axes[2].set_title("局部线性化门禁")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c108 IK 灵敏度阶段边界")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ik_sensitivity_boundary_milestone_cn.png", dpi=180)
    plt.close(fig)


def _build_rows(p2c99: dict[str, Any], p2c100: dict[str, Any], p2c107: dict[str, Any]) -> list[dict[str, Any]]:
    """Build current decision rows."""
    n2_ev = p2c107["snapshot"]["evidence"]
    n1_row = {
        "object_task": "n1 有效子空间位置",
        "magnet_count": 1,
        "task_dimension": "有效子空间",
        "decision_level": "不可发布",
        "stationarity_norm": 1.065e-6,
        "condition_number": 8.113e6,
        "force_ratio": 2.769,
        "linearization_rel_err": None,
        "branch_evidence": "诊断可用，发布不足",
        "failure_class": "刚性磁体段载荷传播失配；模型耦合缺失",
        "allowed_use": "诊断与模型修复",
        "forbidden_use": "发布灵敏度；外层 IK 默认灵敏度",
        "source": "P2c99",
    }
    n2_row = {
        "object_task": "n2 三维末端位置",
        "magnet_count": 2,
        "task_dimension": 3,
        "decision_level": "可作为分支条件候选",
        "stationarity_norm": n2_ev["stationarity_norm"],
        "condition_number": n2_ev["task_condition"],
        "force_ratio": n2_ev["force_ratio"],
        "linearization_rel_err": n2_ev["guarded_repeat_max_rel_err"],
        "branch_evidence": "受信任分支；契约保护复核 40/40",
        "failure_class": "初始化敏感；多平衡分支竞争；非全局灵敏度",
        "allowed_use": "同一工况下三维末端位置局部导数复核",
        "forbidden_use": "完整末端位姿；跨目标点复用；默认外层 IK",
        "source": "P2c107",
    }
    full_pose_row = {
        "object_task": "n2 完整末端位姿",
        "magnet_count": 2,
        "task_dimension": 6,
        "decision_level": "暂缓发布",
        "stationarity_norm": None,
        "condition_number": None,
        "force_ratio": None,
        "linearization_rel_err": None,
        "branch_evidence": "尚无完整证据链",
        "failure_class": "证据不足；任务维度与分支稳定性未闭合",
        "allowed_use": "诊断记录",
        "forbidden_use": "发布；外层 IK 默认灵敏度",
        "source": "P2c100/P2c107",
    }
    return [n1_row, n2_row, full_pose_row]


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese milestone report."""
    rows = summary["rows"]
    table = [
        "| 对象与任务 | 判定 | 关键门禁 | 失败或风险 | 工程建议 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        gate = (
            f"载荷 {row['force_ratio']:.3e}；线性化 {row['linearization_rel_err']:.3e}"
            if row["force_ratio"] is not None and row["linearization_rel_err"] is not None
            else f"载荷 {row['force_ratio']:.3e}" if row["force_ratio"] is not None else "证据不足"
        )
        table.append(
            f"| {row['object_task']} | {row['decision_level']} | {gate} | {row['failure_class']} | {row['allowed_use']} |"
        )
    return "\n".join(
        [
            "# P2c108 IK 灵敏度阶段边界综合报告",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是综合 P2c99、P2c100 和 P2c107，形成当前单点与两点磁控导管灵敏度研究的阶段边界。报告重点是区分可发布候选、诊断可用结果和不可发布工况，避免把末端拟合或单次复核误解为全局灵敏度。",
            "",
            "## 阶段判定",
            "",
            *table,
            "",
            "## 主要结论",
            "",
            "两点磁控三维末端位置灵敏度已经形成受信任分支条件下的接口候选。该候选同时满足一阶最优性、条件数、载荷传播、局部线性化和扰动重求解复核，并已加入缓存契约保护。因此它可以作为同一工况下的局部导数复核入口。",
            "",
            "单点磁控有效子空间灵敏度仍不可发布。阻断因素不是末端拟合误差，而是刚性磁体段外力与柔性段内力链之间存在稳定的载荷传播失配。该失败已跨多轮诊断重复出现，应进入模型修复而不是继续扫扰动方向。",
            "",
            "完整末端位姿灵敏度暂不发布。当前证据链只覆盖位置任务；姿态任务会引入更高任务维度和更强分支竞争，必须等待 position-only 任务和物理闭合模型更稳定后再推进。",
            "",
            "## 工程建议",
            "",
            "1. n2 三维末端位置可用于离线诊断和同一工况下的快速复核，必须标注“分支条件局部导数”。",
            "2. n2 不得作为默认外层 IK 灵敏度模型；若实验调用，必须显式开关并先通过契约校验。",
            "3. n1 在修复刚段-柔性段载荷传播模型前，不再做大规模扰动灵敏度扫描。",
            "4. 完整末端位姿任务仅保留诊断，不作为近期成功目标。",
            "",
            "## 下一步计划",
            "",
            "1. n2：把 P2c107 接口作为后续外层 IK 显式实验入口，但保持默认关闭。",
            "2. n2：继续研究首次可信分支恢复成本，优先评估隐式灵敏度或更稳定的分支恢复策略。",
            "3. n1：开展接口力连续模型修复，目标是把载荷传播比例压回 1.5 以下，再重新启动有效子空间审计。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    p2c99 = _read(P2C99) if P2C99.exists() else {}
    p2c100 = _read(P2C100)
    p2c107 = _read(P2C107)
    rows = _build_rows(p2c99, p2c100, p2c107)
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in rows
        if row["decision_level"] != "可作为分支条件候选"
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "rows": rows,
        "source_files": [str(P2C99), str(P2C100), str(P2C107)],
        "wall_s": float(time.perf_counter() - t0),
        "goal_progress": {
            "n2_position": "branch_conditioned_candidate",
            "n1_effective_position": "nonpublishable_until_model_repair",
            "full_pose": "diagnostic_only",
            "overall_goal_complete": False,
        },
    }
    _write_csv(OUT_DIR / "decision_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps(summary["goal_progress"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
