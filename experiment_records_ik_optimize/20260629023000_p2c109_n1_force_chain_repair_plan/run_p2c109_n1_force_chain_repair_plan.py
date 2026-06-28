"""Build a repair plan for the n1 rigid-flex force-chain mismatch."""

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
P2C84 = EXPERIMENT_ROOT / "20260628121600_p2c84_block_stationarity_load_closure_audit" / "summary.json"
P2C87 = EXPERIMENT_ROOT / "20260628140500_p2c87_n1_rigid_endpoint_load_closure" / "summary.json"
P2C99 = EXPERIMENT_ROOT / "20260628215500_p2c99_n1_nonpublishable_boundary_synthesis" / "summary.json"
P2C108 = EXPERIMENT_ROOT / "20260629020000_p2c108_sensitivity_boundary_milestone" / "summary.json"


def _read(path: Path) -> dict[str, Any]:
    """Read JSON when it exists."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows() -> list[dict[str, Any]]:
    """Return model-repair candidates for the n1 force chain."""
    return [
        {
            "candidate": "A 硬力连续",
            "equation": "n_{i+1}^{0} = n_i^{L_i} + F_i",
            "physical_meaning": "刚段外力必须由相邻柔性段内力链传递",
            "implementation_target": "interface_wrench_residual / forward_interface_slack",
            "expected_benefit": "直接压低力跳变，物理解释最强",
            "main_risk": "可能使当前目标点不可达，导致末端误差或条件数恶化",
            "priority": 1,
        },
        {
            "candidate": "B 低维物理载荷传递参数",
            "equation": "n_{i+1}^{0} = n_i^{L_i} + beta_i F_i, beta_i in [0, 1]",
            "physical_meaning": "只允许沿刚段实际外力方向的可解释载荷传递比例",
            "implementation_target": "physical_direction_slack / load_scaled_slack",
            "expected_benefit": "比任意松弛更可解释，能定位缺失载荷比例",
            "main_risk": "若 beta 只为拟合服务，仍可能退化成松弛变量",
            "priority": 2,
        },
        {
            "candidate": "C 接口残差分块重标定",
            "equation": "r_force = (n_{i+1}^{0} - n_i^{L_i} - F_i) / max(||F_i||, F_floor)",
            "physical_meaning": "使用刚段外力尺度归一化力连续残差",
            "implementation_target": "load_scaled_slack_residual",
            "expected_benefit": "改善尺度病态，适合与 A 或 B 组合",
            "main_risk": "单独使用只改变权重，不修复物理链",
            "priority": 3,
        },
        {
            "candidate": "D 仅改变刚段力矩作用点",
            "equation": "m_{i+1}^{0} = m_i^{L_i} + tau_i + r x F_i",
            "physical_meaning": "修正力矩传播臂",
            "implementation_target": "_advance_rigid_with_wrench / rigid_state_along_dim",
            "expected_benefit": "P2c87 已显示可修复力矩跳变",
            "main_risk": "不能修复力连续性，不应作为 n1 主修复",
            "priority": 4,
        },
    ]


def _evidence_rows(p2c84: dict[str, Any], p2c87: dict[str, Any], p2c108: dict[str, Any]) -> list[dict[str, Any]]:
    """Build evidence rows for the n1 failure mechanism."""
    n1_gate = next((row for row in p2c84.get("gate_rows", []) if row.get("label") == "n1_soft_near_gate"), {})
    p2c108_n1 = next((row for row in p2c108.get("rows", []) if row.get("object_task") == "n1 有效子空间位置"), {})
    best_torque = p2c87.get("summary", {}).get("best_torque_model", {})
    return [
        {
            "source": "P2c84",
            "evidence": "近门样本一阶最优性和条件数可诊断，但载荷传播失败",
            "stationarity_norm": n1_gate.get("stationarity_norm"),
            "condition_number": n1_gate.get("jacobian_condition"),
            "force_ratio": n1_gate.get("max_interface_force_jump_over_rigid_force"),
            "conclusion": "不是简单末端拟合失败",
        },
        {
            "source": "P2c87",
            "evidence": "最优力矩模型可使力矩跳变约为零，但力跳变仍超门禁",
            "stationarity_norm": None,
            "condition_number": None,
            "force_ratio": p2c87.get("summary", {}).get("max_force_jump_over_rigid_force"),
            "conclusion": f"力矩作用点修正不足；最佳力矩模型为 {best_torque.get('model')}",
        },
        {
            "source": "P2c108",
            "evidence": "阶段边界要求 n1 转入模型修复",
            "stationarity_norm": p2c108_n1.get("stationarity_norm"),
            "condition_number": p2c108_n1.get("condition_number"),
            "force_ratio": p2c108_n1.get("force_ratio"),
            "conclusion": p2c108_n1.get("failure_class"),
        },
    ]


def _code_rows() -> list[dict[str, Any]]:
    """Return code locations relevant to the repair."""
    return [
        {
            "path": "experiment_records_ik_optimize/20260605102000_p2c10t_interface_slack_shooting/run_p2c10t_interface_slack_shooting.py",
            "symbol": "forward_interface_slack",
            "role": "任意接口力/力矩松弛的基线实现",
        },
        {
            "path": "experiment_records_ik_optimize/20260605104500_p2c10u_physical_direction_slack/run_p2c10u_physical_direction_slack.py",
            "symbol": "forward_physical_direction_slack",
            "role": "沿刚段外载方向的低维松弛候选",
        },
        {
            "path": "experiment_records_ik_optimize/20260607144044_p2c41_scaled_interface_slack_baseline_gate/run_p2c41_scaled_interface_slack_baseline_gate.py",
            "symbol": "load_scaled_slack_residual",
            "role": "按局部刚段载荷归一化接口残差",
        },
        {
            "path": "experiment_records_ik_optimize/20260607200500_p2c57_force_continuity_weight_repair/run_p2c57_force_continuity_weight_repair.py",
            "symbol": "_make_split_functions",
            "role": "P2c84-P2c87 近门样本恢复入口",
        },
        {
            "path": "experiment_records_ik_optimize/20260628140500_p2c87_n1_rigid_endpoint_load_closure/run_p2c87_n1_rigid_endpoint_load_closure.py",
            "symbol": "_audit_interfaces",
            "role": "当前载荷传播失败审计逻辑",
        },
    ]


def _plot(evidence_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    """Plot evidence and candidate priorities."""
    plotting_fonts.configure_matplotlib_cjk()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [row["source"] for row in evidence_rows]
    force_vals = [row["force_ratio"] or 0.0 for row in evidence_rows]
    axes[0].bar(labels, force_vals, color="#e76f51")
    axes[0].axhline(1.5, color="black", linestyle="--", label="门禁")
    axes[0].set_ylabel("载荷传播比例")
    axes[0].set_title("n1 当前失败证据")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    cand_labels = [row["candidate"] for row in candidate_rows]
    priority_scores = [5 - int(row["priority"]) for row in candidate_rows]
    axes[1].bar(cand_labels, priority_scores, color="#457b9d")
    axes[1].set_ylabel("优先级评分")
    axes[1].set_title("修复候选排序")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c109 n1 载荷传播模型修复预案")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_force_chain_repair_plan_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese repair-plan report."""
    evidence_table = [
        "| 来源 | 证据 | 载荷传播比例 | 结论 |",
        "| --- | --- | ---: | --- |",
    ]
    for row in summary["evidence_rows"]:
        force_text = f"{row['force_ratio']:.3e}" if row["force_ratio"] is not None else "无"
        evidence_table.append(f"| {row['source']} | {row['evidence']} | {force_text} | {row['conclusion']} |")
    candidate_table = [
        "| 候选 | 方程形式 | 物理含义 | 风险 | 优先级 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in summary["candidate_rows"]:
        candidate_table.append(
            f"| {row['candidate']} | `{row['equation']}` | {row['physical_meaning']} | {row['main_risk']} | {row['priority']} |"
        )
    code_table = [
        "| 文件 | 符号 | 作用 |",
        "| --- | --- | --- |",
    ]
    for row in summary["code_rows"]:
        code_table.append(f"| `{row['path']}` | `{row['symbol']}` | {row['role']} |")
    return "\n".join(
        [
            "# P2c109 n1 刚段-柔性段载荷传播修复预案",
            "",
            "## 实验目的",
            "",
            "P2c99 和 P2c108 已判定单点磁控 n1 当前不可发布。本轮不重新求解，而是把既有证据转化为可执行的模型修复路线：明确应优先修复力连续方程，而不是继续调整末端权重、边界权重或力矩作用点。",
            "",
            "## 证据链",
            "",
            *evidence_table,
            "",
            "## 力连续性修复方程",
            "",
            "当前阻断项可写为刚段远端柔性段初始内力与刚段传播后内力之间的跳变：",
            "",
            "$$\\Delta n_i = n_{i+1}^{0} - \\left(n_i^{L_i} + F_i\\right).$$",
            "",
            "若 $\\|\\Delta n_i\\| / \\|F_i\\| > 1.5$，则该分支不能发布为物理可信灵敏度。P2c87 表明改变力矩作用点可以改善 $\\Delta m_i$，但不能改善 $\\Delta n_i$，因此下一轮必须直接处理 $\\Delta n_i$。",
            "",
            "## 修复候选",
            "",
            *candidate_table,
            "",
            "## 代码定位",
            "",
            *code_table,
            "",
            "## 推荐路径",
            "",
            "第一步应实现 A 与 C 的组合：把硬力连续残差作为主物理约束，并使用刚段外力尺度归一化，避免小外力工况下尺度病态。若 A 直接导致目标不可达，再尝试 B：只允许沿实际刚段外力方向的低维载荷传递参数。D 已被 P2c87 证伪为 n1 主修复方案，只能保留为力矩项修正。",
            "",
            "## 验收标准",
            "",
            "1. 载荷传播比例必须从 `2.769` 降至 `1.5` 以下。",
            "2. 一阶最优性仍需不高于 `1e-5`。",
            "3. 条件数需低于 `1e7`，否则只做诊断。",
            "4. 末端位置误差不能以牺牲物理闭合为代价人为压低。",
            "5. 过门后才能重启有效子空间扰动灵敏度审计。",
            "",
            "## 下一步计划",
            "",
            "1. 新建实验实现 A+C 残差形式，仅在实验脚本中验证，不立即替换生产接口。",
            "2. 对 `n1_tip_z_rand15, alpha=0.10` 复测 P2c84 相同门禁。",
            "3. 若 A+C 失败，记录目标不可达或条件数退化，再进入 B 的低维物理载荷传递参数实验。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    evidence_rows = _evidence_rows(_read(P2C84), _read(P2C87), _read(P2C108))
    candidate_rows = _candidate_rows()
    code_rows = _code_rows()
    failures = [
        {"failure": row["conclusion"], "evidence": row}
        for row in evidence_rows
        if row["force_ratio"] is not None and row["force_ratio"] > 1.5
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "evidence_rows": evidence_rows,
        "candidate_rows": candidate_rows,
        "code_rows": code_rows,
        "recommended_next_experiment": "n1_hard_force_continuity_load_scaled_residual",
        "goal_progress": "n1 moved from diagnosis to model-repair specification",
    }
    _write_csv(OUT_DIR / "evidence_rows.csv", evidence_rows)
    _write_csv(OUT_DIR / "candidate_rows.csv", candidate_rows)
    _write_csv(OUT_DIR / "code_rows.csv", code_rows)
    _write_csv(OUT_DIR / "summary.csv", evidence_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(evidence_rows, candidate_rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps({"recommended_next_experiment": summary["recommended_next_experiment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
