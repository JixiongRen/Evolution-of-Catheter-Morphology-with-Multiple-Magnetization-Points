from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C173_DIR = ROOT / "experiment_records_ik_optimize/20260705013000_p2c173_n2_exact_jacobian_reopt_gate"
for path in (ROOT, P2C173_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c173_n2_exact_jacobian_reopt_gate as p2c173


RADIUS_A = 1.0e-4
MODE_INDEX = 0
LINEAR_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
CONTINUITY_GATE = 1.0e-3
OPTIMALITY_GATE = 1.0e-5
CONSISTENCY_Q_GATE = 1.0e-5
RETURN_Q_GATE = 1.0e-5


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def solve_from_q0(branch: dict[str, Any], current: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the reduced state from a given initial offset."""
    return p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.solve_reduced(
        q0=np.asarray(q0, dtype=float),
        y_base=np.asarray(branch["y"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        I=np.asarray(current, dtype=float),
        funcs=branch["funcs"],
    )


def output(branch: dict[str, Any], q: np.ndarray, current: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate output at a reduced state."""
    y = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ np.asarray(q, dtype=float)
    return p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.eval_output(
        branch["funcs"],
        y,
        current,
    )


def build_step(branch: dict[str, Any], system: dict[str, Any], sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one signed current step and predicted reduced state."""
    sensitivity = p2c173.p2c171.sensitivity_truncated(system, 0.0)
    task_direction = sign * np.asarray(branch["U"][:, MODE_INDEX], dtype=float)
    d_i = p2c173.p2c171.current_step_for_direction(sensitivity, task_direction, RADIUS_A)
    current_new = np.asarray(branch["I0"], dtype=float) + d_i
    q_pred = p2c173.p2c171.raw_q_truncated(system, d_i, 0.0)
    return d_i, current_new, q_pred


def line_metrics(branch: dict[str, Any], system: dict[str, Any], sign: float, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run signed perturbation with multiple starts and return-loop checks."""
    d_i, current_new, q_pred = build_step(branch, system, sign)
    starts = {
        "zero": np.zeros_like(q_pred),
        "pred": q_pred,
        "neg_pred": -q_pred,
        "half_pred": 0.5 * q_pred,
    }
    solutions = []
    base_tip = np.asarray(branch["base_out"]["tip_p"], dtype=float)
    base_centerline = np.asarray(branch["base_out"]["centerline"], dtype=float)
    pred_task = p2c173.p2c171.sensitivity_truncated(system, 0.0) @ d_i
    for start_name, q0 in starts.items():
        q, info = solve_from_q0(branch, current_new, q0)
        out = output(branch, q, current_new)
        actual = np.asarray(out["tip_p"], dtype=float) - base_tip
        output_linear = np.asarray(system["t_eff"] @ q, dtype=float)
        y_new = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ q
        blocks = p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.current_aware_interface_blocks(branch, y_new, current_new)
        continuity = p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.metric_from_blocks(blocks)["continuity_norm"]
        q_return, return_info = solve_from_q0(branch, branch["I0"], q)
        row = {
            "direction": label,
            "start": start_name,
            "q_norm": float(np.linalg.norm(q)),
            "q_pred_vs_actual_rel_err": p2c173.p2c171.p2c170.p2c169.p2c168.rel_err(q_pred, q),
            "task_pred_vs_actual_rel_err": p2c173.p2c171.p2c170.p2c169.p2c168.rel_err(pred_task, actual),
            "output_linear_at_actual_q_rel_err": p2c173.p2c171.p2c170.p2c169.p2c168.rel_err(output_linear, actual),
            "centerline_max_change_m": float(np.max(np.linalg.norm(np.asarray(out["centerline"], dtype=float) - base_centerline, axis=1))),
            "current_aware_continuity_norm": float(continuity),
            "optimality": float(info["optimality"]),
            "solve_success": bool(info["success"]),
            "solve_nfev": int(info["nfev"]),
            "solve_steady_s": float(info["steady_s"]),
            "return_q_norm": float(np.linalg.norm(q_return)),
            "return_success": bool(return_info["success"]),
            "return_optimality": float(return_info["optimality"]),
            "return_nfev": int(return_info["nfev"]),
            "return_steady_s": float(return_info["steady_s"]),
        }
        row["gate_pass"] = bool(
            row["solve_success"]
            and row["return_success"]
            and row["optimality"] <= OPTIMALITY_GATE
            and row["current_aware_continuity_norm"] <= CONTINUITY_GATE
            and row["centerline_max_change_m"] <= CENTERLINE_GATE_M
            and row["q_pred_vs_actual_rel_err"] <= LINEAR_GATE
            and row["task_pred_vs_actual_rel_err"] <= LINEAR_GATE
            and row["output_linear_at_actual_q_rel_err"] <= LINEAR_GATE
            and row["return_q_norm"] <= RETURN_Q_GATE
        )
        row["q_solution"] = q
        solutions.append(row)
    q_ref = solutions[0]["q_solution"]
    for row in solutions:
        row["q_consistency_to_zero_start"] = float(np.linalg.norm(np.asarray(row["q_solution"], dtype=float) - q_ref))
        row["gate_pass"] = bool(row["gate_pass"] and row["q_consistency_to_zero_start"] <= CONSISTENCY_Q_GATE)
    public_rows = [{k: v for k, v in row.items() if k != "q_solution"} for row in solutions]
    direction_summary = {
        "direction": label,
        "tested_count": len(public_rows),
        "pass_count": sum(1 for row in public_rows if row["gate_pass"]),
        "max_q_consistency": max(row["q_consistency_to_zero_start"] for row in public_rows),
        "max_return_q_norm": max(row["return_q_norm"] for row in public_rows),
        "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in public_rows),
        "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in public_rows),
    }
    return direction_summary, public_rows


def summarize(rows: list[dict[str, Any]], direction_summaries: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    """Summarize one-dimensional branch-loop gate."""
    passed = [row for row in rows if row["gate_pass"]]
    all_direction_pass = all(item["pass_count"] == item["tested_count"] for item in direction_summaries)
    if all_direction_pass:
        status = "n2_1d_branch_loop_candidate"
        decision = "正负扰动、多初值和回环均通过；可作为 n2 一维有效子空间候选进入跨工作点复核。"
    elif passed:
        status = "n2_1d_branch_loop_partial"
        decision = "部分初值或方向通过，但分支一致性仍不足；不能发布。"
    else:
        status = "n2_1d_branch_loop_failed"
        decision = "一维候选未通过分支一致性或回环门禁；不能发布。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "radius_A": RADIUS_A,
        "tested_count": len(rows),
        "pass_count": len(passed),
        "direction_summaries": direction_summaries,
        "outer_ik_allowed": False,
        "publishable_sensitivity": bool(all_direction_pass),
        "wall_s": float(wall_s),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot branch-loop gate errors."""
    configure_matplotlib_cjk()
    labels = [f"{row['direction']}-{row['start']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    colors = ["#2a9d8f" if row["gate_pass"] else "#e76f51" for row in rows]
    axes[0].bar(x, [row["task_pred_vs_actual_rel_err"] for row in rows], color=colors)
    axes[0].axhline(LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("任务线性化误差")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [row["return_q_norm"] for row in rows], color=colors)
    axes[1].axhline(RETURN_Q_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("回环状态误差")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("n2 一维候选分支一致性与回环门禁")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_1d_branch_loop_gate_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 方向 | 初值 | 任务误差 | 输出误差 | 连续残差 | 回环误差 | 分支差异 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        table.append(
            f"| {row['direction']} | {row['start']} | {row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['output_linear_at_actual_q_rel_err']:.3e} | {row['current_aware_continuity_norm']:.3e} | "
            f"{row['return_q_norm']:.3e} | {row['q_consistency_to_zero_start']:.3e} | "
            f"{'通过' if row['gate_pass'] else '不可发布'} |"
        )
    text = f"""# P2c174 两点磁控 n2 一维候选分支一致性与回环门禁

## 实验目的

P2c173 仅发现第 1 主方向、`1e-4 A` 半径通过局部线性化。本轮对该一维候选做更严格验证：正负扰动、多初值重求解、以及回到原电流的分支回环。

## 实验方案

固定同一 n2 工作点和第一步后局部分支。对第 1 主方向分别施加正负 `1e-4 A` 电流扰动。每个扰动使用四类初值：零初值、线性预测初值、反向预测初值、半预测初值。每个解再回到原电流，检查是否返回原分支。

## 结果

{chr(10).join(table)}

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。若正负扰动和多初值均通过，说明该一维方向至少具备局部分支一致性；若回环失败，则说明扰动解虽然局部拟合，但不应发布为稳定灵敏度。

## 实验结论

{summary['decision']}

## 下一步计划

若本轮通过，应在更多工作点上复核该一维有效子空间，并记录工程边界。若失败，应将 n2 当前配置判为“仅诊断，不发布”，转向单点磁控有效子空间或重新设计两点磁化配置。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the one-dimensional branch-loop gate."""
    t0 = time.perf_counter()
    base = p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch1)
    summaries = []
    rows = []
    for sign, label in ((1.0, "正向"), (-1.0, "反向")):
        direction_summary, direction_rows = line_metrics(branch1, system, sign, label)
        summaries.append(direction_summary)
        rows.extend(direction_rows)
    summary = summarize(rows, summaries, time.perf_counter() - t0)
    summary["case"] = direction_name
    summary["path"] = "mode1_positive_negative_multistart_return_loop"
    write_csv(EXP_DIR / "branch_loop_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                f.write(json.dumps({"failure_class": "分支一致性或回环门禁失败", "evidence": jsonable(row)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate/run_p2c174_n2_1d_branch_loop_gate.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
