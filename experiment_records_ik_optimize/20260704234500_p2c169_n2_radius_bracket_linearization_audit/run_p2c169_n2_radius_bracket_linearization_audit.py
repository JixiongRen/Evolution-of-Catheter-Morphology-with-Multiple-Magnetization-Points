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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C168_DIR = ROOT / "experiment_records_ik_optimize/20260704231500_p2c168_n2_linear_model_breakdown_audit"
for path in (ROOT, P2C168_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c168_n2_linear_model_breakdown_audit as p2c168


RADII_A = (1.0e-3, 5.0e-4, 2.5e-4, 1.0e-4)
LINEAR_GATE = 0.05


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


def evaluate_radius(branch: dict[str, Any], system: dict[str, Any], radius: float) -> dict[str, Any]:
    """Evaluate one current radius with the P2c168 decomposition."""
    q_actual, solve_info, current_new = p2c168.p2c165.solve_with_torque_weight(branch, radius, 0.0)
    d_i = np.asarray(current_new - branch["I0"], dtype=float)
    q_pred = p2c168.raw_q(system, d_i)
    base_tip = np.asarray(branch["base_out"]["tip_p"], dtype=float)
    tip_actual = p2c168.output_tip(branch, q_actual, current_new)
    actual_task = tip_actual - base_tip
    task_pred = np.asarray(branch["S"] @ d_i, dtype=float)
    task_linear_actual_q = np.asarray(system["t_eff"] @ q_actual, dtype=float)
    current_blocks = p2c168.p2c165.current_aware_interface_blocks(branch, branch["y"] + branch["basis"] @ q_actual, current_new)
    current_metrics = p2c168.p2c165.metric_from_blocks(current_blocks)
    row = {
        "radius_A": float(radius),
        "q_actual_norm": float(np.linalg.norm(q_actual)),
        "q_pred_norm": float(np.linalg.norm(q_pred)),
        "q_pred_vs_actual_rel_err": p2c168.rel_err(q_pred, q_actual),
        "q_pred_actual_cosine": p2c168.cosine(q_pred, q_actual),
        "task_pred_norm_um": float(np.linalg.norm(task_pred) * 1.0e6),
        "actual_task_norm_um": float(np.linalg.norm(actual_task) * 1.0e6),
        "task_pred_vs_actual_rel_err": p2c168.rel_err(task_pred, actual_task),
        "task_pred_actual_cosine": p2c168.cosine(task_pred, actual_task),
        "output_linear_at_actual_q_rel_err": p2c168.rel_err(task_linear_actual_q, actual_task),
        "output_linear_actual_q_cosine": p2c168.cosine(task_linear_actual_q, actual_task),
        "physical_residual_actual_norm": p2c168.residual_norm(branch, q_actual, current_new),
        "current_aware_continuity_norm": current_metrics["continuity_norm"],
        "solve_success": bool(solve_info["success"]),
        "solve_nfev": int(solve_info["nfev"]),
        "solve_steady_s": float(solve_info["steady_s"]),
        "optimality": float(solve_info["optimality"]),
    }
    row["output_linear_pass"] = bool(row["output_linear_at_actual_q_rel_err"] < LINEAR_GATE)
    row["task_linear_pass"] = bool(row["task_pred_vs_actual_rel_err"] < LINEAR_GATE)
    row["q_linear_pass"] = bool(row["q_pred_vs_actual_rel_err"] < LINEAR_GATE)
    return row


def summarize(rows: list[dict[str, Any]], condition: float, wall_s: float) -> dict[str, Any]:
    """Summarize the radius bracket."""
    output_pass = [row for row in rows if row["output_linear_pass"]]
    task_pass = [row for row in rows if row["task_linear_pass"]]
    q_pass = [row for row in rows if row["q_linear_pass"]]
    if not output_pass:
        status = "output_linearization_not_recovered"
        decision = "缩小到 1e-4 A 后，末端输出线性化仍未恢复；需更换局部参数化或审计输出映射。"
    elif output_pass and not q_pass:
        status = "output_recovers_but_optimality_breaks"
        decision = "小半径下末端输出线性化可恢复，但电流到状态的一阶最优性灵敏度仍失败；主修病态最优性方程。"
    elif task_pass:
        status = "small_radius_publishable_candidate"
        decision = "存在小半径候选通过线性化门禁，可继续复核分支一致性和载荷传播。"
    else:
        status = "mixed_radius_response"
        decision = "半径夹逼显示部分环节恢复，但任务预测仍不稳定，需要进一步分块审计。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "condition": float(condition),
        "tested_radii_A": [float(row["radius_A"]) for row in rows],
        "min_output_linear_error": min(float(row["output_linear_at_actual_q_rel_err"]) for row in rows),
        "min_task_linear_error": min(float(row["task_pred_vs_actual_rel_err"]) for row in rows),
        "min_q_linear_error": min(float(row["q_pred_vs_actual_rel_err"]) for row in rows),
        "output_pass_count": len(output_pass),
        "task_pass_count": len(task_pass),
        "q_pass_count": len(q_pass),
        "outer_ik_allowed": False,
        "publishable_sensitivity": False,
        "wall_s": float(wall_s),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot radius-bracket linearization errors."""
    configure_matplotlib_cjk()
    radii = np.asarray([row["radius_A"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(radii, [row["q_pred_vs_actual_rel_err"] for row in rows], marker="o", label="状态预测误差")
    ax.plot(radii, [row["task_pred_vs_actual_rel_err"] for row in rows], marker="s", label="任务预测误差")
    ax.plot(radii, [row["output_linear_at_actual_q_rel_err"] for row in rows], marker="^", label="输出线性化误差")
    ax.axhline(LINEAR_GATE, color="black", linestyle="--", linewidth=1, label="门限")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("第二步电流半径/A")
    ax.set_ylabel("相对误差")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle("n2 第二步半径夹逼线性化审计")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_radius_bracket_linearization_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 半径/A | 状态预测误差 | 任务预测误差 | 输出线性化误差 | 连续残差 | 结论 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        label = "通过" if row["task_linear_pass"] else "不可发布"
        table.append(
            f"| {row['radius_A']:.1e} | {row['q_pred_vs_actual_rel_err']:.3e} | "
            f"{row['task_pred_vs_actual_rel_err']:.3e} | {row['output_linear_at_actual_q_rel_err']:.3e} | "
            f"{row['current_aware_continuity_norm']:.3e} | {label} |"
        )
    text = f"""# P2c169 两点磁控 n2 第二步半径夹逼线性化审计

## 实验目的

P2c168 发现第二步同时存在最优性方程病态和末端输出线性化断裂。本轮缩小第二步电流半径，判断输出线性化是否随步长变小而恢复。

## 实验方案

复用 P2c168 的同一工作点、同一分支和同一局部第一主方向。第二步半径取 `{summary['tested_radii_A']}`。每个半径均重新求解扰动状态，并记录状态预测误差、任务预测误差、输出线性化误差和随电流更新后的接口连续残差。

## 结果

{chr(10).join(table)}

## 数据分析

最优性方程条件数为 {summary['condition']:.3e}，远高于 1e9。最小状态预测误差为 {summary['min_q_linear_error']:.3e}，最小任务预测误差为 {summary['min_task_linear_error']:.3e}，最小输出线性化误差为 {summary['min_output_linear_error']:.3e}。

## 实验结论

{summary['decision']}

## 下一步计划

若输出线性化随半径恢复，应优先修复最优性方程病态，包括残差尺度、约化基和弱方向处理。若输出线性化不恢复，应转向更换局部参数化或重新定义任务局部坐标。本轮不接入外层 IK。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the radius-bracket linearization audit."""
    t0 = time.perf_counter()
    base = p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c168.build_optimality_system(branch1)
    rows = [evaluate_radius(branch1, system, radius) for radius in RADII_A]
    summary = summarize(rows, system["raw_condition"], time.perf_counter() - t0)
    summary["case"] = direction_name
    summary["path"] = "first_principal_then_radius_bracket_second_step"
    write_csv(EXP_DIR / "radius_bracket_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"failure_class": summary["decision"], "evidence": jsonable(summary)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260704234500_p2c169_n2_radius_bracket_linearization_audit/run_p2c169_n2_radius_bracket_linearization_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
