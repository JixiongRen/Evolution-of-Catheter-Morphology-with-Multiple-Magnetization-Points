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
P2C171_DIR = ROOT / "experiment_records_ik_optimize/20260705004000_p2c171_n2_truncated_sensitivity_probe"
for path in (ROOT, P2C171_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c171_n2_truncated_sensitivity_probe as p2c171


RADII_A = (1.0e-4, 2.5e-4, 5.0e-4, 1.0e-3)
MODE_INDICES = (0, 1, 2)
LINEAR_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
CONTINUITY_GATE = 1.0e-3
OPTIMALITY_GATE = 1.0e-5


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


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one exact-Jacobian reoptimization row."""
    if not row["solve_success"]:
        return "不可发布", "扰动重求解未收敛"
    if row["optimality"] > OPTIMALITY_GATE:
        return "不可发布", "一阶最优性超限"
    if row["current_aware_continuity_norm"] > CONTINUITY_GATE:
        return "不可发布", "接口连续残差超限"
    if row["centerline_max_change_m"] > CENTERLINE_GATE_M:
        return "不可发布", "中心线分支漂移超限"
    if row["q_pred_vs_actual_rel_err"] > LINEAR_GATE:
        return "不可发布", "状态线性化误差超限"
    if row["task_pred_vs_actual_rel_err"] > LINEAR_GATE:
        return "不可发布", "任务线性化误差超限"
    if row["output_linear_at_actual_q_rel_err"] > LINEAR_GATE:
        return "不可发布", "末端输出线性化误差超限"
    return "通过", "精确雅可比扰动重求解通过"


def evaluate_case(branch: dict[str, Any], system: dict[str, Any], mode_index: int, radius: float) -> dict[str, Any]:
    """Evaluate one mode-radius pair with exact-Jacobian reduced reoptimization."""
    sensitivity = p2c171.sensitivity_truncated(system, 0.0)
    task_direction = np.asarray(branch["U"][:, mode_index], dtype=float)
    d_i = p2c171.current_step_for_direction(sensitivity, task_direction, radius)
    current_new = np.asarray(branch["I0"], dtype=float) + d_i
    q_pred = p2c171.raw_q_truncated(system, d_i, 0.0)
    q_actual, info = p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=np.asarray(branch["y"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        I=current_new,
        funcs=branch["funcs"],
    )
    y_new = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ q_actual
    out_new = p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.eval_output(
        branch["funcs"],
        y_new,
        current_new,
    )
    base_tip = np.asarray(branch["base_out"]["tip_p"], dtype=float)
    base_centerline = np.asarray(branch["base_out"]["centerline"], dtype=float)
    actual = np.asarray(out_new["tip_p"], dtype=float) - base_tip
    pred = sensitivity @ d_i
    output_linear = np.asarray(system["t_eff"] @ q_actual, dtype=float)
    current_blocks = p2c171.p2c170.p2c169.p2c168.p2c165.current_aware_interface_blocks(
        branch,
        y_new,
        current_new,
    )
    current_metrics = p2c171.p2c170.p2c169.p2c168.p2c165.metric_from_blocks(current_blocks)
    row = {
        "mode_index": int(mode_index + 1),
        "radius_A": float(radius),
        "current_step_norm_A": float(np.linalg.norm(d_i)),
        "q_pred_norm": float(np.linalg.norm(q_pred)),
        "q_actual_norm": float(np.linalg.norm(q_actual)),
        "q_pred_vs_actual_rel_err": p2c171.p2c170.p2c169.p2c168.rel_err(q_pred, q_actual),
        "q_pred_actual_cosine": p2c171.p2c170.p2c169.p2c168.cosine(q_pred, q_actual),
        "task_pred_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_task_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "task_pred_vs_actual_rel_err": p2c171.p2c170.p2c169.p2c168.rel_err(pred, actual),
        "task_pred_actual_cosine": p2c171.p2c170.p2c169.p2c168.cosine(pred, actual),
        "output_linear_at_actual_q_rel_err": p2c171.p2c170.p2c169.p2c168.rel_err(output_linear, actual),
        "output_linear_actual_q_cosine": p2c171.p2c170.p2c169.p2c168.cosine(output_linear, actual),
        "centerline_max_change_m": float(np.max(np.linalg.norm(np.asarray(out_new["centerline"], dtype=float) - base_centerline, axis=1))),
        "current_aware_continuity_norm": current_metrics["continuity_norm"],
        "current_aware_dominant_force_norm": current_metrics["dominant_force_norm"],
        "current_aware_dominant_torque_norm": current_metrics["dominant_torque_norm"],
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
        "optimality": float(info["optimality"]),
        "cost": float(info["cost"]),
    }
    row["release_class"], row["reason"] = classify(row)
    return row


def summarize(rows: list[dict[str, Any]], system: dict[str, Any], wall_s: float) -> dict[str, Any]:
    """Summarize exact-Jacobian gate rows."""
    passed = [row for row in rows if row["release_class"] == "通过"]
    by_mode = {}
    for mode in MODE_INDICES:
        mode_rows = [row for row in rows if row["mode_index"] == mode + 1]
        mode_passed = [row for row in mode_rows if row["release_class"] == "通过"]
        by_mode[str(mode + 1)] = {
            "tested_count": len(mode_rows),
            "pass_count": len(mode_passed),
            "max_pass_radius_A": max((row["radius_A"] for row in mode_passed), default=0.0),
            "best_task_error": min(float(row["task_pred_vs_actual_rel_err"]) for row in mode_rows),
        }
    if len(passed) == len(rows):
        status = "exact_jacobian_gate_all_pass"
        decision = "全部方向和半径通过，可进入跨工作点复核。"
    elif passed:
        status = "exact_jacobian_gate_partial_pass"
        decision = "精确雅可比路径存在可用局部范围，但尚未覆盖全部方向和半径。"
    else:
        status = "exact_jacobian_gate_failed"
        decision = "精确雅可比路径未形成可发布局部灵敏度。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "tested_count": len(rows),
        "pass_count": len(passed),
        "raw_optimality_condition": float(system["raw_condition"]),
        "mode_summary": by_mode,
        "outer_ik_allowed": False,
        "publishable_sensitivity": False,
        "wall_s": float(wall_s),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot exact-Jacobian gate task errors."""
    configure_matplotlib_cjk()
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for mode in MODE_INDICES:
        mode_rows = sorted([row for row in rows if row["mode_index"] == mode + 1], key=lambda row: row["radius_A"])
        ax.plot(
            [row["radius_A"] for row in mode_rows],
            [row["task_pred_vs_actual_rel_err"] for row in mode_rows],
            marker="o",
            label=f"方向 {mode + 1}",
        )
    ax.axhline(LINEAR_GATE, color="black", linestyle="--", linewidth=1, label="门限")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("电流半径/A")
    ax.set_ylabel("任务线性化误差")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle("n2 精确雅可比扰动重求解门禁")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_exact_jacobian_reopt_gate_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 方向 | 半径/A | 任务误差 | 状态误差 | 输出误差 | 连续残差 | 最优性 | 结论 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        table.append(
            f"| {row['mode_index']} | {row['radius_A']:.1e} | {row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['q_pred_vs_actual_rel_err']:.3e} | {row['output_linear_at_actual_q_rel_err']:.3e} | "
            f"{row['current_aware_continuity_norm']:.3e} | {row['optimality']:.3e} | {row['release_class']} |"
        )
    text = f"""# P2c173 两点磁控 n2 精确雅可比扰动重求解门禁

## 实验目的

P2c172 将主失败嫌疑从物理模型转向扰动重求解路径。本轮使用精确雅可比约化重求解，在多个局部任务方向和多个电流半径上复测线性化门禁，判断该路径是否具备可发布候选基础。

## 实验方案

复用同一两点磁控工作点和第一步后局部分支。局部任务方向取前三个主方向，电流半径取 `{list(RADII_A)}`。每个样本均记录任务线性化误差、状态线性化误差、末端输出线性化误差、随电流更新后的接口连续残差、一阶最优性和中心线分支漂移。

## 结果

{chr(10).join(table)}

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。原始最优性方程条件数仍为 {summary['raw_optimality_condition']:.3e}，因此即使部分样本通过，也只能视为局部分支条件下的候选灵敏度，不能解释为全局稳定灵敏度。

## 实验结论

{summary['decision']}

## 下一步计划

若存在通过样本，应对通过范围做跨工作点复核，并补充扰动回环与多初值分支一致性测试。若只有第一方向或极小半径通过，则只能发布受限有效子空间，不能发布完整三维末端位置灵敏度。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the exact-Jacobian reoptimization gate."""
    t0 = time.perf_counter()
    base = p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch1)
    rows = [evaluate_case(branch1, system, mode, radius) for mode in MODE_INDICES for radius in RADII_A]
    summary = summarize(rows, system, time.perf_counter() - t0)
    summary["case"] = direction_name
    summary["path"] = "first_principal_then_exact_jacobian_reopt_gate"
    write_csv(EXP_DIR / "exact_jacobian_reopt_gate_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if row["release_class"] != "通过":
                f.write(json.dumps({"failure_class": row["reason"], "evidence": jsonable(row)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705013000_p2c173_n2_exact_jacobian_reopt_gate/run_p2c173_n2_exact_jacobian_reopt_gate.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
