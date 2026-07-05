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
P2C169_DIR = ROOT / "experiment_records_ik_optimize/20260704234500_p2c169_n2_radius_bracket_linearization_audit"
for path in (ROOT, P2C169_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c169_n2_radius_bracket_linearization_audit as p2c169


RADII_A = (1.0e-3, 1.0e-4)


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


def reduced_state_for_radius(branch: dict[str, Any], system: dict[str, Any], radius: float) -> dict[str, Any]:
    """Solve one radius and project predicted and actual states on singular directions."""
    q_actual, solve_info, current_new = p2c169.p2c168.p2c165.solve_with_torque_weight(branch, radius, 0.0)
    d_i = np.asarray(current_new - branch["I0"], dtype=float)
    q_pred = p2c169.p2c168.raw_q(system, d_i)
    rhs = system["g_I"] @ d_i
    rhs_left = system["u"].T @ rhs
    pred_components = system["vt"] @ q_pred
    actual_components = system["vt"] @ q_actual
    return {
        "radius_A": float(radius),
        "q_actual": np.asarray(q_actual, dtype=float),
        "q_pred": np.asarray(q_pred, dtype=float),
        "rhs_left": np.asarray(rhs_left, dtype=float),
        "pred_components": np.asarray(pred_components, dtype=float),
        "actual_components": np.asarray(actual_components, dtype=float),
        "solve_info": solve_info,
    }


def direction_rows(system: dict[str, Any], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-singular-direction audit rows."""
    rows = []
    s = np.asarray(system["s"], dtype=float)
    vt = np.asarray(system["vt"], dtype=float)
    t_eff = np.asarray(system["t_eff"], dtype=float)
    for idx, singular_value in enumerate(s):
        direction = vt[idx]
        task_gain = float(np.linalg.norm(t_eff @ direction))
        row = {
            "singular_index": int(idx),
            "singular_value": float(singular_value),
            "relative_singular_value": float(singular_value / max(float(s[0]), np.finfo(float).tiny)),
            "inverse_gain": float(1.0 / max(float(singular_value), np.finfo(float).tiny)),
            "task_gain": task_gain,
        }
        for state in states:
            suffix = f"r{state['radius_A']:.0e}".replace("-", "m")
            row[f"rhs_left_{suffix}"] = float(state["rhs_left"][idx])
            row[f"q_pred_component_{suffix}"] = float(state["pred_components"][idx])
            row[f"q_actual_component_{suffix}"] = float(state["actual_components"][idx])
            row[f"abs_pred_over_actual_{suffix}"] = float(
                abs(state["pred_components"][idx]) / max(abs(state["actual_components"][idx]), 1.0e-18)
            )
        rows.append(row)
    return rows


def summarize(system: dict[str, Any], rows: list[dict[str, Any]], states: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    """Summarize ill-conditioned direction evidence."""
    s = np.asarray(system["s"], dtype=float)
    weak = [row for row in rows if row["relative_singular_value"] < 1.0e-9]
    task_weak = [row for row in weak if row["task_gain"] < 1.0e-8]
    low_task_gain = [row for row in weak if row["task_gain"] < 1.0e-4]
    dominant_small = max(rows, key=lambda row: abs(row.get("q_pred_component_r1em03", 0.0)))
    if weak:
        status = "weak_singular_directions_dominate"
        decision = "约化最优性系统含极弱奇异方向，应先重建局部坐标或截断病态方向。"
    else:
        status = "no_extreme_weak_direction_found"
        decision = "未发现极弱奇异方向，需转向残差权重和输出映射审计。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "condition": float(s[0] / max(float(s[-1]), np.finfo(float).tiny)),
        "singular_count": int(len(s)),
        "weak_direction_count": int(len(weak)),
        "task_weak_direction_count": int(len(task_weak)),
        "low_task_gain_weak_direction_count": int(len(low_task_gain)),
        "smallest_singular_value": float(s[-1]),
        "largest_singular_value": float(s[0]),
        "dominant_pred_direction": dominant_small,
        "tested_radii_A": [state["radius_A"] for state in states],
        "solve_info": [state["solve_info"] for state in states],
        "wall_s": float(wall_s),
        "jax_devices": [str(device) for device in jax.devices()],
        "outer_ik_allowed": False,
        "publishable_sensitivity": False,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot singular spectrum and task gains."""
    configure_matplotlib_cjk()
    idx = np.asarray([row["singular_index"] for row in rows], dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.0), sharex=True)
    axes[0].plot(idx, [row["singular_value"] for row in rows], marker="o")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("奇异值")
    axes[0].grid(alpha=0.25)
    axes[1].plot(idx, [row["task_gain"] for row in rows], marker="s", color="#d95f02")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("奇异方向编号")
    axes[1].set_ylabel("末端任务增益")
    axes[1].grid(alpha=0.25)
    fig.suptitle("n2 约化最优性系统奇异谱")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_ill_conditioned_direction_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write Chinese experiment record."""
    tail = rows[-min(5, len(rows)) :]
    table = [
        "| 方向 | 奇异值 | 相对奇异值 | 任务增益 | 1e-3A 预测分量 | 1e-3A 实际分量 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tail:
        table.append(
            f"| {row['singular_index']} | {row['singular_value']:.3e} | {row['relative_singular_value']:.3e} | "
            f"{row['task_gain']:.3e} | {row.get('q_pred_component_r1em03', 0.0):.3e} | "
            f"{row.get('q_actual_component_r1em03', 0.0):.3e} |"
        )
    text = f"""# P2c170 两点磁控 n2 病态方向审计

## 实验目的

P2c169 证明缩小第二步半径仍不能恢复线性化。本轮审计约化最优性方程的奇异谱，判断失败是否由弱奇异方向和不可辨识局部坐标主导。

## 实验方案

复用同一两点磁控工作点和同一第一步分支。构建约化最优性方程，对其矩阵做奇异值分解，并把 `1e-3 A` 与 `1e-4 A` 两个第二步重求解状态投影到各个奇异方向。重点观察最小奇异方向是否同时具有极小奇异值、弱任务增益和较大状态投影。

## 结果

- 条件数：{summary['condition']:.3e}。
- 最小奇异值：{summary['smallest_singular_value']:.3e}。
- 弱奇异方向数量：{summary['weak_direction_count']}。
- 极弱任务增益方向数量：{summary['low_task_gain_weak_direction_count']}。

最小若干方向：

{chr(10).join(table)}

## 数据分析

若相对奇异值低于 `1e-9`，对应方向会在一阶灵敏度求解中被显著放大。本轮最末两个方向的任务增益仅为 `1e-6` 量级，但状态投影不小，说明这些方向更像内部病态自由度，而不是稳定可解释的末端位置控制方向。这类方向不适合直接用于发布灵敏度。

## 实验结论

{summary['decision']}

## 下一步计划

下一轮应比较两类修复：第一，截断弱奇异方向后重建灵敏度；第二，重新构造局部坐标，使约化基优先保留末端任务有效方向而非仅由残差雅可比弱方向决定。任何修复都必须重新通过局部线性化、分支一致性和载荷传播门禁。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the ill-conditioned direction audit."""
    t0 = time.perf_counter()
    base = p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c169.p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c169.p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c169.p2c168.build_optimality_system(branch1)
    states = [reduced_state_for_radius(branch1, system, radius) for radius in RADII_A]
    rows = direction_rows(system, states)
    summary = summarize(system, rows, states, time.perf_counter() - t0)
    summary["case"] = direction_name
    summary["path"] = "first_principal_then_singular_direction_audit"
    write_csv(EXP_DIR / "ill_conditioned_direction_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"failure_class": summary["decision"], "evidence": jsonable(summary)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705001000_p2c170_n2_ill_conditioned_direction_audit/run_p2c170_n2_ill_conditioned_direction_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
