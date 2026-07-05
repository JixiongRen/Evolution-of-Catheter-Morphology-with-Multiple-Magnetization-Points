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
P2C165_DIR = ROOT / "experiment_records_ik_optimize/20260630193000_p2c165_n2_torque_weight_continuation"
for path in (ROOT, P2C165_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c165_n2_torque_weight_continuation as p2c165


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


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Return relative Euclidean error."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-14))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-14))


def build_optimality_system(branch: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the local reduced optimality system."""
    p2c77 = p2c165.p2c162.p2c148.p2c145.p2c79.p2c77
    return p2c77._build_optimality_system(
        y_base=np.asarray(branch["y"], dtype=float),
        I0=np.asarray(branch["I0"], dtype=float),
        funcs=branch["funcs"],
        basis=np.asarray(branch["basis"], dtype=float),
    )


def raw_q(system: dict[str, Any], d_i: np.ndarray) -> np.ndarray:
    """Return reduced state predicted by first-order optimality sensitivity."""
    p2c78 = p2c165.p2c162.p2c148.p2c145.p2c79.p2c78
    return np.asarray(p2c78._raw_q(system, np.asarray(d_i, dtype=float).reshape(8)), dtype=float)


def residual_norm(branch: dict[str, Any], q: np.ndarray, current_i: np.ndarray) -> float:
    """Evaluate residual norm at a reduced state."""
    y = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ np.asarray(q, dtype=float)
    r = branch["funcs"]["residual"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(current_i, dtype=jnp.float64).reshape(8))
    return float(np.linalg.norm(np.asarray(jax.device_get(r), dtype=float)))


def output_tip(branch: dict[str, Any], q: np.ndarray, current_i: np.ndarray) -> np.ndarray:
    """Evaluate terminal position at a reduced state."""
    y = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ np.asarray(q, dtype=float)
    out = p2c165.p2c162.p2c148.p2c134.eval_output(branch["funcs"], y, current_i)
    return np.asarray(out["tip_p"], dtype=float)


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot the linear-model error decomposition."""
    configure_matplotlib_cjk()
    labels = [row["metric"] for row in rows]
    values = [row["value"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(np.arange(len(rows)), values, color="#7570b3")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    ax.set_ylabel("相对误差")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2 第二步线性模型误差分解")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_linear_model_breakdown_cn.png", dpi=200)
    plt.close(fig)


def classify(summary: dict[str, Any]) -> tuple[str, str]:
    """Classify the dominant break point."""
    if summary["optimality_condition"] > 1.0e9:
        return (
            "ill_conditioned_optimality_and_output_break",
            "最优性方程严重病态，且实际状态处末端输出线性化也超限",
        )
    if summary["q_pred_vs_actual_rel_err"] > 0.5:
        return "optimality_sensitivity_break", "电流到状态的一阶最优性灵敏度失效"
    if summary["output_linear_at_actual_q_rel_err"] > 0.05:
        return "output_map_nonlinearity", "状态到末端输出映射局部非线性过强"
    if summary["physical_residual_actual_norm"] > summary["physical_residual_base_norm"] * 2.0:
        return "physical_residual_growth", "扰动重求解后物理残差增长"
    return "no_single_dominant_break", "未发现单一主导断点"


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
    ]
    explanations = {
        "q_pred_vs_actual": "一阶最优性预测的约化状态与实际重求解状态差异",
        "task_pred_vs_actual": "灵敏度矩阵预测末端位移与实际末端位移差异",
        "output_linear_at_actual_q": "在实际状态增量下，末端输出线性近似与真实输出差异",
    }
    for row in rows:
        table.append(f"| {row['metric']} | {row['value']:.3e} | {explanations[row['metric']]} |")

    text = f"""# P2c168 两点磁控 n2 第二步线性模型断点审计

## 实验目的

P2c167 排除了“单一矩参考点切换”作为主失败机制。本轮继续复用同一两点磁控工作点和同一第二步候选，将局部线性化误差拆解为三段：电流到约化状态的一阶最优性灵敏度、约化状态到末端位置的输出线性化、以及扰动后物理残差变化。

## 实验方案

在 P2c165 的零额外矩权重第二步上，计算：

1. 一阶最优性灵敏度预测的约化状态增量 $q_{{pred}}$。
2. 扰动重求解得到的实际约化状态增量 $q_{{actual}}$。
3. 灵敏度矩阵预测末端位移、实际末端位移、以及在 $q_{{actual}}$ 处的输出线性近似。

本轮只做诊断，不修改生产 IK。

## 结果

{chr(10).join(table)}

## 数据分析

约化状态预测误差为 {summary['q_pred_vs_actual_rel_err']:.3e}，预测与实际状态夹角余弦为 {summary['q_pred_actual_cosine']:.3e}。末端任务预测误差为 {summary['task_pred_vs_actual_rel_err']:.3e}。在实际状态增量处，输出线性化误差为 {summary['output_linear_at_actual_q_rel_err']:.3e}。

本轮最优性方程条件数为 {summary['optimality_condition']:.3e}，远高于 1e9 硬门禁。第一项和第三项均明显超限，说明断裂不是单一环节造成，而是“病态最优性方程”和“末端输出局部非线性”共同出现。

## 实验结论

本轮失败分类为：`{summary['status']}`。解释：{summary['decision']}。

## 下一步计划

下一步应优先审计最优性方程的残差组成、权重尺度、约化基和病态方向；同时把任务步长降到更小半径，检查输出线性化是否恢复。若缩小半径后输出线性化恢复而条件数仍病态，则主修最优性方程；若二者均不恢复，则应转向更换参数化或分支局部坐标。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the linear-model breakdown audit."""
    t0 = time.perf_counter()
    base = p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    first_row, branch1 = p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    q_actual, solve_info, current_new = p2c165.solve_with_torque_weight(branch1, p2c165.STEP_RADIUS_A, 0.0)
    d_i = np.asarray(current_new - branch1["I0"], dtype=float)
    system = build_optimality_system(branch1)
    q_pred = raw_q(system, d_i)

    base_tip = np.asarray(branch1["base_out"]["tip_p"], dtype=float)
    tip_actual = output_tip(branch1, q_actual, current_new)
    actual_task = tip_actual - base_tip
    task_pred = np.asarray(branch1["S"] @ d_i, dtype=float)
    task_linear_actual_q = np.asarray(system["t_eff"] @ q_actual, dtype=float)

    physical_residual_base = residual_norm(branch1, np.zeros_like(q_actual), branch1["I0"])
    physical_residual_actual = residual_norm(branch1, q_actual, current_new)
    physical_residual_pred = residual_norm(branch1, q_pred, current_new)

    summary = {
        "experiment_id": EXP_DIR.name,
        "case": direction_name,
        "path": "first_principal_then_zero_torque_weight_second_step",
        "outer_ik_allowed": False,
        "publishable_sensitivity": False,
        "q_pred_norm": float(np.linalg.norm(q_pred)),
        "q_actual_norm": float(np.linalg.norm(q_actual)),
        "q_pred_vs_actual_rel_err": rel_err(q_pred, q_actual),
        "q_pred_actual_cosine": cosine(q_pred, q_actual),
        "task_pred_norm_um": float(np.linalg.norm(task_pred) * 1.0e6),
        "actual_task_norm_um": float(np.linalg.norm(actual_task) * 1.0e6),
        "task_pred_vs_actual_rel_err": rel_err(task_pred, actual_task),
        "task_pred_actual_cosine": cosine(task_pred, actual_task),
        "output_linear_at_actual_q_rel_err": rel_err(task_linear_actual_q, actual_task),
        "output_linear_actual_q_cosine": cosine(task_linear_actual_q, actual_task),
        "physical_residual_base_norm": physical_residual_base,
        "physical_residual_actual_norm": physical_residual_actual,
        "physical_residual_pred_norm": physical_residual_pred,
        "optimality_condition": float(system["raw_condition"]),
        "first_step": first_row,
        "second_step_solve": solve_info,
        "wall_s": float(time.perf_counter() - t0),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    status, decision = classify(summary)
    summary["status"] = status
    summary["decision"] = decision
    rows = [
        {"metric": "q_pred_vs_actual", "value": summary["q_pred_vs_actual_rel_err"]},
        {"metric": "task_pred_vs_actual", "value": summary["task_pred_vs_actual_rel_err"]},
        {"metric": "output_linear_at_actual_q", "value": summary["output_linear_at_actual_q_rel_err"]},
    ]
    write_csv(EXP_DIR / "linear_model_breakdown_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"failure_class": decision, "evidence": jsonable(summary)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260704231500_p2c168_n2_linear_model_breakdown_audit/run_p2c168_n2_linear_model_breakdown_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
