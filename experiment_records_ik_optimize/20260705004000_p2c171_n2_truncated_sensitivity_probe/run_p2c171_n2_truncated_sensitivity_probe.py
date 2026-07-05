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
P2C170_DIR = ROOT / "experiment_records_ik_optimize/20260705001000_p2c170_n2_ill_conditioned_direction_audit"
for path in (ROOT, P2C170_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c170_n2_ill_conditioned_direction_audit as p2c170


TRUNCATION_LEVELS = (0.0, 1.0e-12, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-6)
RADIUS_A = 1.0e-4
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


def raw_q_truncated(system: dict[str, Any], d_i: np.ndarray, rel_cutoff: float) -> np.ndarray:
    """Return sensitivity-predicted reduced state with small singular directions removed."""
    s = np.asarray(system["s"], dtype=float)
    keep = s / max(float(s[0]), np.finfo(float).tiny) >= float(rel_cutoff)
    if not np.any(keep):
        keep[0] = True
    rhs = system["g_I"] @ np.asarray(d_i, dtype=float).reshape(8)
    coeff = (system["u"].T @ rhs)[keep] / s[keep]
    return -(system["vt"][keep, :].T @ coeff)


def sensitivity_truncated(system: dict[str, Any], rel_cutoff: float) -> np.ndarray:
    """Build task sensitivity after singular-direction truncation."""
    cols = []
    for i in range(8):
        d_i = np.zeros(8)
        d_i[i] = 1.0
        cols.append(system["t_eff"] @ raw_q_truncated(system, d_i, rel_cutoff))
    return np.column_stack(cols)


def current_step_for_direction(sensitivity: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    """Return a current step for a fixed task direction."""
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-18)
    d_i = np.linalg.pinv(sensitivity, rcond=1.0e-12) @ direction
    d_i /= max(float(np.linalg.norm(d_i)), 1.0e-18)
    return float(radius) * d_i


def evaluate_cutoff(branch: dict[str, Any], system: dict[str, Any], rel_cutoff: float) -> dict[str, Any]:
    """Evaluate one truncation cutoff with re-optimized state solve."""
    sensitivity = sensitivity_truncated(system, rel_cutoff)
    u, singular_values, _vt = np.linalg.svd(sensitivity, full_matrices=False)
    task_direction = np.asarray(branch["U"][:, 0], dtype=float)
    d_i = current_step_for_direction(sensitivity, task_direction, RADIUS_A)
    current_new = np.asarray(branch["I0"], dtype=float) + d_i
    q_pred = raw_q_truncated(system, d_i, rel_cutoff)
    q_actual, info = p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=np.asarray(branch["y"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        I=current_new,
        funcs=branch["funcs"],
    )
    y_new = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ q_actual
    out_new = p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.eval_output(branch["funcs"], y_new, current_new)
    base_tip = np.asarray(branch["base_out"]["tip_p"], dtype=float)
    actual = np.asarray(out_new["tip_p"], dtype=float) - base_tip
    pred = sensitivity @ d_i
    current_blocks = p2c170.p2c169.p2c168.p2c165.current_aware_interface_blocks(branch, y_new, current_new)
    current_metrics = p2c170.p2c169.p2c168.p2c165.metric_from_blocks(current_blocks)
    rel_q = p2c170.p2c169.p2c168.rel_err(q_pred, q_actual)
    rel_task = p2c170.p2c169.p2c168.rel_err(pred, actual)
    output_linear = p2c170.p2c169.p2c168.rel_err(system["t_eff"] @ q_actual, actual)
    kept = np.asarray(system["s"], dtype=float) / max(float(system["s"][0]), np.finfo(float).tiny) >= float(rel_cutoff)
    if not np.any(kept):
        kept[0] = True
    return {
        "rel_cutoff": float(rel_cutoff),
        "kept_direction_count": int(np.sum(kept)),
        "removed_direction_count": int(len(kept) - np.sum(kept)),
        "sensitivity_rank": int(np.linalg.matrix_rank(sensitivity, tol=1.0e-12)),
        "sensitivity_condition": float(singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny)),
        "task_pred_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_task_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "q_pred_vs_actual_rel_err": rel_q,
        "task_pred_vs_actual_rel_err": rel_task,
        "output_linear_at_actual_q_rel_err": output_linear,
        "current_aware_continuity_norm": current_metrics["continuity_norm"],
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
        "optimality": float(info["optimality"]),
        "task_linear_pass": bool(rel_task < LINEAR_GATE),
        "q_linear_pass": bool(rel_q < LINEAR_GATE),
        "output_linear_pass": bool(output_linear < LINEAR_GATE),
    }


def summarize(rows: list[dict[str, Any]], wall_s: float, raw_condition: float) -> dict[str, Any]:
    """Summarize truncation probe rows."""
    best = min(rows, key=lambda row: row["task_pred_vs_actual_rel_err"])
    passed = [row for row in rows if row["task_linear_pass"] and row["output_linear_pass"] and row["q_linear_pass"]]
    if passed and best["rel_cutoff"] == 0.0:
        status = "exact_jacobian_reopt_candidate"
        decision = "不截断且使用精确雅可比重求解时通过线性化门禁；前序失败主要嫌疑转为扰动重求解路径。"
    elif passed:
        status = "truncation_publishable_candidate"
        decision = "存在截断候选通过线性化门禁；下一步需复核分支一致性和载荷传播。"
    elif best["task_pred_vs_actual_rel_err"] < 0.2:
        status = "truncation_partial_improvement"
        decision = "截断弱方向带来部分改善，但仍未达到发布门禁。"
    else:
        status = "truncation_not_sufficient"
        decision = "简单截断弱奇异方向不足以获得稳定灵敏度；需重建局部坐标或任务参数化。"
    return {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": decision,
        "radius_A": RADIUS_A,
        "raw_optimality_condition": float(raw_condition),
        "tested_count": len(rows),
        "pass_count": len(passed),
        "best_cutoff": best["rel_cutoff"],
        "best_task_pred_vs_actual_rel_err": best["task_pred_vs_actual_rel_err"],
        "best_output_linear_at_actual_q_rel_err": best["output_linear_at_actual_q_rel_err"],
        "best_q_pred_vs_actual_rel_err": best["q_pred_vs_actual_rel_err"],
        "outer_ik_allowed": False,
        "publishable_sensitivity": bool(passed),
        "wall_s": float(wall_s),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot truncation probe errors."""
    configure_matplotlib_cjk()
    labels = ["0" if row["rel_cutoff"] == 0 else f"{row['rel_cutoff']:.0e}" for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(x, [row["q_pred_vs_actual_rel_err"] for row in rows], marker="o", label="状态预测误差")
    ax.plot(x, [row["task_pred_vs_actual_rel_err"] for row in rows], marker="s", label="任务预测误差")
    ax.plot(x, [row["output_linear_at_actual_q_rel_err"] for row in rows], marker="^", label="输出线性化误差")
    ax.axhline(LINEAR_GATE, color="black", linestyle="--", linewidth=1, label="门限")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel("相对奇异值截断阈值")
    ax.set_ylabel("相对误差")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle("n2 弱奇异方向截断灵敏度探针")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_truncated_sensitivity_probe_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 截断阈值 | 保留方向 | 任务预测误差 | 输出线性化误差 | 状态预测误差 | 连续残差 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        table.append(
            f"| {row['rel_cutoff']:.0e} | {row['kept_direction_count']} | "
            f"{row['task_pred_vs_actual_rel_err']:.3e} | {row['output_linear_at_actual_q_rel_err']:.3e} | "
            f"{row['q_pred_vs_actual_rel_err']:.3e} | {row['current_aware_continuity_norm']:.3e} |"
        )
    text = f"""# P2c171 两点磁控 n2 弱奇异方向截断灵敏度探针

## 实验目的

P2c170 指出约化最优性系统含多个弱奇异方向。本轮在不修改生产 IK 的前提下，对一阶灵敏度方程做奇异方向截断，判断简单截断是否足以恢复局部线性化。

## 实验方案

固定同一工作点、同一分支和同一原始任务方向。第二步电流半径固定为 `{summary['radius_A']:.1e}` A。对最优性方程奇异值设置多个相对截断阈值，分别计算截断后的任务灵敏度，并用扰动重求解检验预测。

## 结果

{chr(10).join(table)}

## 数据分析

原始最优性方程条件数为 {summary['raw_optimality_condition']:.3e}。最佳截断阈值为 `{summary['best_cutoff']:.0e}`，任务预测误差为 {summary['best_task_pred_vs_actual_rel_err']:.3e}，输出线性化误差为 {summary['best_output_linear_at_actual_q_rel_err']:.3e}，状态预测误差为 {summary['best_q_pred_vs_actual_rel_err']:.3e}。

## 数据解释

最佳候选为未截断情形，这说明本轮改善并不是由截断弱奇异方向带来，而更可能来自扰动重求解器路径差异：本轮使用精确雅可比约化重求解，而 P2c169 使用的是 P2c165 的二点差分重求解路径。该发现需要单独复核，不能直接宣布灵敏度可发布。

## 实验结论

{summary['decision']}

## 下一步计划

下一轮应直接比较两条扰动重求解路径：精确雅可比约化重求解与二点差分重求解。若前者稳定通过而后者失败，则主问题转为工程求解器实现；若两者在更多工况下都不稳定，再回到局部坐标重建。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
- JAX 运行设备：`{summary['jax_devices'][0] if summary['jax_devices'] else 'unknown'}`。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the truncated sensitivity probe."""
    t0 = time.perf_counter()
    base = p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    _first_row, branch1 = p2c170.p2c169.p2c168.p2c165.p2c162.run_step(branch0, step_index=1, mode_index=0)
    system = p2c170.p2c169.p2c168.build_optimality_system(branch1)
    rows = [evaluate_cutoff(branch1, system, cutoff) for cutoff in TRUNCATION_LEVELS]
    summary = summarize(rows, time.perf_counter() - t0, system["raw_condition"])
    summary["case"] = direction_name
    summary["path"] = "first_principal_then_truncated_sensitivity_probe"
    write_csv(EXP_DIR / "truncated_sensitivity_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"failure_class": summary["decision"], "evidence": jsonable(summary)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705004000_p2c171_n2_truncated_sensitivity_probe/run_p2c171_n2_truncated_sensitivity_probe.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
