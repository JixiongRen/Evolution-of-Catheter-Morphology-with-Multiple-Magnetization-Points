"""Instrument strict n2 one-dimensional branch-loop timing."""

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
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C174_DIR = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"
P2C186_DIR = ROOT / "experiment_records_ik_optimize/20260705043000_p2c186_n2_independent_cache_replay"
for path in (ROOT, P2C174_DIR, P2C186_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402

import run_p2c174_n2_1d_branch_loop_gate as p2c174  # noqa: E402
import run_p2c186_n2_independent_cache_replay as p2c186  # noqa: E402


WORKPOINT = "rand_05"
MODE_INDEX = 0
RADIUS_A = 1.0e-4
MAX_NFEV = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.p2c162.p2c148.p2c134.MAX_NFEV


def jsonable(value: Any) -> Any:
    """Convert nested values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def rel_err(pred: np.ndarray, actual: np.ndarray) -> float:
    """Return relative error with the same convention as earlier gates."""
    return p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.rel_err(pred, actual)


def solve_reduced_instrumented(
    *,
    q0: np.ndarray,
    y_base: np.ndarray,
    basis: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the reduced state and expose timing inside the residual path."""
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    calls = {"residual": 0, "jacobian": 0}

    def residual(q: np.ndarray) -> np.ndarray:
        calls["residual"] += 1
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        return np.asarray(jax.device_get(funcs["residual"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)

    def jacobian(q: np.ndarray) -> np.ndarray:
        calls["jacobian"] += 1
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)
        return J @ basis

    t_first = time.perf_counter()
    _ = residual(q0)
    first_residual_s = time.perf_counter() - t_first

    t_ls = time.perf_counter()
    opt = least_squares(
        residual,
        q0,
        jac=jacobian,
        method="trf",
        x_scale=np.ones_like(q0),
        loss="linear",
        max_nfev=MAX_NFEV,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    least_squares_s = time.perf_counter() - t_ls
    return np.asarray(opt.x, dtype=float), {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(least_squares_s),
        "first_residual_s": float(first_residual_s),
        "total_s": float(first_residual_s + least_squares_s),
        "residual_calls": int(calls["residual"]),
        "jacobian_calls": int(calls["jacobian"]),
        "cost": float(opt.cost),
        "optimality": float(opt.optimality),
        "message": str(opt.message),
    }


def output_timed(branch: dict[str, Any], q: np.ndarray, current: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """Evaluate output and return elapsed time."""
    y = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ np.asarray(q, dtype=float)
    t0 = time.perf_counter()
    out = p2c174.output(branch, q, current)
    return out, time.perf_counter() - t0


def continuity_timed(branch: dict[str, Any], q: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Evaluate current-aware interface continuity and return elapsed time."""
    y = np.asarray(branch["y"], dtype=float) + np.asarray(branch["basis"], dtype=float) @ np.asarray(q, dtype=float)
    t0 = time.perf_counter()
    blocks = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.current_aware_interface_blocks(
        branch,
        y,
        current,
    )
    continuity = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.p2c165.metric_from_blocks(blocks)["continuity_norm"]
    return float(continuity), time.perf_counter() - t0


def solve_from_q0_instrumented(
    branch: dict[str, Any],
    current: np.ndarray,
    q0: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one reduced state from a selected initial offset."""
    return solve_reduced_instrumented(
        q0=np.asarray(q0, dtype=float),
        y_base=np.asarray(branch["y"], dtype=float),
        basis=np.asarray(branch["basis"], dtype=float),
        I=np.asarray(current, dtype=float),
        funcs=branch["funcs"],
    )


def line_metrics_instrumented(
    branch: dict[str, Any],
    system: dict[str, Any],
    sign: float,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run signed perturbation with detailed timing attribution."""
    t_step = time.perf_counter()
    d_i, current_new, q_pred = p2c174.build_step(branch, system, sign)
    step_build_s = time.perf_counter() - t_step
    starts = {
        "zero": np.zeros_like(q_pred),
        "pred": q_pred,
        "neg_pred": -q_pred,
        "half_pred": 0.5 * q_pred,
    }
    solutions: list[dict[str, Any]] = []
    base_tip = np.asarray(branch["base_out"]["tip_p"], dtype=float)
    base_centerline = np.asarray(branch["base_out"]["centerline"], dtype=float)
    pred_task = p2c174.p2c173.p2c171.sensitivity_truncated(system, 0.0) @ d_i
    for start_name, q0 in starts.items():
        t_row = time.perf_counter()
        q, info = solve_from_q0_instrumented(branch, current_new, q0)
        out, output_eval_s = output_timed(branch, q, current_new)
        actual = np.asarray(out["tip_p"], dtype=float) - base_tip
        output_linear = np.asarray(system["t_eff"] @ q, dtype=float)
        continuity, continuity_eval_s = continuity_timed(branch, q, current_new)
        q_return, return_info = solve_from_q0_instrumented(branch, branch["I0"], q)
        row_total_s = time.perf_counter() - t_row
        row = {
            "direction": label,
            "start": start_name,
            "step_build_s": float(step_build_s),
            "q_norm": float(np.linalg.norm(q)),
            "q_pred_vs_actual_rel_err": rel_err(q_pred, q),
            "task_pred_vs_actual_rel_err": rel_err(pred_task, actual),
            "output_linear_at_actual_q_rel_err": rel_err(output_linear, actual),
            "centerline_max_change_m": float(np.max(np.linalg.norm(np.asarray(out["centerline"], dtype=float) - base_centerline, axis=1))),
            "current_aware_continuity_norm": float(continuity),
            "output_eval_s": float(output_eval_s),
            "continuity_eval_s": float(continuity_eval_s),
            "row_total_s": float(row_total_s),
            "optimality": float(info["optimality"]),
            "solve_success": bool(info["success"]),
            "solve_nfev": int(info["nfev"]),
            "solve_first_residual_s": float(info["first_residual_s"]),
            "solve_steady_s": float(info["steady_s"]),
            "solve_total_s": float(info["total_s"]),
            "solve_residual_calls": int(info["residual_calls"]),
            "solve_jacobian_calls": int(info["jacobian_calls"]),
            "return_q_norm": float(np.linalg.norm(q_return)),
            "return_success": bool(return_info["success"]),
            "return_optimality": float(return_info["optimality"]),
            "return_nfev": int(return_info["nfev"]),
            "return_first_residual_s": float(return_info["first_residual_s"]),
            "return_steady_s": float(return_info["steady_s"]),
            "return_total_s": float(return_info["total_s"]),
            "return_residual_calls": int(return_info["residual_calls"]),
            "return_jacobian_calls": int(return_info["jacobian_calls"]),
        }
        row["gate_pass"] = bool(
            row["solve_success"]
            and row["return_success"]
            and row["optimality"] <= p2c174.OPTIMALITY_GATE
            and row["current_aware_continuity_norm"] <= p2c174.CONTINUITY_GATE
            and row["centerline_max_change_m"] <= p2c174.CENTERLINE_GATE_M
            and row["q_pred_vs_actual_rel_err"] <= p2c174.LINEAR_GATE
            and row["task_pred_vs_actual_rel_err"] <= p2c174.LINEAR_GATE
            and row["output_linear_at_actual_q_rel_err"] <= p2c174.LINEAR_GATE
            and row["return_q_norm"] <= p2c174.RETURN_Q_GATE
        )
        row["q_solution"] = q
        solutions.append(row)
    q_ref = solutions[0]["q_solution"]
    for row in solutions:
        row["q_consistency_to_zero_start"] = float(np.linalg.norm(np.asarray(row["q_solution"], dtype=float) - q_ref))
        row["gate_pass"] = bool(row["gate_pass"] and row["q_consistency_to_zero_start"] <= p2c174.CONSISTENCY_Q_GATE)
    public_rows = [{k: v for k, v in row.items() if k != "q_solution"} for row in solutions]
    direction_summary = {
        "direction": label,
        "tested_count": len(public_rows),
        "pass_count": sum(1 for row in public_rows if row["gate_pass"]),
        "max_q_consistency": max(row["q_consistency_to_zero_start"] for row in public_rows),
        "max_return_q_norm": max(row["return_q_norm"] for row in public_rows),
        "max_task_error": max(row["task_pred_vs_actual_rel_err"] for row in public_rows),
        "max_output_error": max(row["output_linear_at_actual_q_rel_err"] for row in public_rows),
        "row_total_s": float(sum(row["row_total_s"] for row in public_rows)),
        "solve_total_s": float(sum(row["solve_total_s"] + row["return_total_s"] for row in public_rows)),
    }
    return direction_summary, public_rows


def timing_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate row timing components."""
    solve_first = sum(row["solve_first_residual_s"] + row["return_first_residual_s"] for row in rows)
    solve_ls = sum(row["solve_steady_s"] + row["return_steady_s"] for row in rows)
    return {
        "step_build_s": float(sum(row["step_build_s"] for row in rows)),
        "solve_first_residual_s": float(solve_first),
        "solve_least_squares_s": float(solve_ls),
        "solve_total_s": float(solve_first + solve_ls),
        "output_eval_s": float(sum(row["output_eval_s"] for row in rows)),
        "continuity_eval_s": float(sum(row["continuity_eval_s"] for row in rows)),
        "row_total_s": float(sum(row["row_total_s"] for row in rows)),
    }


def summarize(
    rows: list[dict[str, Any]],
    direction_summaries: list[dict[str, Any]],
    load_rebuild_s: float,
    system_build_s: float,
    gate_wall_s: float,
    wall_s: float,
) -> dict[str, Any]:
    """Summarize instrumented gate results."""
    totals = timing_totals(rows)
    pass_count = sum(1 for row in rows if row["gate_pass"])
    status = "instrumented_gate_passed" if pass_count == len(rows) else "instrumented_gate_failed"
    known_inside_gate = (
        totals["solve_total_s"]
        + totals["output_eval_s"]
        + totals["continuity_eval_s"]
        + totals["step_build_s"]
    )
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": status,
        "decision": (
            "单工作点严格门禁通过；计时结果可用于下一轮减少外部验证开销。"
            if pass_count == len(rows)
            else "单工作点严格门禁未全部通过；需先恢复可信性再做提速。"
        ),
        "workpoint": WORKPOINT,
        "mode_index": MODE_INDEX + 1,
        "radius_A": RADIUS_A,
        "tested_count": len(rows),
        "pass_count": pass_count,
        "load_rebuild_s": float(load_rebuild_s),
        "system_build_s": float(system_build_s),
        "gate_wall_s": float(gate_wall_s),
        "wall_s": float(wall_s),
        "timing_totals": totals,
        "known_inside_gate_s": float(known_inside_gate),
        "unattributed_gate_s": float(max(gate_wall_s - known_inside_gate, 0.0)),
        "solve_hidden_ratio": float(1.0 - totals["solve_total_s"] / gate_wall_s) if gate_wall_s > 0 else None,
        "direction_summaries": direction_summaries,
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    return summary


def plot(summary: dict[str, Any]) -> None:
    """Plot timing attribution and gate metrics."""
    configure_matplotlib_cjk()
    totals = summary["timing_totals"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    labels = ["首次残差", "最小二乘", "输出计算", "连续性计算", "步长构造", "未归因"]
    values = [
        totals["solve_first_residual_s"],
        totals["solve_least_squares_s"],
        totals["output_eval_s"],
        totals["continuity_eval_s"],
        totals["step_build_s"],
        summary["unattributed_gate_s"],
    ]
    axes[0].bar(labels, values, color=["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#8ab17d", "#e76f51"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("耗时/s")
    axes[0].set_title("门禁耗时拆解")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    row_labels = [f"{row['direction']}-{row['start']}" for row in summary["rows"]]
    x = np.arange(len(row_labels))
    axes[1].bar(x, [row["task_pred_vs_actual_rel_err"] for row in summary["rows"]], color="#2a9d8f")
    axes[1].axhline(p2c174.LINEAR_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, row_labels, rotation=30, ha="right")
    axes[1].set_ylabel("任务线性化误差")
    axes[1].set_title("严格门禁质量")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c189 两点磁化一维灵敏度耗时插桩")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "p2c189_instrumented_timing_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    totals = summary["timing_totals"]
    rows_table = [
        "| 方向 | 初值 | 任务误差 | 输出误差 | 回环误差 | 求解总耗时/s | 输出/s | 连续性/s | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        rows_table.append(
            f"| {row['direction']} | {row['start']} | {row['task_pred_vs_actual_rel_err']:.3e} | "
            f"{row['output_linear_at_actual_q_rel_err']:.3e} | {row['return_q_norm']:.3e} | "
            f"{row['solve_total_s'] + row['return_total_s']:.3f} | {row['output_eval_s']:.3f} | "
            f"{row['continuity_eval_s']:.3f} | {'通过' if row['gate_pass'] else '不可发布'} |"
        )
    text = f"""# P2c189 两点磁控一维灵敏度门禁耗时插桩

## 实验目的

P2c187 表明两点磁控第 1 主方向在多工作点上可通过严格分支回环门禁，但 P2c188 发现门禁墙钟耗时中大部分不属于最小二乘迭代本身。本轮选取 `rand_05` 单工作点，对严格门禁内部做计时拆解，定位后续提速对象。

## 实验方案

从 P2c186 的独立缓存加载 `rand_05` 分支，重建物理函数。保持 P2c174 的正负扰动、四类初值、回环求解和全部硬门禁不变。新增计时项包括：首次残差调用、最小二乘迭代、输出计算、接口连续性计算、扰动步长构造和未归因时间。

## 结果

{chr(10).join(rows_table)}

## 耗时拆解

- 总墙钟耗时：{summary['wall_s']:.2f} s。
- 缓存加载与函数重建：{summary['load_rebuild_s']:.2f} s。
- 局部线性系统构建：{summary['system_build_s']:.2f} s。
- 严格门禁墙钟耗时：{summary['gate_wall_s']:.2f} s。
- 首次残差总耗时：{totals['solve_first_residual_s']:.2f} s。
- 最小二乘迭代总耗时：{totals['solve_least_squares_s']:.2f} s。
- 输出计算总耗时：{totals['output_eval_s']:.2f} s。
- 接口连续性计算总耗时：{totals['continuity_eval_s']:.2f} s。
- 未归因门禁耗时：{summary['unattributed_gate_s']:.2f} s。
- 求解外部耗时比例：{summary['solve_hidden_ratio']:.2%}。
- 运行设备：{', '.join(summary['jax_devices'])}。

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。若门禁全部通过，则本轮不改变可信性结论；耗时拆解只用于后续工程优化。若最小二乘迭代占比很小，下一轮应优先合并输出计算、连续性计算和回环验证，而不是继续调节求解器容差。

## 实验结论

{summary['decision']}

## 下一步计划

优先将输出计算与连续性计算批量化，并评估是否可用单次扰动解同时生成回环初值与诊断量。任何提速方案必须保持 P2c174 的硬门禁不变。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def write_commands() -> None:
    """Write the reproducible command."""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c189_instrumented_solve_timing_single_workpoint.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


def main() -> None:
    """Run the instrumented single-workpoint gate."""
    t_wall = time.perf_counter()
    write_commands()
    branch, load_rebuild_s = p2c186.load_branch(WORKPOINT)
    old_radius = p2c174.RADIUS_A
    old_mode = p2c174.MODE_INDEX
    p2c174.RADIUS_A = RADIUS_A
    p2c174.MODE_INDEX = MODE_INDEX
    try:
        t_system = time.perf_counter()
        system = p2c174.p2c173.p2c171.p2c170.p2c169.p2c168.build_optimality_system(branch)
        system_build_s = time.perf_counter() - t_system
        t_gate = time.perf_counter()
        direction_summaries: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for sign, label in ((1.0, "正向"), (-1.0, "反向")):
            direction_summary, direction_rows = line_metrics_instrumented(branch, system, sign, label)
            direction_summaries.append(direction_summary)
            rows.extend(direction_rows)
        gate_wall_s = time.perf_counter() - t_gate
    finally:
        p2c174.RADIUS_A = old_radius
        p2c174.MODE_INDEX = old_mode

    summary = summarize(
        rows,
        direction_summaries,
        load_rebuild_s,
        system_build_s,
        gate_wall_s,
        time.perf_counter() - t_wall,
    )
    write_csv(EXP_DIR / "instrumented_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["gate_pass"]:
                f.write(json.dumps(jsonable({"failure_class": "插桩门禁失败", "evidence": row}), ensure_ascii=False) + "\n")
    plot(summary)
    write_record(summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
