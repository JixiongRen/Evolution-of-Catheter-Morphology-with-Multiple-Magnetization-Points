"""Audit n2 current perturbation reoptimization inside the trusted state basin."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C123_DIR = ROOT / "experiment_records_ik_optimize" / "20260629110000_p2c123_n2_direction_reopt_closure"
P2C134_DIR = ROOT / "experiment_records_ik_optimize" / "20260630023000_p2c134_n2_trusted_branch_basin_radius"
for path in (ROOT, P2C123_DIR, P2C134_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import numpy as np

import run_p2c123_n2_direction_reopt_closure as p2c123
import run_p2c134_n2_trusted_branch_basin_radius as p2c134


OUT_DIR = Path(__file__).resolve().parent
CURRENT_RADII_A = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3)
STATE_RADIUS_GATE = 1.0e-4
LINEAR_REL_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
MIN_RELIABLE_DISPLACEMENT_UM = 1.0e-4


def solve_for_current(
    *,
    I_new: np.ndarray,
    y_base: np.ndarray,
    basis: np.ndarray,
    funcs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve reduced state for a perturbed current."""
    return p2c134.solve_reduced(q0=np.zeros(basis.shape[1]), y_base=y_base, basis=basis, I=I_new, funcs=funcs)


def run_case(
    *,
    name: str,
    task_direction: np.ndarray,
    radius: float,
    S: np.ndarray,
    I0: np.ndarray,
    y_base: np.ndarray,
    basis: np.ndarray,
    funcs: dict[str, Any],
    params: Any,
    base_out: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run one trust-region current perturbation case."""
    dI = p2c123.current_step_for_task_direction(S, task_direction, radius)
    pred = S @ dI
    q, info = solve_for_current(I_new=I0 + dI, y_base=y_base, basis=basis, funcs=funcs)
    y_new = y_base + basis @ q
    out = p2c134.eval_output(funcs, y_new, I0 + dI)
    actual = out["tip_p"] - base_out["tip_p"]
    rel, pred_actual_cos = p2c123.relative_error(pred, actual)
    task_actual_cos = p2c123.direction_cos(task_direction, actual)
    centerline_change = float(np.max(np.linalg.norm(out["centerline"] - base_out["centerline"], axis=1)))
    continuity, force_ratio, torque_ratio = p2c134.load_metrics(params, y_new)
    state_radius = float(np.linalg.norm(q))
    passed = bool(
        info["success"]
        and rel < LINEAR_REL_GATE
        and centerline_change < CENTERLINE_GATE_M
        and state_radius <= STATE_RADIUS_GATE
        and continuity <= 1.0e-3
        and force_ratio <= 1.5
    )
    failure = "通过"
    if not passed:
        if not info["success"]:
            failure = "重求解未收敛"
        elif state_radius > STATE_RADIUS_GATE:
            failure = "状态增量超过分支信赖域"
        elif max(float(np.linalg.norm(pred) * 1.0e6), float(np.linalg.norm(actual) * 1.0e6)) < MIN_RELIABLE_DISPLACEMENT_UM:
            failure = "末端位移低于可靠判读阈值"
        elif rel >= LINEAR_REL_GATE:
            failure = "局部线性化误差超限"
        elif force_ratio > 1.5:
            failure = "载荷传播失配"
        elif continuity > 1.0e-3:
            failure = "接口连续残差超限"
        else:
            failure = "综合失败"
    return {
        "direction": name,
        "radius_A": float(radius),
        "status": "pass" if passed else "fail",
        "failure_class": failure,
        "state_radius": state_radius,
        "linear_rel_err": rel,
        "pred_actual_cosine": pred_actual_cos,
        "task_actual_cosine": task_actual_cos,
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "centerline_max_change_m": centerline_change,
        "continuity_norm": continuity,
        "max_interface_force_jump_over_rigid_force": force_ratio,
        "max_interface_torque_jump_over_rigid_torque": torque_ratio,
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot current trust-region closure metrics."""
    radii = sorted({row["radius_A"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for direction in sorted({row["direction"] for row in rows}):
        group = [row for row in rows if row["direction"] == direction]
        group.sort(key=lambda item: item["radius_A"])
        x = [row["radius_A"] for row in group]
        axes[0, 0].plot(x, [row["linear_rel_err"] for row in group], marker="o", label=direction)
        axes[0, 1].plot(x, [row["state_radius"] for row in group], marker="o")
        axes[1, 0].plot(x, [row["centerline_max_change_m"] for row in group], marker="o")
        axes[1, 1].plot(x, [row["max_interface_force_jump_over_rigid_force"] for row in group], marker="o")
    axes[0, 0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(STATE_RADIUS_GATE, color="black", linestyle="--", linewidth=1)
    axes[1, 0].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[1, 1].axhline(1.5, color="black", linestyle="--", linewidth=1)
    titles = ["局部线性化误差", "有效状态增量", "中心线最大变化", "载荷传播比例"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("电流扰动半径/A")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT_DIR / "n2_trust_region_current_reopt_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    radius_lines = []
    for radius in sorted({row["radius_A"] for row in rows}):
        group = [row for row in rows if row["radius_A"] == radius]
        radius_lines.append(
            f"- 电流半径 {radius:.0e} A：通过 {sum(row['status'] == 'pass' for row in group)}/{len(group)}；"
            f"最大线性化误差 {max(row['linear_rel_err'] for row in group):.3e}；"
            f"最大状态增量 {max(row['state_radius'] for row in group):.3e}；"
            f"最大载荷传播比例 {max(row['max_interface_force_jump_over_rigid_force'] for row in group):.3e}。"
        )
    low_signal = [row for row in rows if row["failure_class"] == "末端位移低于可靠判读阈值"]
    failures = [row for row in rows if row["status"] != "pass"]
    failure_lines = "\n".join(
        f"- {row['direction']}，半径 {row['radius_A']:.0e} A：{row['failure_class']}；线性误差 {row['linear_rel_err']:.3e}；状态增量 {row['state_radius']:.3e}。"
        for row in failures
    ) or "- 无。"
    record = f"""# P2c136 两点磁控信赖域内电流扰动重求解

## 实验目的

P2c134 与 P2c135 给出两点磁控缓存分支的保守状态信赖域：有效子空间半径 `1e-4` 内稳定，`2e-4` 起存在方向相关失败。本轮在电流扰动后真实重求解状态，并要求重求解状态增量仍处于 `1e-4` 信赖域内，从而检验三维末端位置局部灵敏度是否可在线性近似下使用。

## 实验方案

固定 P2c104 两点磁控缓存分支、同一基准电流和同一任务灵敏度矩阵。选取三个主灵敏方向和三个笛卡尔方向，电流扰动半径为 `1e-5, 3e-5, 1e-4, 3e-4, 1e-3 A`。每个扰动均从基准分支零有效子空间初值重求解。

通过门禁包括：重求解收敛、局部线性化相对误差小于 `0.05`、中心线变化小于 `1e-3 m`、有效状态增量不超过 `1e-4`、接口连续残差不超过 `1e-3`、载荷传播比例不超过 `1.5`。

## 结果

{chr(10).join(radius_lines)}

## 失败归因

{failure_lines}

## 结论

最大所有方向通过电流半径：`{summary['largest_all_direction_pass_radius_A']}`。最大任一方向通过电流半径：`{summary['largest_any_direction_pass_radius_A']}`。

本轮只在状态信赖域内评估局部线性化。结果显示，第一主灵敏方向在 `1e-4 A` 内同时满足线性误差、状态半径、中心线漂移和载荷传播门禁；但三维全方向没有共同通过半径。弱主方向、y 方向以及部分 z 方向的末端位移幅值低于 `1e-4` 微米或接近该量级时，相对误差会被数值分辨率放大，因此不宜把这些方向解释为稳定可发布的三维灵敏度。

因此，当前 n2 结果更适合发布为“分支条件下的优势方向局部导数”和“弱方向不可可靠辨识”，不支持发布完整三维末端位置灵敏度。失败主因不是载荷传播失配，而是有效驱动秩和方向增益高度各向异性导致弱方向信号不足。

## 下一步计划

下一步应生成 n2 有效子空间发布候选包：只保留第一主灵敏方向或信号幅值超过可靠阈值的方向；对弱方向给出不可辨识结论。若仍需三维位置控制，应更换工作点、磁化配置或电流基点，而不是继续在当前点强行发布三维灵敏度。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the current perturbation audit."""
    t0 = time.perf_counter()
    p2c134.OUT_DIR = OUT_DIR
    params, I0 = p2c134.p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c134.p2c104.p2c79.p2c57.CASE_NAME,
        p2c134.p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c134.p2c104._load_cached_branch(params)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    S = np.asarray(branch["S"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = p2c134.eval_output(funcs, y_base, I0)
    directions = p2c123.build_task_directions(S)
    rows: list[dict[str, Any]] = []
    for name, direction in directions:
        for radius in CURRENT_RADII_A:
            rows.append(
                run_case(
                    name=name,
                    task_direction=direction,
                    radius=radius,
                    S=S,
                    I0=I0,
                    y_base=y_base,
                    basis=basis,
                    funcs=funcs,
                    params=params,
                    base_out=base_out,
                )
            )
    pass_radii = []
    any_radii = []
    for radius in sorted({row["radius_A"] for row in rows}):
        group = [row for row in rows if row["radius_A"] == radius]
        if all(row["status"] == "pass" for row in group):
            pass_radii.append(radius)
        if any(row["status"] == "pass" for row in group):
            any_radii.append(radius)
    summary = {
        "experiment_id": OUT_DIR.name,
        "case": p2c134.p2c104.p2c79.p2c57.CASE_NAME,
        "alpha": p2c134.p2c104.p2c79.p2c57.ALPHA,
        "tested_cases": len(rows),
        "pass_count": sum(row["status"] == "pass" for row in rows),
        "failure_count": sum(row["status"] != "pass" for row in rows),
        "largest_all_direction_pass_radius_A": max(pass_radii) if pass_radii else None,
        "largest_any_direction_pass_radius_A": max(any_radii) if any_radii else None,
        "min_reliable_displacement_um": MIN_RELIABLE_DISPLACEMENT_UM,
        "low_signal_failure_count": sum(
            row["failure_class"] == "末端位移低于可靠判读阈值" for row in rows
        ),
        "dominant_interpretation": "优势方向可局部发布，完整三维位置灵敏度不可发布；弱方向主要受位移幅值和方向增益限制。",
        "state_radius_gate": STATE_RADIUS_GATE,
        "linear_rel_gate": LINEAR_REL_GATE,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "rows": rows,
        "total_solve_steady_s": sum(row["solve_steady_s"] for row in rows),
        "wall_s": time.perf_counter() - t0,
    }
    p2c134.write_csv(OUT_DIR / "direction_current_rows.csv", rows)
    p2c134.write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "tested_cases": summary["tested_cases"],
                "pass_count": summary["pass_count"],
                "failure_count": summary["failure_count"],
                "largest_all_direction_pass_radius_A": summary["largest_all_direction_pass_radius_A"],
                "largest_any_direction_pass_radius_A": summary["largest_any_direction_pass_radius_A"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c134.jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(p2c134.jsonable(row), ensure_ascii=False) + "\n" for row in rows if row["status"] != "pass"),
        encoding="utf-8",
    )
    plot(rows)
    write_record(summary, rows)
    print(json.dumps(p2c134.jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
