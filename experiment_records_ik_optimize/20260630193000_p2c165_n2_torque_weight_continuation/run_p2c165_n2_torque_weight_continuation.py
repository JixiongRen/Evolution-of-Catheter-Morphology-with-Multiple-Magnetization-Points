from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C162_DIR = ROOT / "experiment_records_ik_optimize/20260630180000_p2c162_n2_rolling_two_step_prototype"
for path in (ROOT, P2C162_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
import run_p2c162_n2_rolling_two_step_prototype as p2c162


STEP_RADIUS_A = 1.0e-3
TORQUE_WEIGHTS = (0.0, 4.0, 16.0)
LINEAR_REL_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
FORCE_RATIO_GATE = 1.5
CONTINUITY_GATE = 1.0e-3
STATE_RADIUS_GATE = 1.0e-3


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


def current_aware_interface_blocks(branch: dict[str, Any], y: np.ndarray, current_i: np.ndarray) -> np.ndarray:
    """Return current-aware normalized interface residual blocks."""
    p2c47 = p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.p2c47
    args = p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.Args
    p2c47 = p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.p2c47
    params_i = p2c47.with_coil_currents(branch["params"], jnp.asarray(current_i, dtype=jnp.float64).reshape(8))
    raw = p2c47.p2c8.interface_wrench_residual(
        params_i,
        jnp.asarray(y, dtype=jnp.float64),
        steps_per_flex=args.steps_per_flex,
    )
    arr = np.asarray(jax.device_get(raw), dtype=float)
    return arr.reshape((-1, 6)) if arr.size else arr.reshape((0, 6))


def metric_from_blocks(blocks: np.ndarray) -> dict[str, float]:
    """Summarize interface residual blocks."""
    if blocks.size == 0:
        return {
            "continuity_norm": 0.0,
            "dominant_interface_index": -1,
            "dominant_force_norm": 0.0,
            "dominant_torque_norm": 0.0,
            "max_force_block_norm": 0.0,
            "max_torque_block_norm": 0.0,
        }
    force_norms = np.linalg.norm(blocks[:, :3], axis=1)
    torque_norms = np.linalg.norm(blocks[:, 3:6], axis=1)
    block_norms = np.linalg.norm(blocks, axis=1)
    idx = int(np.argmax(block_norms))
    return {
        "continuity_norm": float(np.linalg.norm(blocks.reshape(-1))),
        "dominant_interface_index": idx,
        "dominant_force_norm": float(force_norms[idx]),
        "dominant_torque_norm": float(torque_norms[idx]),
        "max_force_block_norm": float(force_norms.max()),
        "max_torque_block_norm": float(torque_norms.max()),
    }


def solve_with_torque_weight(branch: dict[str, Any], radius: float, torque_weight: float) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Solve the second step with an extra torque-continuity penalty."""
    d_i, _task_direction = p2c162.current_step_from_mode(branch, 0)
    d_i = radius * d_i / max(float(np.linalg.norm(d_i)), 1.0e-18)
    current_new = branch["I0"] + d_i
    i_j = jnp.asarray(current_new, dtype=jnp.float64).reshape(8)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    sqrt_w = float(np.sqrt(max(torque_weight, 0.0)))

    def residual(q: np.ndarray) -> np.ndarray:
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        base = np.asarray(jax.device_get(branch["funcs"]["residual"](jnp.asarray(y, dtype=jnp.float64), i_j)), dtype=float)
        if sqrt_w <= 0.0:
            return base
        torque = current_aware_interface_blocks(branch, y, current_new)[:, 3:6].reshape(-1)
        return np.concatenate([base, sqrt_w * torque], axis=0)

    q0 = np.zeros(basis.shape[1], dtype=float)
    _ = residual(q0)
    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        q0,
        jac="2-point",
        method="trf",
        x_scale=np.ones_like(q0),
        loss="linear",
        max_nfev=25,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    return np.asarray(opt.x, dtype=float), {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(time.perf_counter() - t0),
        "cost": float(opt.cost),
        "optimality": float(opt.optimality),
        "message": str(opt.message),
    }, current_new


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one torque-weight row."""
    if not row["solve_success"]:
        return "不可发布", "扰动重求解未收敛"
    if row["linear_rel_err"] >= LINEAR_REL_GATE:
        return "不可发布", "局部线性化误差超限"
    if row["centerline_max_change_m"] >= CENTERLINE_GATE_M:
        return "不可发布", "中心线分支漂移超限"
    if row["current_aware_continuity_norm"] >= CONTINUITY_GATE:
        return "不可发布", "接口连续残差超限"
    if row["force_ratio_legacy"] >= FORCE_RATIO_GATE:
        return "不可发布", "载荷传播比例超限"
    if row["state_radius"] >= STATE_RADIUS_GATE:
        return "谨慎诊断", "有效状态增量偏大"
    if row["optimality"] > 1.0e-5:
        return "不可发布", "一阶最优性退化"
    return "通过", "矩连续权重延续通过"


def run_weight_case(branch: dict[str, Any], torque_weight: float) -> dict[str, Any]:
    """Run one second-step torque-weight case."""
    d_i, _task_direction = p2c162.current_step_from_mode(branch, 0)
    pred = branch["S"] @ d_i
    q, info, current_new = solve_with_torque_weight(branch, STEP_RADIUS_A, torque_weight)
    y_new = branch["y"] + branch["basis"] @ q
    out_new = p2c162.p2c148.p2c134.eval_output(branch["funcs"], y_new, current_new)
    actual = out_new["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = p2c162.relative_error(pred, actual)
    legacy_continuity, force_ratio, torque_ratio = p2c162.p2c148.p2c134.load_metrics(branch["params"], y_new)
    current_blocks = current_aware_interface_blocks(branch, y_new, current_new)
    current_metrics = metric_from_blocks(current_blocks)
    row = {
        "torque_weight": torque_weight,
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "linear_rel_err": rel,
        "pred_actual_cosine": cosine,
        "centerline_max_change_m": float(
            np.max(np.linalg.norm(out_new["centerline"] - branch["base_out"]["centerline"], axis=1))
        ),
        "legacy_continuity_norm": legacy_continuity,
        "current_aware_continuity_norm": current_metrics["continuity_norm"],
        "current_aware_dominant_interface_index": current_metrics["dominant_interface_index"],
        "current_aware_dominant_force_norm": current_metrics["dominant_force_norm"],
        "current_aware_dominant_torque_norm": current_metrics["dominant_torque_norm"],
        "force_ratio_legacy": force_ratio,
        "torque_ratio_legacy": torque_ratio,
        "state_radius": float(np.linalg.norm(q)),
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
        "optimality": float(info["optimality"]),
        "cost": float(info["cost"]),
    }
    release_class, reason = classify(row)
    row["release_class"] = release_class
    row["reason"] = reason
    return row


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot torque-weight continuation metrics."""
    configure_matplotlib_cjk()
    labels = [str(row["torque_weight"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if row["release_class"] == "通过" else "#e76f51" for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.2), sharex=True)
    axes[0].bar(x, [row["current_aware_continuity_norm"] for row in rows], color=colors)
    axes[0].axhline(CONTINUITY_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("电流更新后连续残差")
    axes[1].bar(x, [row["linear_rel_err"] for row in rows], color=colors)
    axes[1].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("线性化误差")
    axes[2].bar(x, [row["optimality"] for row in rows], color=colors)
    axes[2].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("一阶最优性")
    axes[2].set_xlabel("额外矩连续权重")
    axes[2].set_xticks(x, labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2 第二步矩连续权重延续")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_torque_weight_continuation_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]], first_row: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 额外矩权重 | 线性误差 | 电流更新后连续残差 | 一阶最优性 | 状态增量 | 结论 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['torque_weight']:.1f} | {row['linear_rel_err']:.3e} | "
            f"{row['current_aware_continuity_norm']:.3e} | {row['optimality']:.3e} | "
            f"{row['state_radius']:.3e} | {row['release_class']} |"
        )
    best = min(rows, key=lambda row: row["current_aware_continuity_norm"])
    text = f"""# P2c165 两点磁控 n2 矩连续权重延续实验

## 实验目的

P2c164 指出第二步失败由接口矩连续残差主导。本轮在同一工作点、同一路径上，对第二步重求解额外加入矩连续惩罚项，判断增强矩连续约束是否能压低连续残差并保持末端位移线性化。

## 实验方案

第一步仍使用 0.001 A 半径并通过门禁。第二步半径固定为 0.001 A。对第二步残差额外加入接口矩连续分量，权重取 `{list(TORQUE_WEIGHTS)}`。每个权重均记录局部线性化误差、中心线变化、电流更新后的接口连续残差、一阶最优性和状态增量。

## 第一步基线

第一步结论为 `{first_row['release_class']}`，线性化误差为 {first_row['linear_rel_err']:.3e}，接口连续残差为 {first_row['continuity_norm']:.3e}。

## 第二步结果

{chr(10).join(table)}

## 数据分析

最小电流更新后连续残差出现在权重 `{best['torque_weight']}`，数值为 {best['current_aware_continuity_norm']:.3e}。可发布通过样本数为 {summary['pass_count']} / {summary['tested_count']}。若高权重能压低连续残差但一阶最优性或线性化误差退化，说明单纯加权不能作为发布修复，只能作为模型耦合诊断。

## 实验结论

{summary['decision']}

## 下一步计划

若没有权重通过，应转向刚段力臂矩、磁外矩和矩参考点的只读审计；若存在权重通过，应在更多路径上复核，并检查是否引入不可接受的末端位移偏移。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the torque-weight continuation experiment."""
    t0 = time.perf_counter()
    base = p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    first_row, branch1 = p2c162.run_step(branch0, step_index=1, mode_index=0)
    rows = [run_weight_case(branch1, weight) for weight in TORQUE_WEIGHTS]
    passed = [row for row in rows if row["release_class"] == "通过"]
    decision = (
        "存在矩连续权重可使第二步通过，可进入更多路径复核。"
        if passed
        else "矩连续权重延续未获得可发布第二步；应转向矩参考点和刚段力臂矩模型审计。"
    )
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "torque_weight_pass_found" if passed else "torque_weight_no_publishable_pass",
        "case": direction_name,
        "path": "first_principal_then_rebuilt_first_principal",
        "second_step_radius_A": STEP_RADIUS_A,
        "tested_count": len(rows),
        "pass_count": len(passed),
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "multistep_publishable": bool(passed),
        "decision": decision,
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(first_row["solve_steady_s"] + sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "first_step": first_row,
        "rows": rows,
    }
    failures = [
        {"torque_weight": row["torque_weight"], "failure_class": row["reason"], "evidence": row}
        for row in rows
        if row["release_class"] != "通过"
    ]
    write_csv(EXP_DIR / "torque_weight_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(jsonable(failure), ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows, first_row)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630193000_p2c165_n2_torque_weight_continuation/run_p2c165_n2_torque_weight_continuation.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
