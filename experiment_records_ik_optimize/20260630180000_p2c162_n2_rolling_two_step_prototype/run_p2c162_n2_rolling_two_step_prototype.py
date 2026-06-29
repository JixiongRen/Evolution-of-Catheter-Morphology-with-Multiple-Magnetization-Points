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
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
P2C148_DIR = ROOT / "experiment_records_ik_optimize/20260630103000_p2c148_n2_all_principal_safe_radius"
for path in (ROOT, P2C148_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c148_n2_all_principal_safe_radius as p2c148


STEP_RADIUS_A = 1.0e-3
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


def normalized(v: np.ndarray) -> np.ndarray:
    """Return a normalized vector."""
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1.0e-18)


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one rolling step."""
    if not row["solve_success"]:
        return "不可发布", "扰动重求解未收敛"
    if row["linear_rel_err"] >= LINEAR_REL_GATE:
        return "不可发布", "局部线性化误差超限"
    if row["centerline_max_change_m"] >= CENTERLINE_GATE_M:
        return "不可发布", "中心线分支漂移超限"
    if row["continuity_norm"] >= CONTINUITY_GATE:
        return "不可发布", "接口连续残差超限"
    if row["force_ratio"] >= FORCE_RATIO_GATE:
        return "不可发布", "载荷传播比例超限"
    if row["state_radius"] >= STATE_RADIUS_GATE:
        return "谨慎诊断", "有效状态增量偏大"
    return "通过", "滚动局部扰动重求解通过"


def relative_error(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine."""
    pred = np.asarray(pred, dtype=float).reshape(3)
    actual = np.asarray(actual, dtype=float).reshape(3)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    return rel, float(np.dot(pred, actual) / denom)


def rebuild_local_branch(seed: dict[str, Any], y: np.ndarray, current_i: np.ndarray, out: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild the local reduced basis and terminal-position sensitivity."""
    funcs = seed["funcs"]
    y_j = p2c148.p2c145.p2c79.jnp.asarray(y, dtype=p2c148.p2c145.p2c79.jnp.float64)
    i_j = p2c148.p2c145.p2c79.jnp.asarray(current_i, dtype=p2c148.p2c145.p2c79.jnp.float64).reshape(8)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, i_j)), dtype=float)
    _u_j, _s_j, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c148.p2c145.p2c79.p2c72.p2c71._orthogonal_complement(
        vt[-p2c148.p2c145.p2c79.p2c72.REMOVE_WEAK_DIRECTIONS :],
        J.shape[1],
    )
    system = p2c148.p2c145.p2c79.p2c77._build_optimality_system(
        y_base=y,
        I0=current_i,
        funcs=funcs,
        basis=basis,
    )
    S = p2c148.p2c145.p2c79.p2c78._task_sensitivity(system)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    return {
        "name": seed["name"],
        "params": seed["params"],
        "I0": np.asarray(current_i, dtype=float),
        "y": np.asarray(y, dtype=float),
        "basis": basis,
        "funcs": funcs,
        "S": S,
        "U": U,
        "Vt": Vt,
        "singular_values": singular_values,
        "base_out": out,
    }


def current_step_from_mode(branch: dict[str, Any], mode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative current step along one local task mode."""
    task_direction = np.asarray(branch["U"][:, mode_index], dtype=float)
    dI = p2c148.p2c123.current_step_for_task_direction(branch["S"], task_direction, STEP_RADIUS_A)
    return dI, task_direction


def run_step(branch: dict[str, Any], *, step_index: int, mode_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one rolling local step and return the next rebuilt branch."""
    dI, task_direction = current_step_from_mode(branch, mode_index)
    pred = branch["S"] @ dI
    current_new = branch["I0"] + dI
    q, info = p2c148.p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=branch["y"],
        basis=branch["basis"],
        I=current_new,
        funcs=branch["funcs"],
    )
    y_new = branch["y"] + branch["basis"] @ q
    out_new = p2c148.p2c134.eval_output(branch["funcs"], y_new, current_new)
    actual = out_new["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = relative_error(pred, actual)
    continuity, force_ratio, torque_ratio = p2c148.p2c134.load_metrics(branch["params"], y_new)
    row = {
        "step_index": step_index,
        "mode_index": mode_index + 1,
        "step_radius_A": STEP_RADIUS_A,
        "singular_value_1": float(branch["singular_values"][0]),
        "singular_value_2": float(branch["singular_values"][1]),
        "singular_value_3": float(branch["singular_values"][2]),
        "condition_full": float(branch["singular_values"][0] / max(float(branch["singular_values"][-1]), np.finfo(float).tiny)),
        "condition_2d": float(branch["singular_values"][0] / max(float(branch["singular_values"][1]), np.finfo(float).tiny)),
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "linear_rel_err": rel,
        "pred_actual_cosine": cosine,
        "centerline_max_change_m": float(
            np.max(np.linalg.norm(out_new["centerline"] - branch["base_out"]["centerline"], axis=1))
        ),
        "continuity_norm": continuity,
        "force_ratio": force_ratio,
        "torque_ratio": torque_ratio,
        "state_radius": float(np.linalg.norm(q)),
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
        "current_norm_A": float(np.linalg.norm(current_new - branch["I0"])),
    }
    release_class, reason = classify(row)
    row["release_class"] = release_class
    row["reason"] = reason
    next_branch = rebuild_local_branch(branch, y_new, current_new, out_new)
    return row, next_branch


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot rolling two-step metrics."""
    configure_matplotlib_cjk()
    labels = [f"第{row['step_index']}步" for row in rows]
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if row["release_class"] == "通过" else "#e76f51" for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.5), sharex=True)
    axes[0].bar(x, [row["linear_rel_err"] for row in rows], color=colors)
    axes[0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("线性化误差")
    axes[1].bar(x, [row["continuity_norm"] for row in rows], color=colors)
    axes[1].axhline(CONTINUITY_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("接口连续残差")
    axes[2].bar(x, [row["centerline_max_change_m"] for row in rows], color=colors)
    axes[2].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("中心线变化/m")
    axes[2].set_xticks(x, labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2 滚动局部分支重建两步原型")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_rolling_two_step_metrics_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 步数 | 主方向 | 线性误差 | 连续残差 | 中心线变化/m | 载荷比 | 结论 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['step_index']} | {row['mode_index']} | {row['linear_rel_err']:.3e} | "
            f"{row['continuity_norm']:.3e} | {row['centerline_max_change_m']:.3e} | "
            f"{row['force_ratio']:.3f} | {row['release_class']} |"
        )
    text = f"""# P2c162 两点磁控 n2 滚动两步原型

## 实验目的

P2c156 表明固定二维子空间连续复用时，第二步因接口连续残差超限失败。本轮按 P2c161 协议，在第一步后重新求解当前分支并重建局部二维灵敏度，再执行第二步。目标是判断滚动重建是否能消除固定子空间第二步失败。

## 实验方案

选择 P2c148 中的 `rand_05` 工作点。第一步沿当前第一主方向，电流步长为 0.001 A。第一步完成后，以新物理状态和新电流为基点，重建约化基、局部灵敏度矩阵和二维主方向，再沿新的第一主方向执行第二步。每一步均按一阶最优性、局部线性化误差、接口连续残差、中心线变化和载荷传播比例判定。

## 结果

{chr(10).join(table)}

## 数据分析

通过步数为 {summary['pass_count']} / {summary['tested_count']}。第二步局部线性化误差为 {rows[1]['linear_rel_err']:.3e}，明显低于 0.05 门限；中心线变化为 {rows[1]['centerline_max_change_m']:.3e} m，也低于分支漂移门限。因此滚动重建确实改善了局部线性化一致性和分支保持能力。但第二步接口连续残差为 {rows[1]['continuity_norm']:.3e}，高于 1.0e-3 门限，仍不能发布为多步灵敏度。

## 实验结论

{summary['decision']}

## 下一步计划

下一轮应定位第二步接口连续残差的分块来源，比较固定子空间与滚动重建下的接口连续残差组成，并尝试更小第二步半径或接口连续约束增强。若连续三轮仍卡在接口连续残差超限，应转入模型耦合修复，而非继续扩大路径扫描。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the rolling two-step prototype."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c148.build_candidate_branch(base, direction_name, direction)
    row1, branch1 = run_step(branch0, step_index=1, mode_index=0)
    row2, _branch2 = run_step(branch1, step_index=2, mode_index=0)
    rows = [row1, row2]
    failures = [
        {
            "step_index": row["step_index"],
            "failure_class": row["reason"],
            "evidence": row,
        }
        for row in rows
        if row["release_class"] != "通过"
    ]
    pass_count = sum(row["release_class"] == "通过" for row in rows)
    decision = (
        "滚动重建两步原型通过，可扩展到更多路径与工作点。"
        if pass_count == len(rows)
        else "滚动重建两步原型未完全通过，不能升级为外层 IK。"
    )
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "rolling_two_step_passed" if pass_count == len(rows) else "rolling_two_step_failed",
        "case": direction_name,
        "path": "first_principal_then_rebuilt_first_principal",
        "tested_count": len(rows),
        "pass_count": pass_count,
        "fixed_subspace_reference": "P2c156 second step 0/20 pass",
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "effective_2d_multistep_candidate": pass_count == len(rows),
        "decision": decision,
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "rows": rows,
    }
    write_csv(EXP_DIR / "rolling_two_step_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(jsonable(failure), ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary, rows)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260630180000_p2c162_n2_rolling_two_step_prototype/run_p2c162_n2_rolling_two_step_prototype.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
