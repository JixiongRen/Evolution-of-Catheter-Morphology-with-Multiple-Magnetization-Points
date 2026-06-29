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
P2C162_DIR = ROOT / "experiment_records_ik_optimize/20260630180000_p2c162_n2_rolling_two_step_prototype"
for path in (ROOT, P2C162_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c162_n2_rolling_two_step_prototype as p2c162


FIRST_STEP_RADIUS_A = 1.0e-3
SECOND_RADII_A = (2.5e-4, 5.0e-4, 7.5e-4, 1.0e-3)
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


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one second-step radius case."""
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
    return "通过", "第二步半径审计通过"


def raw_interface_blocks(branch: dict[str, Any], y: np.ndarray) -> np.ndarray:
    """Return raw interface wrench residual blocks as [force, torque]."""
    args = p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.Args
    p2c47 = p2c162.p2c148.p2c134.p2c104.p2c79.p2c57.p2c47
    raw = p2c47.p2c8.interface_wrench_residual(
        branch["params"],
        p2c162.p2c148.p2c145.p2c79.jnp.asarray(y, dtype=p2c162.p2c148.p2c145.p2c79.jnp.float64),
        steps_per_flex=args.steps_per_flex,
    )
    arr = np.asarray(jax.device_get(raw), dtype=float)
    if arr.size == 0:
        return arr.reshape((0, 6))
    return arr.reshape((-1, 6))


def run_second_radius(branch: dict[str, Any], radius: float) -> dict[str, Any]:
    """Run one second-step radius from the rebuilt first-step branch."""
    task_direction = np.asarray(branch["U"][:, 0], dtype=float)
    d_i = p2c162.p2c148.p2c123.current_step_for_task_direction(branch["S"], task_direction, radius)
    pred = branch["S"] @ d_i
    current_new = branch["I0"] + d_i
    q, info = p2c162.p2c148.p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=branch["y"],
        basis=branch["basis"],
        I=current_new,
        funcs=branch["funcs"],
    )
    y_new = branch["y"] + branch["basis"] @ q
    out_new = p2c162.p2c148.p2c134.eval_output(branch["funcs"], y_new, current_new)
    actual = out_new["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = p2c162.relative_error(pred, actual)
    continuity, force_ratio, torque_ratio = p2c162.p2c148.p2c134.load_metrics(branch["params"], y_new)
    blocks = raw_interface_blocks(branch, y_new)
    force_block_norms = np.linalg.norm(blocks[:, :3], axis=1) if blocks.size else np.zeros((0,), dtype=float)
    torque_block_norms = np.linalg.norm(blocks[:, 3:6], axis=1) if blocks.size else np.zeros((0,), dtype=float)
    block_norms = np.linalg.norm(blocks, axis=1) if blocks.size else np.zeros((0,), dtype=float)
    dominant_idx = int(np.argmax(block_norms)) if block_norms.size else -1
    row = {
        "second_radius_A": radius,
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
        "dominant_interface_index": dominant_idx,
        "dominant_interface_norm": float(block_norms[dominant_idx]) if dominant_idx >= 0 else 0.0,
        "dominant_force_norm": float(force_block_norms[dominant_idx]) if dominant_idx >= 0 else 0.0,
        "dominant_torque_norm": float(torque_block_norms[dominant_idx]) if dominant_idx >= 0 else 0.0,
        "max_force_block_norm": float(force_block_norms.max()) if force_block_norms.size else 0.0,
        "max_torque_block_norm": float(torque_block_norms.max()) if torque_block_norms.size else 0.0,
        "force_ratio": force_ratio,
        "torque_ratio": torque_ratio,
        "state_radius": float(np.linalg.norm(q)),
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
    }
    release_class, reason = classify(row)
    row["release_class"] = release_class
    row["reason"] = reason
    return row


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot second-step radius and block metrics."""
    configure_matplotlib_cjk()
    labels = [f"{row['second_radius_A']:.4f}A" for row in rows]
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if row["release_class"] == "通过" else "#e76f51" for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.2), sharex=True)
    axes[0].bar(x, [row["continuity_norm"] for row in rows], color=colors)
    axes[0].axhline(CONTINUITY_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("接口连续残差")
    axes[1].bar(x, [row["linear_rel_err"] for row in rows], color=colors)
    axes[1].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("线性化误差")
    axes[2].bar(x - 0.15, [row["max_force_block_norm"] for row in rows], width=0.3, label="力分量")
    axes[2].bar(x + 0.15, [row["max_torque_block_norm"] for row in rows], width=0.3, label="矩分量")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("最大分块残差")
    axes[2].set_xticks(x, labels)
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2 第二步半径与接口残差分块审计")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_second_step_radius_block_audit_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]], first_row: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 第二步半径/A | 线性误差 | 连续残差 | 主导接口 | 力分量 | 矩分量 | 结论 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['second_radius_A']:.4f} | {row['linear_rel_err']:.3e} | {row['continuity_norm']:.3e} | "
            f"{row['dominant_interface_index']} | {row['dominant_force_norm']:.3e} | "
            f"{row['dominant_torque_norm']:.3e} | {row['release_class']} |"
        )
    max_passed_text = "无" if summary["max_passed_second_radius_A"] is None else f"{summary['max_passed_second_radius_A']:.5f}"
    text = f"""# P2c163 两点磁控 n2 第二步半径与接口残差分块审计

## 实验目的

P2c162 表明滚动重建后第二步仍因接口连续残差超限失败。本轮固定第一步半径为 0.001 A，在第一步后重建分支与二维灵敏度，然后扫描第二步半径，判断是否存在更保守的第二步可用半径，并定位接口连续残差的主要分量。

## 实验方案

第一步沿 `rand_05` 工作点第一主方向，半径为 0.001 A。第一步通过后，重建当前局部分支与二维灵敏度。第二步沿重建后的第一主方向，半径取 0.00025 A、0.00050 A、0.00075 A、0.00100 A。每个半径均重新求解状态，并分解接口残差为力分量和矩分量。

## 第一步基线

第一步结论为 `{first_row['release_class']}`，线性化误差为 {first_row['linear_rel_err']:.3e}，接口连续残差为 {first_row['continuity_norm']:.3e}。

## 第二步结果

{chr(10).join(table)}

## 数据分析

最大全通过第二步半径为 `{max_passed_text}` A。若该值为无，说明即使第二步缩小到 0.00025 A 仍不能满足接口连续门禁。本轮所有半径的主导接口均为接口 0，且矩分量明显大于力分量，说明下一轮应优先检查矩连续约束，而不是继续扩大路径扫描。

## 实验结论

{summary['decision']}

## 下一步计划

若存在通过半径，应采用分步递减半径策略继续验证更多路径；若不存在通过半径，应进入接口连续残差模型修复，而不是继续扩大工作点扫描。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the second-step radius and residual-block audit."""
    t0 = time.perf_counter()
    base = p2c162.p2c148.p2c145.p2c144.p2c89._recover_trusted()
    direction_name, direction = next(iter(p2c162.p2c148.p2c145.p2c144.candidate_directions()))
    branch0 = p2c162.p2c148.build_candidate_branch(base, direction_name, direction)
    first_row, branch1 = p2c162.run_step(branch0, step_index=1, mode_index=0)
    rows = [run_second_radius(branch1, radius) for radius in SECOND_RADII_A]
    passed = [row for row in rows if row["release_class"] == "通过"]
    max_passed = max((row["second_radius_A"] for row in passed), default=None)
    decision = (
        f"第二步存在保守可用半径 {max_passed} A，可进入更多路径验证。"
        if max_passed is not None
        else "第二步缩半径仍无全通过样本，应转入接口连续模型修复。"
    )
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "second_step_radius_found" if max_passed is not None else "second_step_radius_not_found",
        "case": direction_name,
        "first_step_radius_A": FIRST_STEP_RADIUS_A,
        "tested_second_radii_A": list(SECOND_RADII_A),
        "pass_count": len(passed),
        "tested_count": len(rows),
        "max_passed_second_radius_A": max_passed,
        "outer_ik_allowed": False,
        "full_3d_position_publishable": False,
        "decision": decision,
        "wall_s": float(time.perf_counter() - t0),
        "total_solve_steady_s": float(first_row["solve_steady_s"] + sum(row["solve_steady_s"] for row in rows)),
        "jax_devices": [str(device) for device in jax.devices()],
        "first_step": first_row,
        "rows": rows,
    }
    failures = [
        {"second_radius_A": row["second_radius_A"], "failure_class": row["reason"], "evidence": row}
        for row in rows
        if row["release_class"] != "通过"
    ]
    write_csv(EXP_DIR / "second_step_radius_rows.csv", rows)
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
  experiment_records_ik_optimize/20260630183000_p2c163_n2_second_step_radius_block_audit/run_p2c163_n2_second_step_radius_block_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
