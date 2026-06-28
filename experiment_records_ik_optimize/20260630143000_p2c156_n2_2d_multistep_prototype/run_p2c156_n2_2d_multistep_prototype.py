"""Prototype multistep n2 effective-subspace diagnostic updates."""

from __future__ import annotations

import csv
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
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


def jsonable(value: Any) -> Any:
    """Convert arrays and scalar values to JSON-compatible values."""
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
    """Write rows with stable field ordering."""
    if not rows:
        raise ValueError("no rows")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized(v: np.ndarray) -> np.ndarray:
    """Return a normalized vector."""
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1.0e-18)


def subspace_directions(branch: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return named task directions inside the audited 2D subspace."""
    u1 = np.asarray(branch["U"][:, 0], dtype=float)
    u2 = np.asarray(branch["U"][:, 1], dtype=float)
    return {
        "第一主方向": u1,
        "第二主方向": u2,
        "二维混合方向": normalized(u1 + u2),
    }


def current_step_from_u2(branch: dict[str, Any], task_direction: np.ndarray) -> np.ndarray:
    """Compute one conservative-radius rank-2 current step."""
    u2 = np.asarray(branch["U"][:, :2], dtype=float)
    v2 = np.asarray(branch["Vt"][:2, :], dtype=float)
    sigma = np.asarray(branch["singular_values"][:2], dtype=float)
    direction = normalized(task_direction).reshape(3)
    coeff = u2.T @ direction
    dI = v2.T @ (coeff / sigma)
    return STEP_RADIUS_A * normalized(dI)


def relative_error(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine."""
    pred = np.asarray(pred, dtype=float).reshape(3)
    actual = np.asarray(actual, dtype=float).reshape(3)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    return rel, float(np.dot(pred, actual) / denom)


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one multistep row."""
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
    return "通过", "二维多步扰动重求解通过"


def run_step(
    *,
    branch: dict[str, Any],
    path_label: str,
    step_index: int,
    q0: np.ndarray,
    I_current: np.ndarray,
    dI_step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one cumulative reduced solve step."""
    I_new = I_current + dI_step
    q, info = p2c148.p2c134.solve_reduced(
        q0=q0,
        y_base=branch["y"],
        basis=branch["basis"],
        I=I_new,
        funcs=branch["funcs"],
    )
    y_new = branch["y"] + branch["basis"] @ q
    out = p2c148.p2c134.eval_output(branch["funcs"], y_new, I_new)
    cumulative_dI = I_new - np.asarray(branch["I0"], dtype=float)
    pred = np.asarray(branch["S"], dtype=float) @ cumulative_dI
    actual = out["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = relative_error(pred, actual)
    centerline_change = float(np.max(np.linalg.norm(out["centerline"] - branch["base_out"]["centerline"], axis=1)))
    continuity, force_ratio, torque_ratio = p2c148.p2c134.load_metrics(branch["params"], y_new)
    row = {
        "direction": branch["name"],
        "path_label": path_label,
        "step_index": step_index,
        "single_step_radius_A": STEP_RADIUS_A,
        "cumulative_current_norm_A": float(np.linalg.norm(cumulative_dI)),
        "linear_rel_err": rel,
        "pred_actual_cosine": cosine,
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "centerline_max_change_m": centerline_change,
        "continuity_norm": continuity,
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
    return q, I_new, row


def paths_for_branch(branch: dict[str, Any]) -> list[tuple[str, list[np.ndarray]]]:
    """Build deterministic two-step paths."""
    dirs = subspace_directions(branch)
    return [
        ("第一主方向连续两步", [dirs["第一主方向"], dirs["第一主方向"]]),
        ("第二主方向连续两步", [dirs["第二主方向"], dirs["第二主方向"]]),
        ("混合方向连续两步", [dirs["二维混合方向"], dirs["二维混合方向"]]),
        ("第一到第二主方向", [dirs["第一主方向"], dirs["第二主方向"]]),
    ]


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot multistep pass rates and gate maxima."""
    configure_matplotlib_cjk()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["step_index"]), []).append(row)
    steps = sorted(grouped)
    x = np.arange(len(steps))
    pass_rates = [sum(r["release_class"] == "通过" for r in grouped[step]) / len(grouped[step]) for step in steps]
    max_linear = [max(float(r["linear_rel_err"]) for r in grouped[step]) for step in steps]
    max_continuity = [max(float(r["continuity_norm"]) for r in grouped[step]) for step in steps]
    max_drift = [max(float(r["centerline_max_change_m"]) for r in grouped[step]) for step in steps]
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 10.5), sharex=True)
    colors = ["#2a9d8f" if rate == 1.0 else "#e76f51" for rate in pass_rates]
    axes[0].bar(x, pass_rates, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("通过率")
    axes[0].set_title("两点磁控二维诊断步多步原型")
    axes[1].bar(x, max_linear, color=colors)
    axes[1].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("最大线性误差")
    axes[2].bar(x, max_continuity, color=colors)
    axes[2].axhline(CONTINUITY_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("最大连续残差")
    axes[3].bar(x, max_drift, color=colors)
    axes[3].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[3].set_yscale("log")
    axes[3].set_ylabel("最大分支漂移/m")
    axes[3].set_xticks(x, [f"第{step}步" for step in steps])
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_2d_multistep_prototype_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    by_step = summary["by_step"]
    step_table = [
        "| 步数 | 通过数/总数 | 最大线性误差 | 最大连续残差 | 最大分支漂移/m | 判定 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for step, stats in by_step.items():
        step_table.append(
            f"| {step} | {stats['pass_count']}/{stats['tested_count']} | {stats['max_linear_rel_err']:.3e} | "
            f"{stats['max_continuity_norm']:.3e} | {stats['max_centerline_change_m']:.3e} | {stats['decision']} |"
        )
    detail_table = [
        "| 工作点 | 路径 | 步数 | 累计电流/A | 线性误差 | 连续残差 | 载荷比 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        detail_table.append(
            f"| {row['direction']} | {row['path_label']} | {row['step_index']} | "
            f"{row['cumulative_current_norm_A']:.3e} | {row['linear_rel_err']:.3e} | "
            f"{row['continuity_norm']:.3e} | {row['force_ratio']:.3f} | {row['release_class']} |"
        )
    text = f"""# P2c156 两点磁控二维诊断步多步原型

## 实验目的

P2c155 已固定 0.0010 A 为二维诊断步保守半径。本轮检验连续使用该单步接口时，物理解分支和门禁是否仍保持稳定，避免把单步诊断能力误扩展为多步外层逆运动学能力。

## 方法

每个工作点测试四条两步路径：第一主方向连续两步、第二主方向连续两步、混合方向连续两步、第一到第二主方向。每一步电流增量范数为 0.0010 A。第二步以上一步重求解结果作为初值，但仍使用同一局部二维子空间模型预测累计末端位移。

## 结果

{chr(10).join(step_table)}

## 明细

{chr(10).join(detail_table)}

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。最大线性化误差为 {summary['max_linear_rel_err']:.3e}，最大接口连续残差为 {summary['max_continuity_norm']:.3e}，最大中心线分支漂移为 {summary['max_centerline_change_m']:.3e} m。

## 实验结论

当前判定：{summary['decision']}。

若第二步显著失败，则说明当前接口只能作为单步局部诊断工具，不能直接升级为多步外层逆运动学。若两步均通过，也仍只能说明该小规模路径集合内多步稳定，尚需更多方向和工作点复核。

## 下一步计划

若本轮失败，应把 P2c155 接口说明进一步收紧为“单步诊断接口”。若本轮通过，可增加三步路径或跨工作点随机路径复核，但仍不接完整三维外层逆运动学。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the multistep prototype."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        branch = p2c148.build_candidate_branch(base, name, direction)
        for path_label, task_path in paths_for_branch(branch):
            q = np.zeros(branch["basis"].shape[1], dtype=float)
            I = np.asarray(branch["I0"], dtype=float).copy()
            for step_index, task_direction in enumerate(task_path, start=1):
                dI = current_step_from_u2(branch, task_direction)
                q, I, row = run_step(
                    branch=branch,
                    path_label=path_label,
                    step_index=step_index,
                    q0=q,
                    I_current=I,
                    dI_step=dI,
                )
                rows.append(row)
                if row["release_class"] != "通过":
                    failures.append(
                        {
                            "direction": row["direction"],
                            "path_label": row["path_label"],
                            "step_index": row["step_index"],
                            "failure_class": row["reason"],
                            "linear_rel_err": row["linear_rel_err"],
                            "continuity_norm": row["continuity_norm"],
                            "centerline_max_change_m": row["centerline_max_change_m"],
                        }
                    )
    wall_s = time.perf_counter() - t0
    pass_count = sum(row["release_class"] == "通过" for row in rows)
    total_solve = sum(float(row["solve_steady_s"]) for row in rows)
    by_step: dict[str, dict[str, Any]] = {}
    for step in sorted({int(row["step_index"]) for row in rows}):
        step_rows = [row for row in rows if int(row["step_index"]) == step]
        step_pass = sum(row["release_class"] == "通过" for row in step_rows)
        by_step[str(step)] = {
            "tested_count": len(step_rows),
            "pass_count": step_pass,
            "all_passed": step_pass == len(step_rows),
            "max_linear_rel_err": max(float(row["linear_rel_err"]) for row in step_rows),
            "max_continuity_norm": max(float(row["continuity_norm"]) for row in step_rows),
            "max_centerline_change_m": max(float(row["centerline_max_change_m"]) for row in step_rows),
            "decision": "通过" if step_pass == len(step_rows) else "不通过",
        }
    failure_by_class = collections.Counter(failure["failure_class"] for failure in failures)
    failure_by_step: dict[str, dict[str, int]] = {}
    for failure in failures:
        step_key = str(failure["step_index"])
        failure_by_step.setdefault(step_key, {})
        failure_by_step[step_key][failure["failure_class"]] = (
            failure_by_step[step_key].get(failure["failure_class"], 0) + 1
        )
    summary = {
        "experiment_id": "20260630143000_p2c156_n2_2d_multistep_prototype",
        "tested_count": len(rows),
        "pass_count": pass_count,
        "all_passed": pass_count == len(rows),
        "single_step_radius_A": STEP_RADIUS_A,
        "path_count": 20,
        "effective_subspace_dimension": 2,
        "max_linear_rel_err": max(float(row["linear_rel_err"]) for row in rows),
        "max_continuity_norm": max(float(row["continuity_norm"]) for row in rows),
        "max_centerline_change_m": max(float(row["centerline_max_change_m"]) for row in rows),
        "max_force_ratio": max(float(row["force_ratio"]) for row in rows),
        "total_solve_steady_s": total_solve,
        "wall_s": wall_s,
        "by_step": by_step,
        "failure_by_class": dict(failure_by_class),
        "failure_by_step": failure_by_step,
        "decision": "二维多步原型候选" if pass_count == len(rows) else "保持单步诊断接口，需多步失败归因",
        "outer_ik_allowed": False,
    }
    write_csv(OUT_DIR / "multistep_rows.csv", jsonable(rows))
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(failure), ensure_ascii=False) + "\n" for failure in failures),
        encoding="utf-8",
    )
    plot(rows)
    write_record(summary, rows)


if __name__ == "__main__":
    main()
