"""Validate n2 effective-subspace steps with branch reoptimization."""

from __future__ import annotations

import csv
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


SAFE_RADIUS_A = 1.0e-3
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


def current_step_from_u2(branch: dict[str, Any], task_direction: np.ndarray, radius_A: float) -> np.ndarray:
    """Compute the rank-2 minimum-norm current step for a task direction."""
    u2 = np.asarray(branch["U"][:, :2], dtype=float)
    v2 = np.asarray(branch["Vt"][:2, :], dtype=float)
    sigma = np.asarray(branch["singular_values"][:2], dtype=float)
    direction = normalized(task_direction).reshape(3)
    coeff = u2.T @ direction
    dI = v2.T @ (coeff / sigma)
    return radius_A * normalized(dI)


def relative_error(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine."""
    pred = np.asarray(pred, dtype=float).reshape(3)
    actual = np.asarray(actual, dtype=float).reshape(3)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    return rel, float(np.dot(pred, actual) / denom)


def classify(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one closure row."""
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
    return "通过", "二维子空间扰动重求解通过"


def run_subspace_case(branch: dict[str, Any], label: str, task_direction: np.ndarray) -> dict[str, Any]:
    """Run one rank-2 subspace closure case."""
    dI = current_step_from_u2(branch, task_direction, SAFE_RADIUS_A)
    pred = np.asarray(branch["S"], dtype=float) @ dI
    I_new = np.asarray(branch["I0"], dtype=float) + dI
    q, info = p2c148.p2c134.solve_reduced(
        q0=np.zeros(branch["basis"].shape[1], dtype=float),
        y_base=branch["y"],
        basis=branch["basis"],
        I=I_new,
        funcs=branch["funcs"],
    )
    y_new = branch["y"] + branch["basis"] @ q
    out = p2c148.p2c134.eval_output(branch["funcs"], y_new, I_new)
    actual = out["tip_p"] - branch["base_out"]["tip_p"]
    rel, cosine = relative_error(pred, actual)
    centerline_change = float(np.max(np.linalg.norm(out["centerline"] - branch["base_out"]["centerline"], axis=1)))
    continuity, force_ratio, torque_ratio = p2c148.p2c134.load_metrics(branch["params"], y_new)
    row = {
        "direction": branch["name"],
        "subspace_direction": label,
        "radius_A": SAFE_RADIUS_A,
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
        "current_step_norm_A": float(np.linalg.norm(dI)),
        "current_step_A": dI,
        "predicted_tip_delta_m": pred,
        "actual_tip_delta_m": actual,
    }
    release_class, reason = classify(row)
    row["release_class"] = release_class
    row["reason"] = reason
    return row


def subspace_directions(branch: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Return task directions inside the audited 2D subspace."""
    u1 = np.asarray(branch["U"][:, 0], dtype=float)
    u2 = np.asarray(branch["U"][:, 1], dtype=float)
    return [
        ("第一主方向", u1),
        ("第二主方向", u2),
        ("二维混合方向", normalized(u1 + u2)),
    ]


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot closure metrics."""
    configure_matplotlib_cjk()
    labels = [f"{r['direction']}\n{r['subspace_direction']}" for r in rows]
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if r["release_class"] == "通过" else "#e76f51" for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 10.0), sharex=True)
    axes[0].bar(x, [r["linear_rel_err"] for r in rows], color=colors)
    axes[0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("线性误差")
    axes[0].set_title("二维有效子空间小步闭环门禁")
    axes[1].bar(x, [r["centerline_max_change_m"] for r in rows], color=colors)
    axes[1].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("中心线漂移/m")
    axes[2].bar(x, [r["force_ratio"] for r in rows], color=colors)
    axes[2].axhline(FORCE_RATIO_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("载荷传播比例")
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_2d_subspace_closure_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 方向 | 线性误差 | 实际位移/um | 中心线漂移/m | 载荷比 | 结论 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['subspace_direction']} | {row['linear_rel_err']:.3e} | "
            f"{row['actual_norm_um']:.3e} | {row['centerline_max_change_m']:.3e} | "
            f"{row['force_ratio']:.3f} | {row['release_class']} |"
        )
    text = f"""# P2c152 两点磁控二维有效子空间小步闭环验证

## 实验目的

P2c151 已导出二维有效子空间的数值基。本轮使用这些基向量生成小电流扰动，并进行真实重求解，验证二维子空间内的局部线性预测是否能通过物理一致性门禁。

## 方法

每个工作点测试三个二维子空间方向：第一主方向、第二主方向和二者等权混合方向。电流步长固定为 0.001 A。对每个扰动，记录预测末端位移与重求解末端位移的相对误差、中心线分支漂移、接口连续残差和载荷传播比例。

## 结果

{chr(10).join(table)}

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。最大线性化误差为 {summary['max_linear_rel_err']:.3e}，最大中心线漂移为 {summary['max_centerline_change_m']:.3e} m，最大载荷传播比例为 {summary['max_force_ratio']:.3f}。

## 实验结论

若全部样本通过，则二维有效子空间可升级为“小步诊断步可生成”候选；仍不得扩展到完整三维位置或完整末端位姿。若存在失败样本，则应保留只读诊断状态，并对失败方向做边界夹逼。

当前判定：{summary['decision']}。

## 下一步计划

若本轮通过，下一轮应更新 P2c150 接口契约，把“数值基缺失”替换为“二维小步已闭环验证”，并继续增加跨半径复核。若未通过，则固定失败方向做更小半径夹逼，判断失败来自步长过大、分支漂移还是模型线性化不足。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the 2D subspace closure audit."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        branch = p2c148.build_candidate_branch(base, name, direction)
        for label, task_direction in subspace_directions(branch):
            row = run_subspace_case(branch, label, task_direction)
            rows.append(row)
            if row["release_class"] != "通过":
                failures.append(
                    {
                        "direction": row["direction"],
                        "subspace_direction": row["subspace_direction"],
                        "failure_class": row["reason"],
                        "linear_rel_err": row["linear_rel_err"],
                        "centerline_max_change_m": row["centerline_max_change_m"],
                        "force_ratio": row["force_ratio"],
                    }
                )
    wall_s = time.perf_counter() - t0
    pass_count = sum(row["release_class"] == "通过" for row in rows)
    total_solve = sum(float(row["solve_steady_s"]) for row in rows)
    summary = {
        "experiment_id": "20260630123000_p2c152_n2_2d_subspace_closure",
        "tested_count": len(rows),
        "pass_count": pass_count,
        "all_passed": pass_count == len(rows),
        "safe_radius_A": SAFE_RADIUS_A,
        "effective_subspace_dimension": 2,
        "max_linear_rel_err": max(float(row["linear_rel_err"]) for row in rows),
        "max_centerline_change_m": max(float(row["centerline_max_change_m"]) for row in rows),
        "max_force_ratio": max(float(row["force_ratio"]) for row in rows),
        "max_continuity_norm": max(float(row["continuity_norm"]) for row in rows),
        "total_solve_steady_s": total_solve,
        "wall_s": wall_s,
        "decision": "二维小步诊断步可生成候选" if pass_count == len(rows) else "保持二维只读诊断，需失败归因",
        "outer_ik_allowed": False,
    }
    write_csv(OUT_DIR / "closure_rows.csv", jsonable(rows))
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(failure), ensure_ascii=False) + "\n" for failure in failures),
        encoding="utf-8",
    )
    plot(rows)
    write_record(summary, rows)


if __name__ == "__main__":
    main()
