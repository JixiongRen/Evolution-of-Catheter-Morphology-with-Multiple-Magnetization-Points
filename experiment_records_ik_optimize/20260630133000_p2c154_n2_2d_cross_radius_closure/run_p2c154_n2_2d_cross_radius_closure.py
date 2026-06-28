"""Audit n2 effective-subspace closure over multiple current radii."""

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


RADII_A = (3.0e-4, 5.0e-4, 1.0e-3, 1.5e-3)
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


def run_subspace_case(branch: dict[str, Any], label: str, task_direction: np.ndarray, radius_A: float) -> dict[str, Any]:
    """Run one rank-2 subspace closure case."""
    dI = current_step_from_u2(branch, task_direction, radius_A)
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
        "radius_A": radius_A,
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
    by_radius: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_radius.setdefault(float(row["radius_A"]), []).append(row)
    radii = sorted(by_radius)
    labels = [f"{radius:.4f}" for radius in radii]
    x = np.arange(len(radii))
    pass_rates = [sum(r["release_class"] == "通过" for r in by_radius[radius]) / len(by_radius[radius]) for radius in radii]
    max_linear = [max(float(r["linear_rel_err"]) for r in by_radius[radius]) for radius in radii]
    max_drift = [max(float(r["centerline_max_change_m"]) for r in by_radius[radius]) for radius in radii]
    max_force = [max(float(r["force_ratio"]) for r in by_radius[radius]) for radius in radii]
    colors = ["#2a9d8f" if rate == 1.0 else "#e76f51" for rate in pass_rates]
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)
    axes[0].bar(x, pass_rates, color=colors)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("通过率")
    axes[0].set_title("二维有效子空间跨半径闭环门禁")
    axes[1].bar(x, max_linear, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("最大线性误差")
    axes[1].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].bar(x, max_drift, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("最大中心线漂移/m")
    axes[2].axhline(CENTERLINE_GATE_M, color="black", linestyle="--", linewidth=1)
    axes[3].bar(x, max_force, color=colors)
    axes[3].set_ylabel("最大载荷传播比例")
    axes[3].axhline(FORCE_RATIO_GATE, color="black", linestyle="--", linewidth=1)
    axes[3].set_xticks(x, labels)
    axes[3].set_xlabel("电流步半径/A")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_2d_cross_radius_closure_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 方向 | 半径/A | 线性误差 | 实际位移/um | 中心线漂移/m | 载荷比 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['subspace_direction']} | {row['radius_A']:.4f} | "
            f"{row['linear_rel_err']:.3e} | "
            f"{row['actual_norm_um']:.3e} | {row['centerline_max_change_m']:.3e} | "
            f"{row['force_ratio']:.3f} | {row['release_class']} |"
        )
    by_radius = summary["by_radius"]
    radius_table = [
        "| 半径/A | 通过数/总数 | 最大线性误差 | 最大中心线漂移/m | 最大载荷比 | 判定 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for radius, stats in by_radius.items():
        radius_table.append(
            f"| {float(radius):.4f} | {stats['pass_count']}/{stats['tested_count']} | "
            f"{stats['max_linear_rel_err']:.3e} | {stats['max_centerline_change_m']:.3e} | "
            f"{stats['max_force_ratio']:.3f} | {stats['decision']} |"
        )
    text = f"""# P2c154 两点磁控二维有效子空间跨半径闭环验证

## 实验目的

P2c153 已允许二维有效子空间内生成小电流诊断步。本轮在多个电流步半径下重复真实重求解闭环，确定保守半径边界，避免把 0.001 A 单点证据过度外推。

## 方法

每个工作点测试三个二维子空间方向：第一主方向、第二主方向和二者等权混合方向。电流步半径取 0.0003 A、0.0005 A、0.0010 A、0.0015 A。对每个扰动，记录预测末端位移与重求解末端位移的相对误差、中心线分支漂移、接口连续残差和载荷传播比例。

## 结果

{chr(10).join(radius_table)}

## 明细

{chr(10).join(table)}

## 数据分析

通过样本数为 {summary['pass_count']} / {summary['tested_count']}。最大线性化误差为 {summary['max_linear_rel_err']:.3e}，最大中心线漂移为 {summary['max_centerline_change_m']:.3e} m，最大载荷传播比例为 {summary['max_force_ratio']:.3f}。

## 实验结论

本轮给出二维诊断步的保守半径边界。即使全部半径通过，该结论仍只适用于二维有效子空间、同一物理解分支和小扰动范围，不能扩展到完整三维位置或完整末端位姿。

当前判定：{summary['decision']}。

## 下一步计划

下一轮应根据保守半径更新 P2c153 接口契约。若 0.0015 A 仍通过，可暂定 0.0015 A 为二维诊断半径上界；若失败，则固定失败半径和方向继续夹逼。

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
            for radius in RADII_A:
                row = run_subspace_case(branch, label, task_direction, radius)
                rows.append(row)
                if row["release_class"] != "通过":
                    failures.append(
                        {
                            "direction": row["direction"],
                            "subspace_direction": row["subspace_direction"],
                            "radius_A": row["radius_A"],
                            "failure_class": row["reason"],
                            "linear_rel_err": row["linear_rel_err"],
                            "centerline_max_change_m": row["centerline_max_change_m"],
                            "force_ratio": row["force_ratio"],
                        }
                    )
    wall_s = time.perf_counter() - t0
    pass_count = sum(row["release_class"] == "通过" for row in rows)
    total_solve = sum(float(row["solve_steady_s"]) for row in rows)
    by_radius = {}
    for radius in RADII_A:
        radius_rows = [row for row in rows if abs(float(row["radius_A"]) - radius) < 1.0e-12]
        radius_pass = sum(row["release_class"] == "通过" for row in radius_rows)
        by_radius[f"{radius:.4f}"] = {
            "tested_count": len(radius_rows),
            "pass_count": radius_pass,
            "all_passed": radius_pass == len(radius_rows),
            "max_linear_rel_err": max(float(row["linear_rel_err"]) for row in radius_rows),
            "max_centerline_change_m": max(float(row["centerline_max_change_m"]) for row in radius_rows),
            "max_force_ratio": max(float(row["force_ratio"]) for row in radius_rows),
            "decision": "通过" if radius_pass == len(radius_rows) else "不通过",
        }
    passed_radii = [radius for radius in RADII_A if by_radius[f"{radius:.4f}"]["all_passed"]]
    conservative_radius = max(passed_radii) if passed_radii else 0.0
    failure_by_class = collections.Counter(failure["failure_class"] for failure in failures)
    failure_by_radius: dict[str, dict[str, int]] = {}
    for failure in failures:
        radius_key = f"{float(failure['radius_A']):.4f}"
        failure_by_radius.setdefault(radius_key, {})
        failure_by_radius[radius_key][failure["failure_class"]] = (
            failure_by_radius[radius_key].get(failure["failure_class"], 0) + 1
        )
    summary = {
        "experiment_id": "20260630133000_p2c154_n2_2d_cross_radius_closure",
        "tested_count": len(rows),
        "pass_count": pass_count,
        "all_passed": pass_count == len(rows),
        "tested_radii_A": list(RADII_A),
        "conservative_passed_radius_A": conservative_radius,
        "effective_subspace_dimension": 2,
        "max_linear_rel_err": max(float(row["linear_rel_err"]) for row in rows),
        "max_centerline_change_m": max(float(row["centerline_max_change_m"]) for row in rows),
        "max_force_ratio": max(float(row["force_ratio"]) for row in rows),
        "max_continuity_norm": max(float(row["continuity_norm"]) for row in rows),
        "total_solve_steady_s": total_solve,
        "wall_s": wall_s,
        "by_radius": by_radius,
        "failure_by_class": dict(failure_by_class),
        "failure_by_radius": failure_by_radius,
        "decision": "扩大二维诊断半径候选" if pass_count == len(rows) else "使用最大全通过半径作为保守半径",
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
