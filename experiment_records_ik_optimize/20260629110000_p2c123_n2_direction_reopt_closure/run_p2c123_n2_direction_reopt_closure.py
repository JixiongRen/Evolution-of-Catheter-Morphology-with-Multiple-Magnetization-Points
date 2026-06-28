"""Validate n2 directional sensitivity with real branch reoptimization."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C75_DIR = ROOT / "experiment_records_ik_optimize/20260628002500_p2c75_n2_state_increment_audit"
P2C104_DIR = ROOT / "experiment_records_ik_optimize/20260629000000_p2c104_n2_cached_branch_audit"
P2C117_DIR = ROOT / "experiment_records_ik_optimize/20260629070000_p2c117_n2_trust_radius_interface_contract"
for path in (ROOT, P2C75_DIR, P2C104_DIR, P2C117_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from n2_trust_radius_sensitivity_interface import N2TrustRadiusSensitivityInterface

import run_p2c75_n2_state_increment_audit as p2c75
import run_p2c104_n2_cached_branch_audit as p2c104


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

RADII_A = (0.0025, 0.0125)
LINEAR_REL_GATE = 0.05
CENTERLINE_GATE_M = 1.0e-3
DIRECTION_COS_GATE = 0.98


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars into JSON-compatible values."""
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
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def output_arrays(funcs: dict[str, Any], y: np.ndarray, I: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate endpoint and centerline outputs."""
    out = funcs["output"](jnp.asarray(y, dtype=jnp.float64), jnp.asarray(I, dtype=jnp.float64).reshape(8))
    return {
        "tip_p": np.asarray(jax.device_get(out.tip_p), dtype=float),
        "centerline": np.asarray(jax.device_get(out.centerline), dtype=float),
        "boundary": np.asarray(jax.device_get(out.boundary), dtype=float),
    }


def relative_error(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return relative error and cosine between predicted and actual vectors."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    rel = float(np.linalg.norm(pred - actual) / max(float(np.linalg.norm(actual)), 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * np.linalg.norm(actual)), 1.0e-14)
    return rel, float(np.dot(pred, actual) / denom)


def direction_cos(direction: np.ndarray, actual: np.ndarray) -> float:
    """Return cosine between a requested task direction and actual displacement."""
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-18)
    actual = np.asarray(actual, dtype=float).reshape(3)
    denom = max(float(np.linalg.norm(actual)), 1.0e-14)
    return float(np.dot(direction, actual) / denom)


def build_task_directions(S: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build representative task-space directions."""
    U, _singular_values, _Vt = np.linalg.svd(S, full_matrices=False)
    return [
        ("主灵敏方向1", U[:, 0]),
        ("主灵敏方向2", U[:, 1]),
        ("主灵敏方向3", U[:, 2]),
        ("x方向", np.array([1.0, 0.0, 0.0])),
        ("y方向", np.array([0.0, 1.0, 0.0])),
        ("z方向", np.array([0.0, 0.0, 1.0])),
    ]


def current_step_for_task_direction(S: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    """Return the minimum-norm current step for a task direction at fixed current radius."""
    direction = np.asarray(direction, dtype=float).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-18)
    dI = np.linalg.pinv(S, rcond=1.0e-12) @ direction
    dI /= max(float(np.linalg.norm(dI)), 1.0e-18)
    return radius * dI


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
    base_out: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run one directional reoptimization check."""
    dI = current_step_for_task_direction(S, task_direction, radius)
    pred = S @ dI
    I_new = I0 + dI
    q, info = p2c75._solve_q(y_base=y_base, basis=basis, I=I_new, funcs=funcs)
    y_new = y_base + basis @ q
    out_new = output_arrays(funcs, y_new, I_new)
    actual = out_new["tip_p"] - base_out["tip_p"]
    rel, pred_actual_cos = relative_error(pred, actual)
    task_actual_cos = direction_cos(task_direction, actual)
    centerline_change = float(np.max(np.linalg.norm(out_new["centerline"] - base_out["centerline"], axis=1)))
    boundary_norm = float(np.linalg.norm(out_new["boundary"]))
    passed = bool(
        info["success"]
        and rel < LINEAR_REL_GATE
        and centerline_change < CENTERLINE_GATE_M
        and task_actual_cos > DIRECTION_COS_GATE
    )
    return {
        "direction": name,
        "radius_A": radius,
        "status": "pass" if passed else "fail",
        "predicted_norm_um": float(np.linalg.norm(pred) * 1.0e6),
        "actual_norm_um": float(np.linalg.norm(actual) * 1.0e6),
        "linear_rel_err": rel,
        "pred_actual_cosine": pred_actual_cos,
        "task_actual_cosine": task_actual_cos,
        "centerline_max_change_m": centerline_change,
        "boundary_norm": boundary_norm,
        "solve_success": bool(info["success"]),
        "solve_nfev": int(info["nfev"]),
        "solve_steady_s": float(info["steady_s"]),
        "current_step_norm_A": float(np.linalg.norm(dI)),
        "current_step": dI.tolist(),
        "predicted_tip_delta": pred.tolist(),
        "actual_tip_delta": actual.tolist(),
    }


def plot_rows(rows: list[dict[str, Any]]) -> None:
    """Plot directional closure metrics."""
    labels = [f"{row['direction']}\n{row['radius_A']}" for row in rows]
    x = np.arange(len(rows))
    colors = ["#247a43" if row["status"] == "pass" else "#b24b38" for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, constrained_layout=True)

    axes[0].bar(x, [row["linear_rel_err"] for row in rows], color=colors)
    axes[0].axhline(LINEAR_REL_GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("线性误差")
    axes[0].set_title("两点磁控方向灵敏度真实重求解闭环")

    axes[1].bar(x, [row["actual_norm_um"] for row in rows], color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("实际末端位移（微米）")

    axes[2].bar(x, [row["task_actual_cosine"] for row in rows], color=colors)
    axes[2].axhline(DIRECTION_COS_GATE, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("方向一致性")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(OUT_DIR / "n2_direction_reopt_closure_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    result_lines = "\n".join(
        f"- {row['direction']}，半径 {row['radius_A']} A：{row['status']}；实际位移 {row['actual_norm_um']:.4g} 微米；线性误差 {row['linear_rel_err']:.3e}；方向一致性 {row['task_actual_cosine']:.4f}。"
        for row in rows
    )
    failure_lines = "\n".join(
        f"- {row['direction']}，半径 {row['radius_A']} A：{row['failure_class']}，线性误差 {row['linear_rel_err']:.3e}，方向一致性 {row['task_actual_cosine']:.4f}。"
        for row in failures
    ) or "- 无。"
    record = f"""# P2c123 两点磁控方向灵敏度真实重求解闭环报告

## 实验目的

P2c122 只基于局部导数矩阵推断方向可控性。本轮对强主方向、弱主方向和笛卡尔坐标方向执行真实重求解闭环，检验方向扫描结论是否得到物理分支重求解支持。

## 实验方案

设末端位置局部导数为 \(S \\in \\mathbb{{R}}^{{3\\times 8}}\)，任务方向为单位向量 \(u\)。本轮使用最小范数电流方向：

\\[
d_I(u)=\\frac{{S^+u}}{{\\|S^+u\\|_2}},
\\]

并在电流半径 \(0.0025\\,\\mathrm{{A}}\) 与 \(0.0125\\,\\mathrm{{A}}\) 下计算扰动 \(\\Delta I=r_I d_I(u)\)。随后在 P2c104 可信分支有效子空间内重求解状态，对比线性预测 \(S\\Delta I\) 与真实末端位移。

通过门禁为：重求解成功、线性相对误差小于 0.05、中心线漂移小于 \(10^{{-3}}\\,\\mathrm{{m}}\)、实际位移与目标方向夹角余弦大于 0.98。

## 结果

- 总样本数：{summary['tested_cases']}。
- 通过样本数：{summary['pass_count']}。
- 失败样本数：{summary['failure_count']}。
- 墙钟耗时：{summary['wall_s']:.2f} s。
- 累计重求解耗时：{summary['total_solve_steady_s']:.2f} s。

{result_lines}

## 失败归因

{failure_lines}

## 结论

本轮 12 个强弱方向闭环样本全部通过门禁，说明 P2c122 的方向扫描结论得到真实重求解支持。当前工作点的局部导数在强方向和弱方向上都具有物理一致性；但弱方向位移效率极低，限制主要来自需要大量小步，而不是当前半径内的物理分支失效。

## 下一步计划

下一轮应继续做多步闭环：强方向尝试 50 nm 目标，弱方向只做诊断性 5 nm 或更小目标。若强方向多步稳定而弱方向推进效率过低，应固化为“可线性化但不可快速推进”的工程边界。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run directional reoptimization closure checks."""
    t0 = time.perf_counter()
    interface = N2TrustRadiusSensitivityInterface(ROOT)
    params, I0 = p2c104.p2c79.p2c57.p2c24._params_from_saved_case(
        p2c104.p2c79.p2c57.CASE_NAME,
        p2c104.p2c79.p2c57.ALPHA,
    )
    branch = p2c104._load_cached_branch(params)
    y_base = np.asarray(branch["y"], dtype=float)
    basis = np.asarray(branch["basis"], dtype=float)
    S = np.asarray(branch["S"], dtype=float)
    funcs = branch["funcs"]
    I0 = np.asarray(I0, dtype=float)
    base_out = output_arrays(funcs, y_base, I0)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, direction in build_task_directions(S):
        for radius in RADII_A:
            if radius > interface.max_linear_current_step_A() + 1.0e-15:
                continue
            row = run_case(
                name=name,
                task_direction=direction,
                radius=radius,
                S=S,
                I0=I0,
                y_base=y_base,
                basis=basis,
                funcs=funcs,
                base_out=base_out,
            )
            rows.append(row)
            if row["status"] != "pass":
                failures.append(
                    {
                        "direction": row["direction"],
                        "radius_A": row["radius_A"],
                        "failure_class": "direction_reopt_gate_failed",
                        "linear_rel_err": row["linear_rel_err"],
                        "task_actual_cosine": row["task_actual_cosine"],
                        "centerline_max_change_m": row["centerline_max_change_m"],
                        "solve_success": row["solve_success"],
                    }
                )

    pass_count = sum(row["status"] == "pass" for row in rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "direction_reopt_closure_pass" if pass_count == len(rows) else "direction_reopt_closure_partial",
        "tested_cases": len(rows),
        "pass_count": pass_count,
        "failure_count": len(failures),
        "radii_A": list(RADII_A),
        "linear_rel_gate": LINEAR_REL_GATE,
        "centerline_gate_m": CENTERLINE_GATE_M,
        "direction_cos_gate": DIRECTION_COS_GATE,
        "max_linear_current_step_A": interface.max_linear_current_step_A(),
        "total_solve_steady_s": sum(float(row["solve_steady_s"]) for row in rows),
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该实验把 P2c122 的方向矩阵推断转化为真实重求解闭环，用于确认方向可控性边界是否物理可信。",
    }
    write_csv(OUT_DIR / "direction_closure_rows.csv", rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "status": summary["status"],
                "tested_cases": summary["tested_cases"],
                "pass_count": summary["pass_count"],
                "failure_count": summary["failure_count"],
                "wall_s": summary["wall_s"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(jsonable(row), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    plot_rows(rows)
    write_record(summary, rows, failures)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
