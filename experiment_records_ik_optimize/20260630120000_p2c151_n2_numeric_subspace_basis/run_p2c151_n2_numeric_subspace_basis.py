"""Export numeric bases for the n2 effective-subspace candidate."""

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


def export_branch(branch: dict[str, Any]) -> dict[str, Any]:
    """Export a compact numeric basis record for one branch."""
    S = np.asarray(branch["S"], dtype=float)
    U = np.asarray(branch["U"], dtype=float)
    Vt = np.asarray(branch["Vt"], dtype=float)
    singular_values = np.asarray(branch["singular_values"], dtype=float)
    task_basis = U[:, :2]
    current_basis = Vt[:2, :]
    task_projector = task_basis @ task_basis.T
    reconstructed = task_basis @ np.diag(singular_values[:2]) @ current_basis
    residual = S - reconstructed
    return {
        "direction": branch["name"],
        "safe_radius_A": SAFE_RADIUS_A,
        "terminal_quantity": "position",
        "task_dimension_full": 3,
        "effective_subspace_dimension": 2,
        "S_matrix_m_per_A": S,
        "task_basis_U2": task_basis,
        "current_basis_V2": current_basis,
        "task_projector_U2U2T": task_projector,
        "singular_values_m_per_A": singular_values,
        "condition_full": float(singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny)),
        "condition_effective_2d": float(singular_values[0] / max(float(singular_values[1]), np.finfo(float).tiny)),
        "discarded_singular_value": float(singular_values[2]),
        "discarded_fraction": float(singular_values[2] / max(float(singular_values[0]), np.finfo(float).tiny)),
        "rank2_reconstruction_error_norm": float(np.linalg.norm(residual)),
        "rank2_reconstruction_error_rel": float(np.linalg.norm(residual) / max(float(np.linalg.norm(S)), np.finfo(float).tiny)),
        "base_current_A": np.asarray(branch["I0"], dtype=float),
        "base_tip_position_m": np.asarray(branch["base_out"]["tip_p"], dtype=float),
        "diagnostic_step_formula": "For a small desired terminal-position increment dp, use dp2=U2@U2.T@dp and dI=V2.T@diag(1/sigma1,1/sigma2)@U2.T@dp, then enforce ||dI||<=safe_radius_A.",
    }


def build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build summary rows for the exported bases."""
    rows = []
    for record in records:
        rows.append(
            {
                "direction": record["direction"],
                "safe_radius_A": record["safe_radius_A"],
                "sigma1_m_per_A": record["singular_values_m_per_A"][0],
                "sigma2_m_per_A": record["singular_values_m_per_A"][1],
                "sigma3_m_per_A": record["singular_values_m_per_A"][2],
                "condition_full": record["condition_full"],
                "condition_effective_2d": record["condition_effective_2d"],
                "discarded_fraction": record["discarded_fraction"],
                "rank2_reconstruction_error_rel": record["rank2_reconstruction_error_rel"],
                "basis_available": True,
                "control_step_allowed": False,
                "reason": "数值基已保存；控制步仍需二维闭环重求解验证",
            }
        )
    return rows


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot singular-value structure for the exported bases."""
    configure_matplotlib_cjk()
    labels = [row["direction"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for idx, key in enumerate(("sigma1_m_per_A", "sigma2_m_per_A", "sigma3_m_per_A")):
        axes[0].bar(x + (idx - 1) * width, [row[key] for row in rows], width=width, label=f"第{idx + 1}奇异值")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("末端位置响应 / (m/A)")
    axes[0].set_title("两点磁控局部灵敏度奇异值")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, [row["condition_effective_2d"] for row in rows], color="#2a9d8f", label="二维有效子空间")
    axes[1].bar(
        x,
        [row["condition_full"] for row in rows],
        color="#e76f51",
        alpha=0.55,
        label="完整三维",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("条件数")
    axes[1].set_title("二维子空间与完整三维条件数")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_numeric_subspace_basis_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    table = [
        "| 工作点 | 第一奇异值 | 第二奇异值 | 第三奇异值 | 二维条件数 | 完整三维条件数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['sigma1_m_per_A']:.3e} | {row['sigma2_m_per_A']:.3e} | "
            f"{row['sigma3_m_per_A']:.3e} | {row['condition_effective_2d']:.2f} | {row['condition_full']:.2f} |"
        )
    text = f"""# P2c151 两点磁控二维有效子空间数值基导出

## 实验目的

P2c150 已明确当前结果只能作为二维有效子空间只读诊断。本轮补充每个候选工作点的数值基向量，包括任务空间二维基、电流空间二维基、任务投影矩阵和局部灵敏度矩阵，为下一轮二维子空间小步闭环验证提供可复核输入。

## 方法

复用 P2c148 的分支重建与局部灵敏度矩阵构造流程。对每个候选工作点得到局部矩阵 \(S \\in \\mathbb{{R}}^{{3\\times 8}}\)，并作奇异值分解：

\\[
S = U \\Sigma V^\\top.
\\]

保留前两个主响应方向：

\\[
U_2 = [u_1,u_2], \\quad V_2 = [v_1,v_2]^\\top.
\\]

对小末端位移需求 \(\\Delta p\)，二维投影为：

\\[
\\Delta p_2 = U_2 U_2^\\top \\Delta p.
\\]

诊断用电流增量公式为：

\\[
\\Delta I = V_2^\\top \\operatorname{{diag}}(1/\\sigma_1,1/\\sigma_2) U_2^\\top \\Delta p.
\\]

该公式本轮只作为可复核数值接口，不声明可用于控制；控制使用仍需下一轮扰动重求解闭环验证。

## 结果

{chr(10).join(table)}

## 数据分析

五个工作点均已导出二维数值基。二维有效子空间条件数约为 {summary['condition_2d_min']:.2f} 至 {summary['condition_2d_max']:.2f}，明显优于完整三维条件数。第三奇异值相对第一奇异值很小，解释了 P2c148 中第三主方向弱信号和线性化误差放大的现象。

## 实验结论

当前已经具备二维有效子空间的数值表达，但尚未具备控制接口发布资格。原因是数值公式尚未通过独立的小步任务闭环重求解验证。下一轮必须使用这些基向量生成二维子空间内的目标位移和电流步，并检查重求解后的末端位移、分支漂移、载荷传播和线性化误差。

## 下一步计划

开展 P2c152：二维有效子空间小步闭环验证。只测试 \(U_2\) 张成空间内的任务步，不测试完整三维位置，不接外层逆运动学。若通过，再升级接口状态为“二维诊断步可生成”；若失败，则归因到模型线性化、分支漂移或电流映射误差。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run basis export."""
    t0 = time.perf_counter()
    base = p2c148.p2c145.p2c144.p2c89._recover_trusted()
    records: list[dict[str, Any]] = []
    for name, direction in p2c148.p2c145.p2c144.candidate_directions():
        branch = p2c148.build_candidate_branch(base, name, direction)
        records.append(export_branch(branch))
    wall_s = time.perf_counter() - t0
    rows = build_rows(records)
    summary = {
        "experiment_id": "20260630120000_p2c151_n2_numeric_subspace_basis",
        "candidate_count": len(records),
        "basis_available_count": len(records),
        "safe_radius_A": SAFE_RADIUS_A,
        "effective_subspace_dimension": 2,
        "numeric_basis_available": True,
        "control_step_allowed": False,
        "outer_ik_allowed": False,
        "condition_2d_min": min(float(row["condition_effective_2d"]) for row in rows),
        "condition_2d_max": max(float(row["condition_effective_2d"]) for row in rows),
        "condition_full_min": min(float(row["condition_full"]) for row in rows),
        "condition_full_max": max(float(row["condition_full"]) for row in rows),
        "wall_s": wall_s,
        "source_experiment": "20260630103000_p2c148_n2_all_principal_safe_radius",
        "next_required_artifact": "2D effective-subspace reoptimized closure audit",
    }
    (OUT_DIR / "numeric_subspace_basis.json").write_text(
        json.dumps(jsonable({"summary": summary, "records": records}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_DIR / "basis_rows.csv", jsonable(rows))
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(
            {
                "failure_class": "控制步尚未闭环验证",
                "interpretation": "数值基已导出，但未完成二维子空间内扰动重求解闭环，暂不能作为外层控制接口。",
                "next_action": "使用导出的 U2/V2 生成二维任务步并重求解验证。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot(rows)
    write_record(summary, rows)


if __name__ == "__main__":
    main()
