"""Audit n2 effective subspaces by removing weak singular directions."""

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
P2C69_DIR = ROOT / "experiment_records_ik_optimize" / "20260627230500_p2c69_n2_condition_precondition_audit"
for path in (ROOT, P2C57_DIR, P2C69_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c69_n2_condition_precondition_audit as p2c69

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    return p2c57._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _condition(mat: np.ndarray) -> float:
    s = np.linalg.svd(np.asarray(mat, dtype=float), compute_uv=False)
    if not s.size or s[-1] <= np.finfo(float).tiny:
        return float("inf")
    return float(s[0] / s[-1])


def _orthogonal_complement(vectors: np.ndarray, dim: int) -> np.ndarray:
    if vectors.size == 0:
        return np.eye(dim)
    q, _ = np.linalg.qr(np.asarray(vectors, dtype=float).T, mode="complete")
    return q[:, vectors.shape[0] :]


def _audit_subspaces(J: np.ndarray, r: np.ndarray, n_flex: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    u, s, vt = np.linalg.svd(J, full_matrices=False)
    pos_slice = p2c69._block_slices(n_flex)["terminal_position"]
    J_pos = J[pos_slice, :]
    rows = []
    weak_rows = []
    for remove_k in range(0, 5):
        weak_vectors = vt[-remove_k:] if remove_k else np.zeros((0, J.shape[1]))
        basis = _orthogonal_complement(weak_vectors, J.shape[1])
        J_eff = J @ basis
        J_pos_eff = J_pos @ basis
        s_eff = np.linalg.svd(J_eff, compute_uv=False)
        pos_s = np.linalg.svd(J_pos_eff, compute_uv=False)
        grad_eff = basis.T @ (J.T @ r)
        rows.append(
            {
                "removed_weak_directions": remove_k,
                "subspace_dim": int(basis.shape[1]),
                "condition": _condition(J_eff),
                "largest_singular": float(s_eff[0]) if s_eff.size else 0.0,
                "smallest_singular": float(s_eff[-1]) if s_eff.size else 0.0,
                "projected_stationarity_norm": float(np.linalg.norm(grad_eff)),
                "terminal_position_rank": int(np.linalg.matrix_rank(J_pos_eff)),
                "terminal_position_condition": _condition(J_pos_eff),
                "terminal_position_smallest_singular": float(pos_s[-1]) if pos_s.size else 0.0,
                "candidate_gate": bool(
                    basis.shape[1] >= 15
                    and _condition(J_eff) < 1.0e7
                    and float(np.linalg.norm(grad_eff)) <= 1.0e-5
                    and int(np.linalg.matrix_rank(J_pos_eff)) >= 3
                ),
            }
        )
    for idx in range(1, min(6, len(s)) + 1):
        vec = vt[-idx]
        blocks = vec.reshape((n_flex, 6))
        weak_rows.append(
            {
                "weak_order": idx,
                "singular_value": float(s[-idx]),
                "condition_if_removed_through_this": float(s[0] / s[-idx - 1]) if idx < len(s) else float("inf"),
                "segment0_force_energy": float(np.linalg.norm(blocks[0, :3]) ** 2),
                "segment0_torque_energy": float(np.linalg.norm(blocks[0, 3:]) ** 2),
                "segment1_force_energy": float(np.linalg.norm(blocks[1, :3]) ** 2),
                "segment1_torque_energy": float(np.linalg.norm(blocks[1, 3:]) ** 2),
                "segment2_force_energy": float(np.linalg.norm(blocks[2, :3]) ** 2),
                "segment2_torque_energy": float(np.linalg.norm(blocks[2, 3:]) ** 2),
                "terminal_position_projection": float(np.linalg.norm(J_pos @ vec)),
                "full_residual_projection": float(np.linalg.norm(J @ vec)),
            }
        )
    return rows, weak_rows, s


def _plot(rows: list[dict[str, Any]], weak_rows: list[dict[str, Any]], svals: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = [row["removed_weak_directions"] for row in rows]
    axes[0].plot(x, [row["condition"] for row in rows], marker="o", label="全残差")
    axes[0].plot(x, [row["terminal_position_condition"] for row in rows], marker="s", label="末端位置")
    axes[0].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("移除最弱方向数")
    axes[0].set_ylabel("条件数")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.arange(1, len(svals) + 1), svals, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("奇异值序号")
    axes[1].set_ylabel("奇异值")
    axes[1].grid(alpha=0.25)
    w = weak_rows[0]
    labels = ["0内力", "0内矩", "1内力", "1内矩", "2内力", "2内矩"]
    vals = [
        w["segment0_force_energy"],
        w["segment0_torque_energy"],
        w["segment1_force_energy"],
        w["segment1_torque_energy"],
        w["segment2_force_energy"],
        w["segment2_torque_energy"],
    ]
    axes[2].bar(labels, vals, color="#70ad47")
    axes[2].set_ylabel("第一最弱方向能量")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c71 n2 有效子空间审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_effective_subspace_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(base_row: dict[str, Any], rows: list[dict[str, Any]], weak_rows: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 移除方向数 | 子空间维度 | 条件数 | 投影一阶最优性 | 末端位置秩 | 末端位置条件数 | 候选过门 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['removed_weak_directions']} | {row['subspace_dim']} | {row['condition']:.3e} | "
            f"{row['projected_stationarity_norm']:.3e} | {row['terminal_position_rank']} | "
            f"{row['terminal_position_condition']:.3e} | {row['candidate_gate']} |"
        )
    weak = weak_rows[0]
    passing = [row for row in rows if row["candidate_gate"]]
    best = passing[0] if passing else min(rows, key=lambda row: row["condition"])
    return "\n".join(
        [
            "# P2c71 n2 有效子空间审计记录",
            "",
            "## 实验目的",
            "",
            "P2c70 表明块白化重求解会偏离原物理近门分支，不能直接得到可发布三维灵敏度。本轮在原 P2c59/P2c69 近门分支上做静态有效子空间审计，判断去除最弱病态方向后是否存在可继续验证的局部位置灵敏度子空间。",
            "",
            "## 实验方案",
            "",
            "恢复 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4` 近门分支，计算原物理残差雅可比矩阵。依次移除 0 到 4 个最弱右奇异方向，评估投影雅可比条件数、投影一阶最优性和三维末端位置秩。该轮不声称发布，只筛选是否存在值得进入分支一致性和扰动重求解的有效子空间。",
            "",
            "## 基线状态",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 物理条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续误差：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 有效子空间结果",
            "",
            *table,
            "",
            "## 最弱方向解释",
            "",
            f"第一最弱方向的末端位置投影范数为 `{weak['terminal_position_projection']:.3e}`，全残差投影范数为 `{weak['full_residual_projection']:.3e}`。其能量主要位于第 0 段内矩 `{weak['segment0_torque_energy']:.3e}` 和第 1 段内矩 `{weak['segment1_torque_energy']:.3e}`，内力能量很小。这说明当前病态方向主要是分段内矩组合，而不是末端位置主控方向。",
            "",
            "## 结论",
            "",
            f"最佳候选为移除 `{best['removed_weak_directions']}` 个最弱方向后的 `{best['subspace_dim']}` 维子空间，条件数 `{best['condition']:.3e}`，投影一阶最优性 `{best['projected_stationarity_norm']:.3e}`，末端位置秩 `{best['terminal_position_rank']}`。",
            "",
            "若该候选过门，则下一步应在该子空间内做多初值分支一致性和小扰动重求解；若未过门，则说明 n2 当前分支不能发布完整三维或高维有效子空间灵敏度，需要继续缩小任务维度或归因为不可辨识。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在候选过门，构造 P2c72 子空间扰动重求解，先测 eps=0.005。",
            "2. 若没有候选过门，按最弱方向解释形成 n2 不可辨识边界。",
            "3. 后续发布只允许称为子空间局部灵敏度，不得称全局三维灵敏度。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    recovered = p2c69._recover_near_gate(params, current_i, target_p, y_seed)
    y_final = np.asarray(recovered["solve"]["y"], dtype=float)
    _args, funcs = p2c69._make_funcs(params, target_p, recovered["y_ref"], p2c69.STAGES[-1][0])
    I_j = jnp.asarray(current_i, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y_final, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    rows, weak_rows, svals = _audit_subspaces(J, r, len(params.flex))
    _plot(rows, weak_rows, svals)
    base_row = max(recovered["rows"], key=lambda row: row["stage"])
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "subspace_rows.csv", rows)
    _write_csv(OUT_DIR / "weak_direction_rows.csv", weak_rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failure_rows = [row for row in rows if not row["candidate_gate"]]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps({"failure": "effective_subspace_gate_failed", "evidence": row}, ensure_ascii=False) + "\n" for row in failure_rows),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "base_row": base_row,
        "subspace_rows": rows,
        "weak_direction_rows": weak_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(base_row, rows, weak_rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "subspace_rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
