"""Audit conditioning of the near-gate n2 branch without adding slack."""

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
P2C58_DIR = ROOT / "experiment_records_ik_optimize" / "20260607202500_p2c58_force_continuity_condition_probe"
for path in (ROOT, P2C57_DIR, P2C58_DIR):
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
import run_p2c58_force_continuity_condition_probe as p2c58

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
FORCE_SCALE = 4.0
REG_SCALE = 3.0e-4
TORQUE_SCALE = p2c57.TORQUE_SCALE
STAGES = p2c57.STAGES


def _jsonable(value: Any) -> Any:
    return p2c57._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _condition_number(mat: np.ndarray) -> float:
    svals = np.linalg.svd(np.asarray(mat, dtype=float), compute_uv=False)
    if not svals.size or svals[-1] <= np.finfo(float).tiny:
        return float("inf")
    return float(svals[0] / svals[-1])


def _make_funcs(params: Any, target_p: np.ndarray, y_ref: np.ndarray, position_scale: float) -> tuple[Any, Any]:
    args = p2c58.Args()
    args.position_scale = float(position_scale)
    args.reg_scale = float(REG_SCALE)
    funcs = p2c57._make_split_functions(
        params,
        target_p,
        y_ref,
        args,
        force_scale=FORCE_SCALE,
        torque_scale=TORQUE_SCALE,
    )
    return args, funcs


def _recover_near_gate(params: Any, current_i: np.ndarray, target_p: np.ndarray, y_seed: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_seed, dtype=float).copy()
    y_ref = y.copy()
    rows = []
    final_solve = None
    for stage, (position_scale, max_iter) in enumerate(STAGES, start=1):
        args, funcs = _make_funcs(params, target_p, y_ref, position_scale)
        args.max_iter = int(max_iter)
        solve = p2c57.p2c47._solve(y, current_i, funcs, args)
        row = p2c57._row(
            force_scale=FORCE_SCALE,
            stage=stage,
            pos_scale=position_scale,
            max_iter=max_iter,
            params=params,
            target_p=target_p,
            solve=solve,
        )
        row["reg_scale"] = REG_SCALE
        rows.append(row)
        y = solve["y"]
        final_solve = solve
    assert final_solve is not None
    return {"rows": rows, "solve": final_solve, "y_ref": y_ref}


def _block_slices(n_flex: int) -> dict[str, slice]:
    pos = slice(0, 3)
    boundary = slice(3, 9)
    continuity_len = 6 * max(n_flex - 1, 0)
    continuity = slice(9, 9 + continuity_len)
    reg = slice(9 + continuity_len, 9 + continuity_len + 6 * n_flex)
    return {"terminal_position": pos, "terminal_boundary": boundary, "interface_continuity": continuity, "regularization": reg}


def _row_scale_for_blocks(J: np.ndarray, slices: dict[str, slice]) -> np.ndarray:
    scale = np.ones((J.shape[0],), dtype=float)
    for slc in slices.values():
        block = J[slc, :]
        if block.size == 0:
            continue
        denom = max(float(np.linalg.norm(block, ord="fro")) / max(np.sqrt(block.shape[0]), 1.0), 1.0e-12)
        scale[slc] = 1.0 / denom
    return scale


def _analyze_jacobian(J: np.ndarray, r: np.ndarray, n_flex: int) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    slices = _block_slices(n_flex)
    col_norms = np.linalg.norm(J, axis=0)
    col_scale = 1.0 / np.maximum(col_norms, 1.0e-12)
    row_scale = _row_scale_for_blocks(J, slices)
    variants = {
        "physical": J,
        "column_scaled": J @ np.diag(col_scale),
        "block_whitened": row_scale[:, None] * J,
        "block_and_column_scaled": (row_scale[:, None] * J) @ np.diag(col_scale),
    }
    rows = []
    for name, mat in variants.items():
        svals = np.linalg.svd(mat, compute_uv=False)
        rows.append(
            {
                "variant": name,
                "condition": _condition_number(mat),
                "largest_singular": float(svals[0]) if svals.size else 0.0,
                "smallest_singular": float(svals[-1]) if svals.size else 0.0,
                "rank": int(np.linalg.matrix_rank(mat)),
            }
        )
    U, svals, Vt = np.linalg.svd(J, full_matrices=False)
    weak = Vt[-1]
    weak_blocks = weak.reshape((n_flex, 6))
    block_energy = []
    for i, block in enumerate(weak_blocks):
        force_e = float(np.linalg.norm(block[:3]) ** 2)
        torque_e = float(np.linalg.norm(block[3:]) ** 2)
        block_energy.append({"segment": i, "force_energy": force_e, "torque_energy": torque_e, "total_energy": force_e + torque_e})
    residual_blocks = []
    for name, slc in slices.items():
        Jb = J[slc, :]
        rb = r[slc]
        residual_blocks.append(
            {
                "block": name,
                "residual_norm": float(np.linalg.norm(rb)),
                "jacobian_fro_norm": float(np.linalg.norm(Jb, ord="fro")),
                "weak_projection_norm": float(np.linalg.norm(Jb @ weak)),
                "stationarity_contribution_norm": float(np.linalg.norm(Jb.T @ rb)),
            }
        )
    attribution = {
        "physical_condition": rows[0]["condition"],
        "weakest_singular": float(svals[-1]),
        "largest_singular": float(svals[0]),
        "weak_segment_energy": block_energy,
        "residual_blocks": residual_blocks,
        "weak_vector": weak.tolist(),
    }
    return rows, attribution, svals


def _plot(condition_rows: list[dict[str, Any]], svals: np.ndarray, attribution: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = [row["variant"] for row in condition_rows]
    axes[0].bar(labels, [row["condition"] for row in condition_rows], color="#4f81bd")
    axes[0].axhline(1.0e7, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("条件数")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(np.arange(1, len(svals) + 1), svals, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("奇异值序号")
    axes[1].set_ylabel("奇异值")
    axes[1].grid(alpha=0.25)
    energies = attribution["weak_segment_energy"]
    x = np.arange(len(energies))
    axes[2].bar(x - 0.15, [row["force_energy"] for row in energies], width=0.3, label="内力")
    axes[2].bar(x + 0.15, [row["torque_energy"] for row in energies], width=0.3, label="内矩")
    axes[2].set_xlabel("分段")
    axes[2].set_ylabel("最弱方向能量")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c69 n2 近门条件数归因")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_condition_precondition_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(stage_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]], attribution: dict[str, Any], wall_s: float) -> str:
    final = max(stage_rows, key=lambda row: row["stage"])
    best_pre = min(condition_rows, key=lambda row: row["condition"])
    weak_block = max(attribution["weak_segment_energy"], key=lambda row: row["total_energy"])
    block_lines = [
        "| 残差块 | 残差范数 | 雅可比范数 | 最弱方向投影 | 一阶贡献 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in attribution["residual_blocks"]:
        block_lines.append(
            f"| {row['block']} | {row['residual_norm']:.3e} | {row['jacobian_fro_norm']:.3e} | "
            f"{row['weak_projection_norm']:.3e} | {row['stationarity_contribution_norm']:.3e} |"
        )
    cond_lines = [
        "| 变换 | 条件数 | 最大奇异值 | 最小奇异值 | 秩 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in condition_rows:
        cond_lines.append(
            f"| {row['variant']} | {row['condition']:.3e} | {row['largest_singular']:.3e} | "
            f"{row['smallest_singular']:.3e} | {row['rank']} |"
        )
    return "\n".join(
        [
            "# P2c69 n2 近门条件数与预条件归因记录",
            "",
            "## 实验目的",
            "",
            "P2c68 将两点 n2 主战场收敛到 P2c59 近门分支：末端误差、一阶最优性、接口连续和载荷传播均可接受，主要阻断项是物理坐标条件数。本轮不引入自由松弛变量，只重算该近门分支并审计条件数来源。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4`。按 P2c59 同一路径恢复最终状态，随后计算最终残差雅可比矩阵。比较四类矩阵：原物理矩阵、列范数缩放、残差块白化、块白化叠加列缩放。所有发布判定仍以原物理矩阵和原物理残差为准。",
            "",
            "## 基线复现",
            "",
            f"- 末端位置误差：`{final['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{final['stationarity_norm']:.3e}`",
            f"- 物理坐标条件数：`{final['jacobian_condition']:.3e}`",
            f"- 接口连续误差：`{final['continuity_norm']:.3e}`",
            f"- 相对刚段外力跳变：`{final['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 条件数变换结果",
            "",
            *cond_lines,
            "",
            "## 残差块归因",
            "",
            *block_lines,
            "",
            "## 结论",
            "",
            f"物理坐标条件数仍为 `{condition_rows[0]['condition']:.3e}`，不满足发布门禁。最优数值变换为 `{best_pre['variant']}`，条件数降至 `{best_pre['condition']:.3e}`，说明当前病态主要来自残差块尺度和单位不均衡，而不是雅可比矩阵秩亏。该结果不能直接视为可发布灵敏度，只能作为预条件重求解和分支复核的入口。",
            "",
            f"最弱方向能量主要集中在第 `{weak_block['segment']}` 段，内力能量 `{weak_block['force_energy']:.3e}`，内矩能量 `{weak_block['torque_energy']:.3e}`。该方向将作为下一轮有效子空间判断依据：若预条件重求解后物理条件数仍不达标，则应判断该方向是否为任务不可辨识方向，并转向有效子空间灵敏度，而不是完整三维发布。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. P2c70 采用块白化残差做预条件重求解，但发布门禁仍回到原物理残差评价。",
            "2. 若预条件重求解保持物理门禁并改善分支稳定性，再做多初值分支一致性。",
            "3. 若物理条件数仍不达标，提取最弱方向并转入 n2 有效子空间审计。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    recovered = _recover_near_gate(params, current_i, target_p, y_seed)
    final = recovered["solve"]
    y_final = np.asarray(final["y"], dtype=float)
    np.save(OUT_DIR / "y_final.npy", y_final)
    _args, funcs = _make_funcs(params, target_p, recovered["y_ref"], STAGES[-1][0])
    I_j = jnp.asarray(current_i, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y_final, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    condition_rows, attribution, svals = _analyze_jacobian(J, r, len(params.flex))
    _plot(condition_rows, svals, attribution)
    final_row = max(recovered["rows"], key=lambda row: row["stage"])
    if final_row["jacobian_condition"] >= 1.0e7:
        failures.write_text(
            json.dumps({"failure": "条件数偏高", "evidence": final_row}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "stage_rows.csv", recovered["rows"])
    _write_csv(OUT_DIR / "condition_rows.csv", condition_rows)
    _write_csv(OUT_DIR / "residual_block_rows.csv", attribution["residual_blocks"])
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "force_scale": FORCE_SCALE,
        "reg_scale": REG_SCALE,
        "stage_rows": recovered["rows"],
        "condition_rows": condition_rows,
        "attribution": attribution,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "summary.csv", [max(recovered["rows"], key=lambda row: row["stage"])])
    (OUT_DIR / "record.md").write_text(_record(recovered["rows"], condition_rows, attribution, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "condition_rows": condition_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
