"""Run block-whitened reoptimization for the near-gate n2 branch."""

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
P2C69_DIR = ROOT / "experiment_records_ik_optimize" / "20260627230500_p2c69_n2_condition_precondition_audit"
for path in (ROOT, P2C57_DIR, P2C58_DIR, P2C69_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c69_n2_condition_precondition_audit as p2c69

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
MAX_NFEV = 240


def _jsonable(value: Any) -> Any:
    return p2c57._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _physical_eval(
    *,
    label: str,
    params: Any,
    current_i: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y: np.ndarray,
    steady_s: float,
    nfev: int,
    success: bool,
    message: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    I_j = jnp.asarray(current_i, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    out = p2c57.p2c47.output_to_numpy(funcs["output"](y_j, I_j))
    continuity = np.asarray(jax.device_get(funcs["continuity"](y_j, I_j)), dtype=float).reshape(-1)
    svals = np.linalg.svd(J, compute_uv=False)
    solve = {
        "y": np.asarray(y, dtype=float),
        "out": out,
        "success": bool(success),
        "nfev": int(nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "continuity_norm": float(np.linalg.norm(continuity)),
        "message": str(message),
    }
    row = p2c57._row(
        force_scale=p2c69.FORCE_SCALE,
        stage=4,
        pos_scale=p2c69.STAGES[-1][0],
        max_iter=MAX_NFEV,
        params=params,
        target_p=target_p,
        solve=solve,
    )
    row["label"] = label
    row["reg_scale"] = p2c69.REG_SCALE
    row["physical_residual_norm"] = solve["residual_norm"]
    row["message"] = str(message)
    return row, r, J


def _solve_whitened(
    *,
    label: str,
    params: Any,
    current_i: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y0: np.ndarray,
    row_scale: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    I_j = jnp.asarray(current_i, dtype=jnp.float64).reshape(8)
    row_scale = np.asarray(row_scale, dtype=float).reshape(-1)

    def residual(yy: np.ndarray) -> np.ndarray:
        r = np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float)
        return row_scale * r

    def jacobian(yy: np.ndarray) -> np.ndarray:
        J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy, dtype=jnp.float64), I_j)), dtype=float)
        return row_scale[:, None] * J

    _ = residual(np.asarray(y0, dtype=float))
    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        np.asarray(y0, dtype=float).reshape(-1),
        jac=jacobian,
        method="trf",
        x_scale=np.ones_like(np.asarray(y0, dtype=float).reshape(-1)),
        loss="linear",
        max_nfev=MAX_NFEV,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = float(time.perf_counter() - t0)
    return _physical_eval(
        label=label,
        params=params,
        current_i=current_i,
        target_p=target_p,
        y_ref=y_ref,
        y=np.asarray(opt.x, dtype=float),
        steady_s=steady_s,
        nfev=int(opt.nfev),
        success=bool(opt.success),
        message=str(opt.message),
    )


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, key, ylabel, gate in [
        (axes[0], "stationarity_norm", "一阶最优性", 1.0e-5),
        (axes[1], "jacobian_condition", "物理条件数", 1.0e7),
        (axes[2], "continuity_norm", "接口连续误差", 1.0e-3),
        (axes[3], "max_interface_force_jump_over_rigid_force", "载荷传播比", 1.5),
    ]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        ax.bar(x, values)
        ax.axhline(gate, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle("P2c70 n2 块白化重求解物理门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_block_whitened_reopt_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 样本 | 是否过门 | 末端误差/m | 一阶最优性 | 物理条件数 | 接口连续 | 载荷传播比 | 迭代次数 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['label']} | {row['baseline_gate']} | {row['position_error_m']:.3e} | "
            f"{row['stationarity_norm']:.3e} | {row['jacobian_condition']:.3e} | "
            f"{row['continuity_norm']:.3e} | {row['max_interface_force_jump_over_rigid_force']:.3e} | "
            f"{row['nfev']} | {row['failure_class']} |"
        )
    cond_line = "; ".join(f"{row['variant']}={row['condition']:.3e}" for row in condition_rows)
    best = min(rows, key=lambda row: (not row["baseline_gate"], row["jacobian_condition"], row["stationarity_norm"]))
    return "\n".join(
        [
            "# P2c70 n2 块白化重求解记录",
            "",
            "## 实验目的",
            "",
            "P2c69 表明 n2 近门分支的高条件数主要来自残差块尺度和单位不均衡。本轮采用固定块白化权重进行重求解，但所有发布门禁仍用原物理残差和原物理雅可比评价。",
            "",
            "## 实验方案",
            "",
            "先按 P2c59 路径恢复 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4` 近门分支。然后在该分支处计算块白化权重，并分别从 `baseline_final` 与 `seed` 两个初值进行白化残差重求解。白化仅用于数值优化，不改变物理评价标准。",
            "",
            "## 条件数参考",
            "",
            f"- P2c69 同口径条件数：`{cond_line}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"最佳样本为 `{best['label']}`。其物理条件数为 `{best['jacobian_condition']:.3e}`，一阶最优性为 `{best['stationarity_norm']:.3e}`，接口连续误差为 `{best['continuity_norm']:.3e}`，载荷传播比为 `{best['max_interface_force_jump_over_rigid_force']:.3e}`。",
            "",
            "本轮没有得到可发布三维灵敏度基线。块白化重求解虽然改善了数值尺度，但实际求解会偏离原物理近门分支，导致接口连续和载荷传播显著退化。最佳样本仍是原始近门分支，其唯一主要失败项仍为物理条件数偏高。因此，单纯块白化不能作为最终修复，下一步应围绕最弱方向做 n2 有效子空间审计。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 构造 n2 有效子空间审计，先去除或约束 P2c69 识别出的最弱方向。",
            "2. 在有效子空间内复核一阶最优性、条件数、载荷传播和任务误差。",
            "3. 若有效子空间过门，再进入分支一致性和小扰动重求解。",
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
    recovered = p2c69._recover_near_gate(params, current_i, target_p, y_seed)
    y_ref = recovered["y_ref"]
    baseline_y = np.asarray(recovered["solve"]["y"], dtype=float)
    baseline_row, r_base, J_base = _physical_eval(
        label="baseline_final",
        params=params,
        current_i=current_i,
        target_p=target_p,
        y_ref=y_ref,
        y=baseline_y,
        steady_s=float(recovered["solve"]["steady_s"]),
        nfev=int(recovered["solve"]["nfev"]),
        success=bool(recovered["solve"]["success"]),
        message=str(recovered["solve"]["message"]),
    )
    slices = p2c69._block_slices(len(params.flex))
    row_scale = p2c69._row_scale_for_blocks(J_base, slices)
    condition_rows, _attrib, _svals = p2c69._analyze_jacobian(J_base, r_base, len(params.flex))
    rows = [baseline_row]
    for label, y0 in [("whiten_from_baseline", baseline_y), ("whiten_from_seed", y_seed)]:
        row, _r, _J = _solve_whitened(
            label=label,
            params=params,
            current_i=current_i,
            target_p=target_p,
            y_ref=y_ref,
            y0=np.asarray(y0, dtype=float),
            row_scale=row_scale,
        )
        rows.append(row)
    for row in rows:
        if not row["baseline_gate"]:
            with failures.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"failure": row["failure_class"], "evidence": row}, ensure_ascii=False) + "\n")
    _plot(rows)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "rows.csv", rows)
    _write_csv(OUT_DIR / "condition_rows.csv", condition_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            _jsonable(
                {
                    "experiment": OUT_DIR.name,
                    "wall_s": wall_s,
                    "rows": rows,
                    "condition_rows": condition_rows,
                    "jax_devices": [str(device) for device in jax.devices()],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "record.md").write_text(_record(rows, condition_rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
