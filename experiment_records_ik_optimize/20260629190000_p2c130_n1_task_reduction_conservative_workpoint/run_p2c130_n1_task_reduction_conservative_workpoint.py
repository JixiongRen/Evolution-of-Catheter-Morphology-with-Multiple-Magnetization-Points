"""Compare n1 task reduction and conservative current levels."""

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
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
P2C110_DIR = EXPERIMENT_ROOT / "20260629030000_p2c110_n1_hard_force_scaled_continuity"
P2C129_DIR = EXPERIMENT_ROOT / "20260629170000_p2c129_n1_explicit_rigid_load_residual"
for path in (ROOT, EXPERIMENT_ROOT, P2C110_DIR, P2C129_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c110_n1_hard_force_scaled_continuity as p2c110
import run_p2c129_n1_explicit_rigid_load_residual as p2c129


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CURRENT_SCALES = (0.50, 1.00)
TASK_MODES = ("effective_1d", "position_3d")
FORCE_SCALE = 4.0
TORQUE_SCALE = 1.0
STAGES = ((10.0, 120), (30.0, 160), (60.0, 240))


def jsonable(value: Any) -> Any:
    """Convert values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write tabular rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_target(params: Any, current_i: np.ndarray, y0: np.ndarray, target_full: np.ndarray, current_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the interpolated target and effective direction."""
    zero_i = np.zeros_like(current_i)
    base_out = p2c110.p2c57.p2c47.output_to_numpy(
        p2c110.p2c57.p2c47.forward_segment_wrench(
            p2c110.p2c57.p2c47.with_coil_currents(params, jnp.asarray(zero_i, dtype=jnp.float64)),
            jnp.asarray(y0, dtype=jnp.float64),
            steps_per_flex=p2c110.Args.steps_per_flex,
        )
    )
    base_tip = np.asarray(base_out["tip_p"], dtype=float).reshape(3)
    displacement = np.asarray(target_full, dtype=float).reshape(3) - base_tip
    target = base_tip + float(current_scale) * displacement
    norm = float(np.linalg.norm(displacement))
    direction = displacement / max(norm, 1.0e-12)
    return target, direction, base_tip


def make_functions(params_base: Any, target_p: np.ndarray, direction: np.ndarray, y_ref: np.ndarray, args: p2c110.Args, *, task_mode: str) -> dict[str, Any]:
    """Build residual and derivative functions for one reduced task."""
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    direction_j = jnp.asarray(direction, dtype=jnp.float64).reshape(3)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    mask = p2c110.p2c57.p2c47._mode_mask("full")

    def params_for_I(I: jnp.ndarray) -> Any:
        return p2c110.p2c57.p2c47.with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c110.p2c57.p2c47.forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return p2c110.p2c57.p2c47.p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = p2c110.p2c57.p2c47.forward_segment_wrench(params, y, steps_per_flex=args.steps_per_flex)
        raw = p2c129.explicit_rigid_load_residual(params, y, steps_per_flex=args.steps_per_flex, load_floor=p2c110.LOAD_FLOOR).reshape((-1, 6))
        diff = out.tip_p - target_p_j
        if task_mode == "effective_1d":
            r_p = float(args.position_scale) * jnp.dot(diff, direction_j).reshape(1)
        else:
            r_p = float(args.position_scale) * diff.reshape(3)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_force = float(FORCE_SCALE) * raw[:, 0:3].reshape(-1)
        r_torque = float(TORQUE_SCALE) * raw[:, 3:6].reshape(-1)
        r_reg = float(args.reg_scale) * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_force.reshape(-1), r_torque.reshape(-1), r_reg.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
        "continuity": jax.jit(continuity_yI),
    }


def solve_stage(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], args: p2c110.Args) -> dict[str, Any]:
    """Solve one least-squares stage."""
    return p2c110._solve(y0, I, funcs, args)


def classify(row: dict[str, Any]) -> str:
    """Return the dominant failure class."""
    if row["baseline_gate"]:
        return "通过"
    if row["stationarity_norm"] > 1.0e-5:
        return "一阶最优性退化"
    if row["jacobian_condition"] >= 1.0e7:
        return "条件数病态"
    if row["continuity_norm"] > 1.0e-3:
        return "接口连续残差超限"
    if row["max_interface_force_jump_over_rigid_force"] > 1.5:
        return "载荷传播失配"
    return "综合失败"


def run_variant(task_mode: str, current_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one task and current-level variant."""
    params, current_i_full = p2c110.p2c57.p2c24._params_from_saved_case(p2c110.CASE_NAME, p2c110.ALPHA)
    target_full, y_slack = p2c110.p2c57.p2c24._load_saved_target_and_seed(p2c110.CASE_NAME, p2c110.ALPHA)
    y = p2c110.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c110.Args.steps_per_flex)
    y_ref = np.asarray(y, dtype=float).copy()
    target_p, direction, base_tip = build_target(params, current_i_full, y, target_full, current_scale)
    current_i = np.asarray(current_i_full, dtype=float) * float(current_scale)
    rows: list[dict[str, Any]] = []
    for stage, (position_scale, max_iter) in enumerate(STAGES, start=1):
        args = p2c110.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = make_functions(params, target_p, direction, y_ref, args, task_mode=task_mode)
        solve = solve_stage(y, current_i, funcs, args)
        load = p2c110.p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        tip_p = np.asarray(solve["out"]["tip_p"], dtype=float).reshape(3)
        position_error_vec = tip_p - target_p
        row = {
            "case": p2c110.CASE_NAME,
            "alpha_reference": p2c110.ALPHA,
            "current_scale": current_scale,
            "task_mode": task_mode,
            "stage": stage,
            "position_scale": position_scale,
            "max_iter": max_iter,
            "force_scale": FORCE_SCALE,
            "torque_scale": TORQUE_SCALE,
            "target_p_x": target_p[0],
            "target_p_y": target_p[1],
            "target_p_z": target_p[2],
            "base_tip_x": base_tip[0],
            "base_tip_y": base_tip[1],
            "base_tip_z": base_tip[2],
            "tip_p_x": tip_p[0],
            "tip_p_y": tip_p[1],
            "tip_p_z": tip_p[2],
            "position_error_m": float(np.linalg.norm(position_error_vec)),
            "projected_position_error_m": float(abs(np.dot(position_error_vec, direction))),
            "lateral_position_error_m": float(np.linalg.norm(position_error_vec - np.dot(position_error_vec, direction) * direction)),
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "continuity_norm": solve["continuity_norm"],
            "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
            "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
            **load,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "success": solve["success"],
            "message": solve["message"],
        }
        row["task_error_gate"] = bool(row["projected_position_error_m"] <= 5.0e-5 if task_mode == "effective_1d" else row["position_error_m"] <= 5.0e-5)
        row["baseline_gate"] = bool(
            row["success"]
            and row["task_error_gate"]
            and row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] < 1.0e7
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
        )
        row["failure_class"] = classify(row)
        rows.append(row)
        y = solve["y"]
    return rows, rows[-1]


def plot_summary(final_rows: list[dict[str, Any]]) -> None:
    """Plot final gate metrics."""
    labels = [f"{row['task_mode']}\n电流{row['current_scale']:.2f}" for row in final_rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(labels, [row["stationarity_norm"] for row in final_rows], color="#457b9d")
    axes[0, 0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("一阶最优性")
    axes[0, 1].bar(labels, [row["jacobian_condition"] for row in final_rows], color="#2a9d8f")
    axes[0, 1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("条件数")
    axes[1, 0].bar(labels, [row["max_interface_force_jump_over_rigid_force"] for row in final_rows], color="#e76f51")
    axes[1, 0].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("载荷传播比例")
    axes[1, 1].bar(labels, [row["position_error_m"] for row in final_rows], color="#f4a261")
    axes[1, 1].axhline(5.0e-5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("末端位置误差")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c130 单点磁控任务降维与保守工作点对照")
    fig.savefig(OUT_DIR / "n1_task_reduction_conservative_workpoint_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    """Write the experiment record in Chinese."""
    lines = "\n".join(
        f"- {row['task_mode']}，电流比例 {row['current_scale']:.2f}：门禁 {row['baseline_gate']}；"
        f"一阶最优性 {row['stationarity_norm']:.3e}；条件数 {row['jacobian_condition']:.3e}；"
        f"载荷传播比例 {row['max_interface_force_jump_over_rigid_force']:.3e}；"
        f"投影误差 {row['projected_position_error_m']:.3e} m；失败分类 {row['failure_class']}。"
        for row in final_rows
    )
    record = f"""# P2c130 单点磁控任务降维与保守工作点对照

## 实验目的

P2c129 表明显式刚段载荷残差不能直接使 `n1_tip_z_rand15, alpha=0.10` 通过基线门禁。本轮进一步判断失败是否主要来自任务维度过高或电流工作点过激。实验仍只做基线门禁，不进入扰动重求解灵敏度扫描。

## 实验方案

以 P2c129 的显式力/矩载荷残差为接口模型，固定力权重 4、矩权重 1。比较两种任务口径：

- `position_3d`：三维末端位置同时约束。
- `effective_1d`：只约束从零电流末端到参考目标末端的位移方向，即有效一维子空间。

比较两档电流比例：0.50 和 1.00。保守目标由零电流末端到原 `alpha=0.10` 目标之间线性插值得到，因此它是工作点强度对照，不作为新的可发布灵敏度证据。

## 结果

{lines}

## 结论

本轮只用于判断基线门禁。若有效一维任务在保守电流下通过，下一轮应围绕该工作点做多初值分支一致性和小扰动重求解；若仍不通过，则说明单点磁控当前模型的主要障碍不是末端任务维度，而是物理闭合或数值病态。

## 下一步计划

若存在通过样本，执行分支库复核；若不存在通过样本，执行 P2c131：改变单点磁体位置或选择更弱磁化/更低电流的样本，确认单点磁控不可发布边界是否具有配置依赖性。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run all task-reduction variants."""
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for current_scale in CURRENT_SCALES:
        for task_mode in TASK_MODES:
            rows, final = run_variant(task_mode, current_scale)
            all_rows.extend(rows)
            final_rows.append(final)
    pass_count = sum(row["baseline_gate"] for row in final_rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "n1_reduced_task_baseline_pass" if pass_count else "n1_reduced_task_baseline_failed",
        "case": p2c110.CASE_NAME,
        "alpha_reference": p2c110.ALPHA,
        "current_scales": list(CURRENT_SCALES),
        "task_modes": list(TASK_MODES),
        "pass_count": pass_count,
        "tested_final_variants": len(final_rows),
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": bool(pass_count),
        "final_rows": final_rows,
        "total_solve_steady_s": sum(row["steady_s"] for row in all_rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "stage_rows.csv", all_rows)
    write_csv(OUT_DIR / "final_rows.csv", final_rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "pass_count": pass_count, "wall_s": summary["wall_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_mode": row["task_mode"],
                    "current_scale": row["current_scale"],
                    "failure_class": row["failure_class"],
                    "baseline_gate": row["baseline_gate"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in final_rows
            if not row["baseline_gate"]
        ),
        encoding="utf-8",
    )
    plot_summary(final_rows)
    write_record(summary, final_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
