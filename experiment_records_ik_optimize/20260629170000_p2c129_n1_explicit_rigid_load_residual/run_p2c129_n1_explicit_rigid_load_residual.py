"""Probe an explicit rigid-load residual for the n1 branch."""

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
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
P2C110_DIR = EXPERIMENT_ROOT / "20260629030000_p2c110_n1_hard_force_scaled_continuity"
for path in (ROOT, EXPERIMENT_ROOT, P2C110_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c110_n1_hard_force_scaled_continuity as p2c110


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

FORCE_SCALE = 4.0
TORQUE_SCALES = (1.0, 4.0)


def jsonable(value: Any) -> Any:
    """Convert arrays and scalars into JSON-compatible values."""
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
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def explicit_rigid_load_residual(params: Any, y: jnp.ndarray, *, steps_per_flex: int, load_floor: float) -> jnp.ndarray:
    """Return force and torque jumps normalized by local rigid-load scales."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = p2c110.p2c57.p2c47.initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    f_floor = float(load_floor) * max(f_ref, 1.0e-12)
    m_floor = float(load_floor) * max(m_ref, 1.0e-12)
    pieces = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = p2c110.p2c57.p2c47.normalize_state(x)
        x, _flex_points = p2c110.p2c57.p2c47._integrate_flex(params, x, i, steps_per_flex)
        prox_p = x[0:3]
        f_total, tau_total = p2c110.p2c57.p2c47._rigid_wrench(params, x, i)
        x_after, mid_p, _mid_q = p2c110.p2c57.p2c47._advance_rigid(params, x, i)
        next_force = blocks[i + 1, 0:3] * f_ref
        next_torque = blocks[i + 1, 3:6] * m_ref
        jump_force = next_force - x_after[7:10]
        jump_torque = next_torque - x_after[10:13]
        lever_torque = jnp.cross(mid_p - prox_p, f_total)
        force_denom = jnp.maximum(jnp.linalg.norm(f_total), f_floor)
        torque_denom = jnp.maximum(jnp.linalg.norm(tau_total) + jnp.linalg.norm(lever_torque), m_floor)
        pieces.append(jnp.concatenate([jump_force / force_denom, jump_torque / torque_denom], axis=0))
        x = x_after
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(pieces, axis=0)


def make_functions(params_base: Any, target_p: np.ndarray, y_ref: np.ndarray, args: p2c110.Args, *, torque_scale: float) -> dict[str, Any]:
    """Build residual and Jacobian functions for the explicit-load prototype."""
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
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
        raw = explicit_rigid_load_residual(params, y, steps_per_flex=args.steps_per_flex, load_floor=p2c110.LOAD_FLOOR).reshape((-1, 6))
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_force = float(FORCE_SCALE) * raw[:, 0:3].reshape(-1)
        r_torque = float(torque_scale) * raw[:, 3:6].reshape(-1)
        r_reg = float(args.reg_scale) * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_force.reshape(-1), r_torque.reshape(-1), r_reg.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
        "continuity": jax.jit(continuity_yI),
    }


def run_variant(torque_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run staged continuation for one torque scale."""
    params, current_i = p2c110.p2c57.p2c24._params_from_saved_case(p2c110.CASE_NAME, p2c110.ALPHA)
    target_p, y_slack = p2c110.p2c57.p2c24._load_saved_target_and_seed(p2c110.CASE_NAME, p2c110.ALPHA)
    y = p2c110.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c110.Args.steps_per_flex)
    y_ref = np.asarray(y, dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for stage, (position_scale, max_iter) in enumerate(p2c110.STAGES, start=1):
        args = p2c110.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = make_functions(params, target_p, y_ref, args, torque_scale=torque_scale)
        solve = p2c110._solve(y, current_i, funcs, args)
        load = p2c110.p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        raw_explicit = np.asarray(
            jax.device_get(explicit_rigid_load_residual(params, jnp.asarray(solve["y"], dtype=jnp.float64), steps_per_flex=args.steps_per_flex, load_floor=p2c110.LOAD_FLOOR)),
            dtype=float,
        )
        row = {
            "stage": stage,
            "position_scale": position_scale,
            "max_iter": max_iter,
            "force_scale": FORCE_SCALE,
            "torque_scale": torque_scale,
            "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "continuity_norm": solve["continuity_norm"],
            "explicit_load_residual_norm": float(np.linalg.norm(raw_explicit)),
            "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
            "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
            **load,
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "success": solve["success"],
            "message": solve["message"],
        }
        row["baseline_gate"] = bool(
            row["success"]
            and row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] < 1.0e7
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
        )
        if row["baseline_gate"]:
            row["failure_class"] = "通过"
        elif row["stationarity_norm"] > 1.0e-5:
            row["failure_class"] = "一阶最优性退化"
        elif row["jacobian_condition"] >= 1.0e7:
            row["failure_class"] = "条件数病态"
        elif row["continuity_norm"] > 1.0e-3:
            row["failure_class"] = "接口连续残差超限"
        elif row["max_interface_force_jump_over_rigid_force"] > 1.5:
            row["failure_class"] = "载荷传播失配"
        else:
            row["failure_class"] = "综合失败"
        rows.append(row)
        y = solve["y"]
    return rows, rows[-1]


def plot_rows(final_rows: list[dict[str, Any]]) -> None:
    """Plot final gate metrics."""
    labels = [f"矩尺度 {row['torque_scale']:g}" for row in final_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].bar(labels, [row["stationarity_norm"] for row in final_rows])
    axes[0, 0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("一阶最优性")
    axes[0, 1].bar(labels, [row["jacobian_condition"] for row in final_rows])
    axes[0, 1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("条件数")
    axes[1, 0].bar(labels, [row["continuity_norm"] for row in final_rows])
    axes[1, 0].axhline(1.0e-3, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("原接口连续残差")
    axes[1, 1].bar(labels, [row["explicit_load_residual_norm"] for row in final_rows])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("显式载荷残差范数")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n1 显式刚段载荷残差最小原型")
    fig.savefig(OUT_DIR / "n1_explicit_rigid_load_residual_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    result_lines = "\n".join(
        f"- 矩尺度 {row['torque_scale']:.1f}：基线门禁 {row['baseline_gate']}；一阶最优性 {row['stationarity_norm']:.3e}；条件数 {row['jacobian_condition']:.3e}；接口连续残差 {row['continuity_norm']:.3e}；失败分类 {row['failure_class']}。"
        for row in final_rows
    )
    record = f"""# P2c129 n1 显式刚段载荷残差最小原型

## 实验目的

本轮根据 P2c128 的只读观测结果，在实验级 runner 中加入显式刚段载荷残差原型。该实验不修改生产求解器、不发布灵敏度、不进入扰动扫描，只检查基线物理门禁是否因更合理的力/矩归一化而改善。

## 实验方案

固定 `n1_tip_z_rand15, alpha=0.10`，保留 P2c110 的 staged continuation。新残差将接口力跳变按刚段合外力归一化，将接口矩跳变按“刚段合外矩 + 力臂矩”尺度归一化：

\\[
r_F = \\frac{{\\Delta F}}{{\\max(\\|F_m\\|, F_{{floor}})}},\\qquad
r_M = \\frac{{\\Delta M}}{{\\max(\\|\\tau_m\\| + \\|r\\times F_m\\|, M_{{floor}})}}.
\\]

测试矩残差权重为 1 和 4。验收仍使用原硬门禁：一阶最优性、条件数、接口连续残差、载荷传播比例同时通过。

## 结果

{result_lines}

## 结论

若显式载荷残差仍不能通过基线门禁，则说明单点磁控当前工作点的问题不只是残差尺度，而可能需要改变状态参数化、任务维度或工作点。若某一设置通过基线门禁，下一轮才允许进入分支一致性与扰动重求解审计。

## 下一步计划

若本轮无通过设置，应执行 P2c130：任务降维与保守工作点对照；若有通过设置，应执行可信分支复核，不直接接入外层 IK。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the explicit residual prototype."""
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for torque_scale in TORQUE_SCALES:
        rows, final = run_variant(torque_scale)
        all_rows.extend(rows)
        final_rows.append(final)
    pass_count = sum(row["baseline_gate"] for row in final_rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "explicit_residual_baseline_pass" if pass_count else "explicit_residual_baseline_failed",
        "case": p2c110.CASE_NAME,
        "alpha": p2c110.ALPHA,
        "force_scale": FORCE_SCALE,
        "torque_scales": list(TORQUE_SCALES),
        "pass_count": pass_count,
        "tested_final_variants": len(final_rows),
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
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
            json.dumps({"torque_scale": row["torque_scale"], "failure_class": row["failure_class"], "baseline_gate": row["baseline_gate"]}, ensure_ascii=False) + "\n"
            for row in final_rows
            if not row["baseline_gate"]
        ),
        encoding="utf-8",
    )
    plot_rows(final_rows)
    write_record(summary, final_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
