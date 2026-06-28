"""Run a readonly rigid-load propagation observer for the n1 branch."""

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


def observe_interfaces(params: Any, y: np.ndarray, *, steps_per_flex: int) -> list[dict[str, Any]]:
    """Observe rigid-load propagation terms without changing the residual."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = p2c110.p2c57.p2c47.initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    rows: list[dict[str, Any]] = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = p2c110.p2c57.p2c47.normalize_state(x)
        x, _flex_points = p2c110.p2c57.p2c47._integrate_flex(params, x, i, steps_per_flex)
        flex_force = np.asarray(jax.device_get(x[7:10]), dtype=float)
        flex_torque = np.asarray(jax.device_get(x[10:13]), dtype=float)
        f_total_j, tau_total_j = p2c110.p2c57.p2c47._rigid_wrench(params, x, i)
        f_total = np.asarray(jax.device_get(f_total_j), dtype=float)
        tau_total = np.asarray(jax.device_get(tau_total_j), dtype=float)
        x_after, mid_p, _mid_q = p2c110.p2c57.p2c47._advance_rigid(params, x, i)
        after_force = np.asarray(jax.device_get(x_after[7:10]), dtype=float)
        after_torque = np.asarray(jax.device_get(x_after[10:13]), dtype=float)
        next_force = np.asarray(jax.device_get(blocks[i + 1, 0:3] * f_ref), dtype=float)
        next_torque = np.asarray(jax.device_get(blocks[i + 1, 3:6] * m_ref), dtype=float)
        force_jump = next_force - after_force
        torque_jump = next_torque - after_torque
        prox_p = np.asarray(jax.device_get(x[0:3]), dtype=float)
        mid_p_np = np.asarray(jax.device_get(mid_p), dtype=float)
        lever = mid_p_np - prox_p
        lever_torque = np.cross(lever, f_total)
        rows.append(
            {
                "interface": i,
                "flex_force_norm": float(np.linalg.norm(flex_force)),
                "flex_torque_norm": float(np.linalg.norm(flex_torque)),
                "rigid_force_norm": float(np.linalg.norm(f_total)),
                "rigid_torque_norm": float(np.linalg.norm(tau_total)),
                "lever_torque_norm": float(np.linalg.norm(lever_torque)),
                "after_force_norm": float(np.linalg.norm(after_force)),
                "after_torque_norm": float(np.linalg.norm(after_torque)),
                "next_force_norm": float(np.linalg.norm(next_force)),
                "next_torque_norm": float(np.linalg.norm(next_torque)),
                "force_jump_norm": float(np.linalg.norm(force_jump)),
                "torque_jump_norm": float(np.linalg.norm(torque_jump)),
                "force_jump_over_rigid_force": float(np.linalg.norm(force_jump) / max(float(np.linalg.norm(f_total)), 1.0e-30)),
                "torque_jump_over_rigid_torque": float(np.linalg.norm(torque_jump) / max(float(np.linalg.norm(tau_total)), 1.0e-30)),
                "lever_torque_over_rigid_torque": float(np.linalg.norm(lever_torque) / max(float(np.linalg.norm(tau_total)), 1.0e-30)),
                "force_jump_x": float(force_jump[0]),
                "force_jump_y": float(force_jump[1]),
                "force_jump_z": float(force_jump[2]),
                "torque_jump_x": float(torque_jump[0]),
                "torque_jump_y": float(torque_jump[1]),
                "torque_jump_z": float(torque_jump[2]),
            }
        )
        x = x_after
    return rows


def rerun_representative_branch() -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Rerun the representative force-scale-4 n1 branch to recover y."""
    params, current_i = p2c110.p2c57.p2c24._params_from_saved_case(p2c110.CASE_NAME, p2c110.ALPHA)
    target_p, y_slack = p2c110.p2c57.p2c24._load_saved_target_and_seed(p2c110.CASE_NAME, p2c110.ALPHA)
    y = p2c110.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c110.Args.steps_per_flex)
    y_ref = np.asarray(y, dtype=float).copy()
    rows: list[dict[str, Any]] = []
    final_solve: dict[str, Any] | None = None
    for stage, (position_scale, max_iter) in enumerate(p2c110.STAGES, start=1):
        args = p2c110.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = p2c110._make_functions(params, target_p, y_ref, args, force_scale=FORCE_SCALE)
        solve = p2c110._solve(y, current_i, funcs, args)
        load = p2c110.p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        rows.append(
            {
                "stage": stage,
                "position_scale": position_scale,
                "max_iter": max_iter,
                "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
                "stationarity_norm": solve["stationarity_norm"],
                "jacobian_condition": solve["jacobian_condition"],
                "continuity_norm": solve["continuity_norm"],
                "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
                "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
                **load,
                "nfev": solve["nfev"],
                "steady_s": solve["steady_s"],
                "success": solve["success"],
            }
        )
        y = solve["y"]
        final_solve = solve
    assert final_solve is not None
    return params, current_i, target_p, final_solve, rows


def plot_observer(stage_rows: list[dict[str, Any]], interface_rows: list[dict[str, Any]]) -> None:
    """Plot observer metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    stages = [row["stage"] for row in stage_rows]
    axes[0, 0].plot(stages, [row["stationarity_norm"] for row in stage_rows], marker="o")
    axes[0, 0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("一阶最优性")
    axes[0, 1].plot(stages, [row["jacobian_condition"] for row in stage_rows], marker="o")
    axes[0, 1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("条件数")
    if interface_rows:
        labels = [f"接口 {row['interface']}" for row in interface_rows]
        axes[1, 0].bar(labels, [row["force_jump_over_rigid_force"] for row in interface_rows])
        axes[1, 0].axhline(1.5, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("力跳变相对刚段外力")
        axes[1, 1].bar(labels, [row["torque_jump_over_rigid_torque"] for row in interface_rows], label="矩跳变")
        axes[1, 1].bar(labels, [row["lever_torque_over_rigid_torque"] for row in interface_rows], alpha=0.55, label="力臂矩")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("矩项对比")
        axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("n1 只读刚段载荷传播观测")
    fig.savefig(OUT_DIR / "n1_readonly_rigid_load_observer_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], stage_rows: list[dict[str, Any]], interface_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    final = stage_rows[-1]
    interface_lines = "\n".join(
        f"- 接口 {row['interface']}：力跳变/刚段外力 {row['force_jump_over_rigid_force']:.3e}，矩跳变/刚段外矩 {row['torque_jump_over_rigid_torque']:.3e}，力臂矩/刚段外矩 {row['lever_torque_over_rigid_torque']:.3e}。"
        for row in interface_rows
    ) or "- 当前参数无中间接口。"
    record = f"""# P2c128 n1 只读刚性磁体段载荷传播观测器

## 实验目的

本轮实现 P2c127 规划的只读观测器。该观测器不修改残差、不发布灵敏度、不接入外层 IK，只用于量化当前 n1 代表分支中的刚段外载、接口力跳变、接口矩跳变和力臂矩贡献。

## 实验方案

固定 `n1_tip_z_rand15, alpha=0.10`，只重跑 P2c110 中最接近物理闭合但仍不可发布的 `force_scale=4` 分支。得到最终状态后，沿 shooting 积分路径逐接口输出：

- 柔性段远端内力与内矩。
- 刚性磁体段合外力 \(F_m\) 与合外矩 \(\tau_m\)。
- 刚段传播后的接口内力链。
- 下一柔性段近端内力链。
- 力跳变、矩跳变和力臂矩项。

## 基线复现

- 末端位置误差：{final['position_error_m']:.3e} m。
- 一阶最优性：{final['stationarity_norm']:.3e}。
- 条件数：{final['jacobian_condition']:.3e}。
- 接口连续残差：{final['continuity_norm']:.3e}。
- 最大力跳变比例：{final['max_interface_force_jump_over_rigid_force']:.3e}。
- 最大矩跳变比例：{final['max_interface_torque_jump_over_rigid_torque']:.3e}。

## 观测结果

{interface_lines}

## 结论

观测器复现了 P2c110/P2c126 的核心现象：`force_scale=4` 能把力跳变比例压到很小，但基线仍因接口连续残差和一阶最优性组合约束不能发布。下一步不能继续调权重，而应把观测器中的力跳变、矩跳变和力臂矩写成显式残差原型，并重新跑基线门禁。

## 下一步计划

执行 P2c129：在实验级分支中加入显式 \(r_F\)、\(r_M\) 和刚段传播几何残差。只跑基线门禁；若一阶最优性、条件数和载荷传播仍不能同时通过，不进入扰动灵敏度扫描。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the readonly observer."""
    t0 = time.perf_counter()
    params, _current_i, _target_p, final_solve, stage_rows = rerun_representative_branch()
    interface_rows = observe_interfaces(params, final_solve["y"], steps_per_flex=p2c110.Args.steps_per_flex)
    final = stage_rows[-1]
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "readonly_observer_ran_n1_still_not_publishable",
        "case": p2c110.CASE_NAME,
        "alpha": p2c110.ALPHA,
        "force_scale": FORCE_SCALE,
        "publishable_claim": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "final_gate": {
            "stationarity_norm": final["stationarity_norm"],
            "jacobian_condition": final["jacobian_condition"],
            "continuity_norm": final["continuity_norm"],
            "force_ratio": final["max_interface_force_jump_over_rigid_force"],
            "torque_ratio": final["max_interface_torque_jump_over_rigid_torque"],
        },
        "interface_count": len(interface_rows),
        "max_observed_force_jump_over_rigid_force": max((row["force_jump_over_rigid_force"] for row in interface_rows), default=0.0),
        "max_observed_torque_jump_over_rigid_torque": max((row["torque_jump_over_rigid_torque"] for row in interface_rows), default=0.0),
        "max_lever_torque_over_rigid_torque": max((row["lever_torque_over_rigid_torque"] for row in interface_rows), default=0.0),
        "total_solve_steady_s": sum(row["steady_s"] for row in stage_rows),
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "stage_rows.csv", stage_rows)
    write_csv(OUT_DIR / "interface_observer_rows.csv", interface_rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "wall_s": summary["wall_s"], "total_solve_steady_s": summary["total_solve_steady_s"]}])
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text("", encoding="utf-8")
    plot_observer(stage_rows, interface_rows)
    write_record(summary, stage_rows, interface_rows)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
