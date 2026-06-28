"""Audit whether n1 baseline failures depend on magnet placement."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment_records_ik_optimize"
P2C4_DIR = EXPERIMENT_ROOT / "20260603095445_p2c4_clean_terminal_multi_case"
P2C2_DIR = EXPERIMENT_ROOT / "20260602103750_p2c2_terminal_pose_constraint"
P2C110_DIR = EXPERIMENT_ROOT / "20260629030000_p2c110_n1_hard_force_scaled_continuity"
P2C129_DIR = EXPERIMENT_ROOT / "20260629170000_p2c129_n1_explicit_rigid_load_residual"
for path in (ROOT, EXPERIMENT_ROOT, P2C4_DIR, P2C2_DIR, P2C110_DIR, P2C129_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_clean_multi_case as p2c4
import run_p2c2_terminal_pose as p2c2
import run_p2c110_n1_hard_force_scaled_continuity as p2c110
import run_p2c129_n1_explicit_rigid_load_residual as p2c129


jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
M_MAG = 0.005301
BASE_CURRENTS = np.asarray((3.0, -12.0, 6.0, 0.0, 14.0, -9.0, 4.5, -2.0), dtype=float)
CURRENT_ALPHA = 0.10
TASK_MODES = ("effective_1d", "position_3d")
FORCE_SCALE = 4.0
TORQUE_SCALE = 1.0
STAGES = ((10.0, 120), (30.0, 160), (60.0, 240))


@dataclass(frozen=True)
class N1Spec:
    """Single-magnet configuration."""

    name: str
    m_body_list: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def cases() -> list[N1Spec]:
    """Return the audited single-magnet configurations."""
    z = (0.0, 0.0, M_MAG)
    zero = (0.0, 0.0, 0.0)
    return [
        N1Spec("n1_base_z_rand15_a010", (z, zero, zero)),
        N1Spec("n1_mid_z_rand15_a010", (zero, z, zero)),
        N1Spec("n1_tip_z_rand15_a010", (zero, zero, z)),
    ]


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
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_case(spec: N1Spec) -> p2c4.CaseSpec:
    """Build a clean-multi-case compatible case."""
    currents = tuple((BASE_CURRENTS * CURRENT_ALPHA).tolist())
    return p2c4.CaseSpec(spec.name, spec.m_body_list, currents)


def solve_oracles(case: p2c4.CaseSpec) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Solve zero-current and target-current FK oracle states."""
    engine = p2c4._build_engine(case, max_iter=120, tol=1.0e-5)
    target_oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.asarray(case.currents, dtype=float), z0=None)
    zero_oracle = p2c2.p2c1.p2c._solve_oracle(engine, np.zeros(8, dtype=float), z0=None)
    return zero_oracle, target_oracle, engine


def make_functions(params_base: Any, target_p: np.ndarray, direction: np.ndarray, y_ref: np.ndarray, args: p2c110.Args, *, task_mode: str) -> dict[str, Any]:
    """Build residual and derivative functions."""
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


def classify(row: dict[str, Any]) -> str:
    """Classify the dominant failed gate."""
    if row["baseline_gate"]:
        return "通过"
    if not row["task_error_gate"]:
        return "任务误差超限"
    if row["stationarity_norm"] > 1.0e-5:
        return "一阶最优性退化"
    if row["jacobian_condition"] >= 1.0e7:
        return "条件数病态"
    if row["continuity_norm"] > 1.0e-3:
        return "接口连续残差超限"
    if row["max_interface_force_jump_over_rigid_force"] > 1.5:
        return "载荷传播失配"
    return "综合失败"


def run_variant(spec: N1Spec, task_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one case and task-mode variant."""
    case = build_case(spec)
    zero_oracle, target_oracle, _engine = solve_oracles(case)
    params = target_oracle["params"]
    current_i = np.asarray(case.currents, dtype=float)
    target_p = np.asarray(jax.device_get(target_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    zero_p = np.asarray(jax.device_get(zero_oracle["sites"]["tip_p_dim"]), dtype=float).reshape(3)
    direction_raw = target_p - zero_p
    direction_norm = float(np.linalg.norm(direction_raw))
    direction = direction_raw / max(direction_norm, 1.0e-12)
    y = p2c2.p2c1._y_guess_from_oracle(target_oracle["z"], params, "segment_wrench")
    y_ref = np.asarray(y, dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for stage, (position_scale, max_iter) in enumerate(STAGES, start=1):
        args = p2c110.Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = make_functions(params, target_p, direction, y_ref, args, task_mode=task_mode)
        solve = p2c110._solve(y, current_i, funcs, args)
        load = p2c110.p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        tip_p = np.asarray(solve["out"]["tip_p"], dtype=float).reshape(3)
        error = tip_p - target_p
        projected = float(abs(np.dot(error, direction)))
        lateral = float(np.linalg.norm(error - np.dot(error, direction) * direction))
        row = {
            "case": spec.name,
            "task_mode": task_mode,
            "stage": stage,
            "position_scale": position_scale,
            "max_iter": max_iter,
            "force_scale": FORCE_SCALE,
            "torque_scale": TORQUE_SCALE,
            "current_alpha": CURRENT_ALPHA,
            "target_displacement_m": direction_norm,
            "position_error_m": float(np.linalg.norm(error)),
            "projected_position_error_m": projected,
            "lateral_position_error_m": lateral,
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
        row["task_error_gate"] = bool(projected <= 5.0e-5 if task_mode == "effective_1d" else row["position_error_m"] <= 5.0e-5)
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


def plot_rows(final_rows: list[dict[str, Any]]) -> None:
    """Plot final gate metrics."""
    labels = [f"{row['case'].replace('_rand15_a010', '')}\n{row['task_mode']}" for row in final_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].bar(labels, [row["position_error_m"] for row in final_rows], color="#f4a261")
    axes[0, 0].axhline(5.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("末端位置误差")
    axes[0, 1].bar(labels, [row["stationarity_norm"] for row in final_rows], color="#457b9d")
    axes[0, 1].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("一阶最优性")
    axes[1, 0].bar(labels, [row["jacobian_condition"] for row in final_rows], color="#2a9d8f")
    axes[1, 0].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("条件数")
    axes[1, 1].bar(labels, [row["max_interface_force_jump_over_rigid_force"] for row in final_rows], color="#e76f51")
    axes[1, 1].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("载荷传播比例")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c131 单点磁体位置配置边界")
    fig.savefig(OUT_DIR / "n1_configuration_boundary_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    lines = "\n".join(
        f"- {row['case']}，{row['task_mode']}：门禁 {row['baseline_gate']}；"
        f"位置误差 {row['position_error_m']:.3e} m；一阶最优性 {row['stationarity_norm']:.3e}；"
        f"条件数 {row['jacobian_condition']:.3e}；载荷传播比例 {row['max_interface_force_jump_over_rigid_force']:.3e}；"
        f"失败分类 {row['failure_class']}。"
        for row in final_rows
    )
    pass_rows = [row for row in final_rows if row["baseline_gate"]]
    if pass_rows:
        conclusion = "本轮发现至少一个单点配置通过基线门禁。该结果只能说明配置依赖性存在，尚不能发布灵敏度；下一步必须先对通过配置做多初值分支一致性复核，复核通过后才允许扰动重求解。"
    else:
        conclusion = "本轮三种单点磁体位置均未通过基线门禁。结合 P2c126-P2c130，单点磁控在当前残差模型和电流尺度下更可能受有效驱动秩、条件数病态和物理闭合不足共同限制，而不是单一末端磁体位置导致。"
    record = f"""# P2c131 单点磁体位置配置边界

## 实验目的

P2c130 表明 `n1_tip_z_rand15` 即使降维和降低电流比例仍不能通过基线门禁。本轮更换单点磁体所在刚段，判断单点磁控失败是否具有明显配置依赖性。

## 实验方案

构造三种单磁体配置：基端刚段轴向磁化、中段刚段轴向磁化、末端刚段轴向磁化。电流采用 `n1_tip_z_rand15` 的随机电流并乘以 `alpha=0.10`。每种配置先用 FK oracle 生成目标末端位置，再在显式刚段载荷残差模型下求解 shooting 基线门禁。

任务分两类：三维末端位置任务和有效一维任务。有效一维任务只约束从零电流末端到目标电流末端的位移方向；横向误差只记录，不作为该任务的否决项。

## 结果

{lines}

## 结论

{conclusion}

## 下一步计划

若本轮没有通过样本，应将单点磁控主线转为不可发布边界归纳，并优先推进两点磁控三维末端位置灵敏度；若存在通过样本，应只对通过样本继续分支一致性复核。本轮使用 FK oracle 生成目标和初始内力，因此只能作为配置筛选证据，不能作为灵敏度发布证据。

## 耗时

- 墙钟耗时：{summary['wall_s']:.2f} s。
- 重求解净耗时：{summary['total_solve_steady_s']:.2f} s。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Run the configuration boundary audit."""
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for spec in cases():
        for task_mode in TASK_MODES:
            rows, final = run_variant(spec, task_mode)
            all_rows.extend(rows)
            final_rows.append(final)
    pass_count = sum(row["baseline_gate"] for row in final_rows)
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "n1_configuration_candidate_found" if pass_count else "n1_configuration_boundary_failed",
        "current_alpha": CURRENT_ALPHA,
        "task_modes": list(TASK_MODES),
        "case_count": len(cases()),
        "tested_final_variants": len(final_rows),
        "pass_count": pass_count,
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
            json.dumps({"case": row["case"], "task_mode": row["task_mode"], "failure_class": row["failure_class"], "baseline_gate": row["baseline_gate"]}, ensure_ascii=False) + "\n"
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
