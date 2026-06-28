"""Probe hard load-scaled force continuity for the n1 branch."""

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
P2C57_DIR = EXPERIMENT_ROOT / "20260607200500_p2c57_force_continuity_weight_repair"
P2C84_DIR = EXPERIMENT_ROOT / "20260628121600_p2c84_block_stationarity_load_closure_audit"
for path in (ROOT, EXPERIMENT_ROOT, P2C57_DIR, P2C84_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import plotting_fonts
import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c84_block_stationarity_load_closure_audit as p2c84


jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n1_tip_z_rand15"
ALPHA = 0.10
REG_SCALE = 8.0e-4
BOUNDARY_SCALE = 0.25
TORQUE_SCALE = 0.25
LOAD_FLOOR = 1.0e-3
FORCE_SCALES = (1.0, 4.0, 16.0)
STAGES = (
    (10.0, 120),
    (20.0, 120),
    (40.0, 120),
    (60.0, 240),
)


class Args:
    """Settings for the local least-squares solve."""

    boundary_scale = BOUNDARY_SCALE
    reg_scale = REG_SCALE
    max_iter = 120
    steps_per_flex = 4
    position_scale = 60.0


def _jsonable(value: Any) -> Any:
    """Convert arrays and paths into JSON-safe values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _force_scaled_residual(params: Any, y: jnp.ndarray, *, steps_per_flex: int, load_floor: float) -> jnp.ndarray:
    """Return force jumps normalized by local rigid load."""
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = p2c57.p2c47.initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    floor = float(load_floor) * max(f_ref, 1.0e-12)
    pieces = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = p2c57.p2c47.normalize_state(x)
        x, _flex_points = p2c57.p2c47._integrate_flex(params, x, i, steps_per_flex)
        f_total, _tau_total = p2c57.p2c47._rigid_wrench(params, x, i)
        x, _mid_p, _mid_q = p2c57.p2c47._advance_rigid(params, x, i)
        next_force = blocks[i + 1, 0:3] * f_ref
        jump_force = next_force - x[7:10]
        denom = jnp.maximum(jnp.linalg.norm(f_total), floor)
        pieces.append(jump_force / denom)
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(pieces, axis=0)


def _make_functions(params_base: Any, target_p: np.ndarray, y_ref: np.ndarray, args: Args, *, force_scale: float) -> dict[str, Any]:
    """Build residual and Jacobian functions."""
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    mask = p2c57.p2c47._mode_mask("full")

    def params_for_I(I: jnp.ndarray) -> Any:
        return p2c57.p2c47.with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c57.p2c47.forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return p2c57.p2c47.p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = p2c57.p2c47.forward_segment_wrench(params, y, steps_per_flex=args.steps_per_flex)
        raw = p2c57.p2c47.p2c8.interface_wrench_residual(params, y, steps_per_flex=args.steps_per_flex).reshape((-1, 6))
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_force = float(force_scale) * _force_scaled_residual(params, y, steps_per_flex=args.steps_per_flex, load_floor=LOAD_FLOOR)
        r_torque = float(TORQUE_SCALE) * raw[:, 3:6].reshape(-1)
        r_reg = float(args.reg_scale) * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_force.reshape(-1), r_torque.reshape(-1), r_reg.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
        "continuity": jax.jit(continuity_yI),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], args: Args) -> dict[str, Any]:
    """Solve one least-squares stage."""
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j).block_until_ready()
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=int(args.max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    out = p2c57.p2c47.output_to_numpy(funcs["output"](y_j, I_j))
    continuity = np.asarray(jax.device_get(funcs["continuity"](y_j, I_j)), dtype=float).reshape(-1)
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "continuity_norm": float(np.linalg.norm(continuity)),
        "message": str(opt.message),
    }


def _run_variant(force_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run staged continuation for one force-continuity weight."""
    params, current_i = p2c57.p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)
    y_ref = np.asarray(y, dtype=float).copy()
    rows = []
    final_solve = None
    final_args = None
    for stage, (position_scale, max_iter) in enumerate(STAGES, start=1):
        args = Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = _make_functions(params, target_p, y_ref, args, force_scale=force_scale)
        solve = _solve(y, current_i, funcs, args)
        load = p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        row = {
            "case": CASE_NAME,
            "alpha": ALPHA,
            "stage": stage,
            "position_scale": position_scale,
            "force_scale": force_scale,
            "torque_scale": TORQUE_SCALE,
            "load_floor": LOAD_FLOOR,
            "reg_scale": REG_SCALE,
            "success": solve["success"],
            "nfev": solve["nfev"],
            "steady_s": solve["steady_s"],
            "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
            "stationarity_norm": solve["stationarity_norm"],
            "jacobian_condition": solve["jacobian_condition"],
            "jacobian_rank": solve["jacobian_rank"],
            "continuity_norm": solve["continuity_norm"],
            "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
            "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
            "max_interface_force_jump_over_rigid_force": float(load["max_interface_force_jump_over_rigid_force"]),
            "max_interface_torque_jump_over_rigid_torque": float(load["max_interface_torque_jump_over_rigid_torque"]),
            "message": solve["message"],
        }
        row["publish_gate"] = bool(
            row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] < 1.0e7
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
        )
        if row["publish_gate"]:
            row["failure_class"] = "通过"
        elif row["max_interface_force_jump_over_rigid_force"] > 1.5:
            row["failure_class"] = "载荷传播失配"
        elif row["stationarity_norm"] > 1.0e-5:
            row["failure_class"] = "一阶最优性退化"
        elif row["jacobian_condition"] >= 1.0e7:
            row["failure_class"] = "条件数病态"
        elif row["continuity_norm"] > 1.0e-3:
            row["failure_class"] = "接口连续残差超限"
        else:
            row["failure_class"] = "综合失败"
        rows.append(row)
        y = solve["y"]
        final_solve = solve
        final_args = args
    assert final_solve is not None and final_args is not None
    return rows, rows[-1]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot gate metrics for all force-scale variants."""
    plotting_fonts.configure_matplotlib_cjk()
    finals = [row for row in rows if row["stage"] == len(STAGES)]
    labels = [f"力权重 {row['force_scale']:g}" for row in finals]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(labels, [row["max_interface_force_jump_over_rigid_force"] for row in finals], color="#e76f51")
    axes[0].axhline(1.5, color="black", linestyle="--", label="门禁")
    axes[0].set_ylabel("载荷传播比例")
    axes[0].set_title("力连续性")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["stationarity_norm"] for row in finals], color="#457b9d")
    axes[1].axhline(1.0e-5, color="black", linestyle="--", label="门禁")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("一阶最优性")
    axes[1].set_title("最优性")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(labels, [row["position_error_m"] * 1e6 for row in finals], color="#2a9d8f")
    axes[2].set_ylabel("末端误差/微米")
    axes[2].set_title("末端位置")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c110 n1 硬力连续归一化实验")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_hard_force_scaled_continuity_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese experiment report."""
    finals = summary["final_rows"]
    table = [
        "| 力权重 | 末端误差/m | 一阶最优性 | 条件数 | 接口连续 | 载荷比 | 判定 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in finals:
        table.append(
            f"| {row['force_scale']:.1f} | {row['position_error_m']:.3e} | {row['stationarity_norm']:.3e} | "
            f"{row['jacobian_condition']:.3e} | {row['continuity_norm']:.3e} | "
            f"{row['max_interface_force_jump_over_rigid_force']:.3e} | {row['failure_class']} |"
        )
    best = min(finals, key=lambda row: row["max_interface_force_jump_over_rigid_force"])
    publishable_count = sum(row["publish_gate"] for row in finals)
    if publishable_count:
        conclusion = (
            f"本轮有 `{publishable_count}` 组权重通过全部门禁。下一步应复跑 P2c84 分块最优性和有效子空间秩审计，"
            "再决定是否恢复 n1 扰动灵敏度扫描。"
        )
    else:
        conclusion = (
            f"硬力连续归一化可以把最佳载荷传播比例压至 `{best['max_interface_force_jump_over_rigid_force']:.3e}`，"
            f"但最佳权重 `{best['force_scale']:.1f}` 仍未通过全部门禁。主要代价是末端误差升至 `{best['position_error_m']:.3e} m`，"
            "且接口总连续、一阶最优性或条件数仍存在超限。说明 n1 当前工作点在严格力链闭合下可能与原末端目标不相容；"
            "下一步应进入低维物理载荷传递参数模型，而不是继续盲目增大硬力连续权重。"
        )
    return "\n".join(
        [
            "# P2c110 n1 硬力连续归一化实验",
            "",
            "## 实验目的",
            "",
            "P2c109 建议优先尝试“硬力连续 + 刚段外力尺度归一化”。本轮在实验脚本中实现该残差，不修改生产接口，复测 `n1_tip_z_rand15, alpha=0.10` 是否能够把载荷传播比例压回 1.5 以下。",
            "",
            "## 方法",
            "",
            "状态变量仍为每段近端内力和内矩，不引入任意接口松弛。接口力残差写为",
            "",
            "$$r_{F,i}=\\frac{n_{i+1}^{0}-n_i^{L_i}-F_i}{\\max(\\|F_i\\|,F_{floor})}.$$",
            "",
            "其中 `F_floor=1e-3 * F_ref`。同时保留末端位置残差、末端边界残差、力矩连续残差和相对参考状态的弱正则项。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            conclusion,
            "",
            "本轮仍不发布 n1 灵敏度。只有当载荷传播、一阶最优性、条件数、接口连续全部过门后，才允许恢复有效子空间扰动审计。",
            "",
            "## 下一步计划",
            "",
            "1. 若所有硬力连续权重仍失败，进入 P2c111：低维物理载荷传递参数 `beta_i`。",
            "2. 若某一权重过门，复跑 P2c84 分块最优性和有效子空间秩审计。",
            "3. 不使用降低门禁或只看末端误差的方式宣布 n1 可发布。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for force_scale in FORCE_SCALES:
        rows, final = _run_variant(force_scale)
        all_rows.extend(rows)
        final_rows.append(final)
    failures = [
        {"failure": row["failure_class"], "evidence": row}
        for row in final_rows
        if not row["publish_gate"]
    ]
    summary = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "case": CASE_NAME,
        "alpha": ALPHA,
        "force_scales": list(FORCE_SCALES),
        "stage_rows": all_rows,
        "final_rows": final_rows,
        "publishable_count": sum(row["publish_gate"] for row in final_rows),
        "best_force_ratio": min(row["max_interface_force_jump_over_rigid_force"] for row in final_rows),
        "recommended_next": "P2c111 low-dimensional physical load-transfer beta model",
        "jax_devices": [str(device) for device in jax.devices()],
    }
    _write_csv(OUT_DIR / "stage_rows.csv", all_rows)
    _write_csv(OUT_DIR / "summary.csv", final_rows)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(all_rows)
    (OUT_DIR / "record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps(_jsonable({"publishable_count": summary["publishable_count"], "best_force_ratio": summary["best_force_ratio"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
