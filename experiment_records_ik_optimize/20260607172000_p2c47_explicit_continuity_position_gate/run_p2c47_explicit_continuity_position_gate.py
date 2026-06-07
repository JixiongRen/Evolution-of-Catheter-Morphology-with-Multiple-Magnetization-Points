"""Gate explicit interface-continuity shooting for n1/n2 position tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
P2C8_DIR = ROOT / "experiment_records_ik_optimize" / "20260604114419_p2c8_continuity_branch_sensitivity"
P2C10T_DIR = ROOT / "experiment_records_ik_optimize" / "20260605102000_p2c10t_interface_slack_shooting"
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
for path in (ROOT, FK_OPT_DIR, P2C8_DIR, P2C10T_DIR, P2C24_DIR):
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
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents
from forward_kinematics_optimized.external_wrench_nondim import compute_external_wrench_total_rigid
from inverse_kinematics.ik_terminal_shooting_jax import (
    _advance_rigid,
    _integrate_flex,
    forward_segment_wrench,
    initial_state,
    normalize_state,
    output_to_numpy,
)

import run_p2c8_continuity_branch_sensitivity as p2c8
import run_p2c10t_interface_slack_shooting as p2c10t
import run_p2c24_interface_weight_continuation_repair as p2c24

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _parse_specs(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            case_name, alpha = raw.split(":", 1)
            specs.append((case_name.strip(), float(alpha)))
    return specs


def _parse_floats(text: str) -> list[float]:
    return [float(raw.strip()) for raw in text.split(",") if raw.strip()]


def _mode_mask(mode: str) -> jnp.ndarray:
    if mode == "none":
        return jnp.asarray([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
    if mode == "torque":
        return jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.float64)
    if mode == "full":
        return jnp.asarray([1, 1, 1, 1, 1, 1], dtype=jnp.float64)
    raise ValueError(f"Unknown boundary mode: {mode}")


def _base_slack_to_segment_y(params: Any, y_slack: np.ndarray, *, steps_per_flex: int) -> np.ndarray:
    """Convert base-plus-interface-jump state to segment-wrench blocks."""
    n_interfaces = max(len(params.flex) - 1, 0)
    base, force_slack, torque_slack = p2c10t._split_state(jnp.asarray(y_slack, dtype=jnp.float64), "wrench", n_interfaces)
    x = initial_state(params, base)
    blocks = [base]
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    for i in range(len(params.flex) - 1):
        x, _flex_points = _integrate_flex(params, x, i, steps_per_flex)
        x, _mid_p, _mid_q = _advance_rigid(params, x, i)
        x = x.at[7:10].add(force_slack[i] * f_ref)
        x = x.at[10:13].add(torque_slack[i] * m_ref)
        x = normalize_state(x)
        blocks.append(jnp.concatenate([x[7:10] / f_ref, x[10:13] / m_ref], axis=0))
    return np.asarray(jax.device_get(jnp.stack(blocks, axis=0).reshape(-1)), dtype=float)


def _rigid_wrench(params: Any, x_prox: jnp.ndarray, i: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    f_total, tau_total = compute_external_wrench_total_rigid(
        x_proximal=x_prox,
        rigid_length=float(params.rigid[i].length),
        gravity=params.gravity_rigid_list[i],
        magnetic_model=params.magnetic_model,
        magnet_params=params.magnet_params_list[i],
        coil_currents=params.coil_currents,
    )
    f_total = f_total + jnp.asarray(params.f_ext_list[i], dtype=jnp.float64).reshape(3)
    tau_total = tau_total + jnp.asarray(params.tau_ext_list[i], dtype=jnp.float64).reshape(3)
    return f_total, tau_total


def _interface_load_metrics(params: Any, y: np.ndarray, *, steps_per_flex: int) -> dict[str, float]:
    blocks = jnp.asarray(y, dtype=jnp.float64).reshape((len(params.flex), 6))
    x = initial_state(params, blocks[0])
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    force_ratios = []
    torque_ratios = []
    jump_norms = []
    for i in range(len(params.flex) - 1):
        x = x.at[7:10].set(blocks[i, 0:3] * f_ref)
        x = x.at[10:13].set(blocks[i, 3:6] * m_ref)
        x = normalize_state(x)
        x, _flex_points = _integrate_flex(params, x, i, steps_per_flex)
        f_total, tau_total = _rigid_wrench(params, x, i)
        x, _mid_p, _mid_q = _advance_rigid(params, x, i)
        next_force = blocks[i + 1, 0:3] * f_ref
        next_torque = blocks[i + 1, 3:6] * m_ref
        jump_force = next_force - x[7:10]
        jump_torque = next_torque - x[10:13]
        force_ratios.append(float(jnp.linalg.norm(jump_force) / jnp.maximum(jnp.linalg.norm(f_total), 1.0e-30)))
        torque_ratios.append(float(jnp.linalg.norm(jump_torque) / jnp.maximum(jnp.linalg.norm(tau_total), 1.0e-30)))
        jump_norms.append(float(jnp.linalg.norm(jnp.concatenate([jump_force / f_ref, jump_torque / m_ref], axis=0))))
    return {
        "max_interface_force_jump_over_rigid_force": max(force_ratios, default=0.0),
        "max_interface_torque_jump_over_rigid_torque": max(torque_ratios, default=0.0),
        "max_interface_jump_norm": max(jump_norms, default=0.0),
    }


def _make_functions(
    params_base: Any,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    args: argparse.Namespace,
    *,
    continuity_scale: float,
    boundary_mode: str,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    mask = _mode_mask(boundary_mode)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def continuity_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        return p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = forward_segment_wrench(params, y, steps_per_flex=args.steps_per_flex)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_c = float(continuity_scale) * p2c8.interface_wrench_residual(params, y, steps_per_flex=args.steps_per_flex)
        r_reg = float(args.reg_scale) * (jnp.asarray(y, dtype=jnp.float64).reshape(-1) - y_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_c.reshape(-1), r_reg.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
        "continuity": jax.jit(continuity_yI),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
    out = output_to_numpy(funcs["output"](y_j, I_j))
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


def _run_one(case_name: str, alpha: float, boundary_mode: str, continuity_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_slack = p2c24._load_saved_target_and_seed(case_name, alpha)
    y0 = _base_slack_to_segment_y(params, y_slack, steps_per_flex=args.steps_per_flex)
    funcs = _make_functions(params, target_p, y0, args, continuity_scale=continuity_scale, boundary_mode=boundary_mode)
    solve = _solve(y0, I0, funcs, args)
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load_metrics = _interface_load_metrics(params, solve["y"], steps_per_flex=args.steps_per_flex)
    row = {
        "case": case_name,
        "alpha": alpha,
        "boundary_mode": boundary_mode,
        "continuity_scale": continuity_scale,
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "steady_s": solve["steady_s"],
        "nfev": solve["nfev"],
        "success": solve["success"],
        **load_metrics,
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
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "刚性磁体段载荷传播失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for case in sorted({row["case"] for row in rows}):
        sub = [row for row in rows if row["case"] == case]
        best = min(
            sub,
            key=lambda row: (
                not row["baseline_gate"],
                row["stationarity_norm"],
                row["continuity_norm"],
                row["max_interface_force_jump_over_rigid_force"],
            ),
        )
        min_cont = min(sub, key=lambda row: row["continuity_norm"])
        out.append(
            {
                "case": case,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "best_boundary_mode": best["boundary_mode"],
                "best_continuity_scale": best["continuity_scale"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_condition": best["jacobian_condition"],
                "best_continuity_norm": best["continuity_norm"],
                "best_position_error_m": best["position_error_m"],
                "best_force_jump_ratio": best["max_interface_force_jump_over_rigid_force"],
                "min_continuity_scale": min_cont["continuity_scale"],
                "min_continuity_norm": min_cont["continuity_norm"],
                "dominant_failure": best["failure_class"],
                "row_count": len(sub),
            }
        )
    return out


def _plot(rows: list[dict[str, Any]]) -> None:
    mode_cn = {"torque": "仅矩", "full": "力矩全闭合"}
    for case in sorted({row["case"] for row in rows}):
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for mode in sorted({row["boundary_mode"] for row in rows}):
            sub = sorted([row for row in rows if row["case"] == case and row["boundary_mode"] == mode], key=lambda row: row["continuity_scale"])
            x = [row["continuity_scale"] for row in sub]
            axes[0].plot(x, [row["stationarity_norm"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
            axes[1].plot(x, [row["continuity_norm"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
            axes[2].plot(x, [row["max_interface_force_jump_over_rigid_force"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
        axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
        axes[1].axhline(1.0e-3, color="black", ls="--", lw=1.0)
        axes[2].axhline(1.5, color="black", ls="--", lw=1.0)
        axes[0].set_ylabel("一阶最优性残差")
        axes[1].set_ylabel("接口连续残差")
        axes[2].set_ylabel("接口力跳变/刚段外力")
        axes[2].set_xlabel("显式接口连续权重")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"{case} 显式接口连续基线门禁")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case}_explicit_continuity_gate_cn.png", dpi=180)
        plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 是否过门 | 最优边界模式 | 最优连续权重 | 一阶最优性 | 条件数 | 接口连续残差 | 接口力跳变比 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    mode_cn = {"torque": "仅矩", "full": "力矩全闭合"}
    for summary in summaries:
        table.append(
            f"| {summary['case']} | {summary['any_gate_pass']} | {mode_cn.get(summary['best_boundary_mode'], summary['best_boundary_mode'])} | "
            f"{summary['best_continuity_scale']:.3g} | {summary['best_stationarity_norm']:.3e} | "
            f"{summary['best_condition']:.3e} | {summary['best_continuity_norm']:.3e} | "
            f"{summary['best_force_jump_ratio']:.3e} | {summary['dominant_failure']} |"
        )
    return "\n".join(
        [
            "# P2c47 显式接口连续基线门禁记录",
            "",
            "## 实验目的",
            "",
            "P2c46 表明继续调节自由接口松弛权重无法形成可信基线。本轮回到分段内力/内矩表述，使用显式接口连续残差约束相邻柔性段之间的内力/内矩传递，检验当前单点与两点 position-only 工况是否能通过基线门禁。",
            "",
            "## 实验方案",
            "",
            "将 P2c22 保存的接口松弛解转换为 18 维分段内力/内矩初值。残差包含末端位置、末端边界、显式接口连续和弱正则。只扫描少量接口连续权重和两个有意义的边界模式，不做大范围扫参。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 边界模式：`{args.modes}`",
            f"- 显式接口连续权重：`{args.continuity_scales}`",
            f"- 边界权重：`{args.boundary_scale}`",
            f"- 弱正则权重：`{args.reg_scale}`",
            f"- 最大函数调用次数：`{args.max_iter}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若显式接口连续残差不能同时降低一阶最优性、接口连续残差和接口力跳变比例，则当前失败应归因于分段内力/内矩表述下的任务-边界-连续约束相容性不足，而不是自由松弛变量权重选择问题。",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门基线，则进入小扰动重求解灵敏度复核。",
            "2. 若无过门基线，但接口连续显著改善，则尝试分阶段延拓或更可靠初值。",
            "3. 若无过门基线且一阶最优性持续退化，则停止当前工况扰动扫描，转入有效子空间或不可发布边界结论。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--modes", default="torque,full")
    parser.add_argument("--continuity_scales", default="0.25,1,4,16")
    parser.add_argument("--boundary_scale", type=float, default=0.25)
    parser.add_argument("--reg_scale", type=float, default=1.0e-4)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    continuity_scales = _parse_floats(args.continuity_scales)
    for case_name, alpha in _parse_specs(args.specs):
        for mode in modes:
            for continuity_scale in continuity_scales:
                row = _run_one(case_name, alpha, mode, continuity_scale, args)
                rows.append(row)
                if not row["baseline_gate"]:
                    _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})
    summaries = _summaries(rows)
    _plot(rows)
    _write_csv(OUT_DIR / "explicit_continuity_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "summaries": summaries,
        "rows": rows,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(args, summaries), encoding="utf-8")
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
