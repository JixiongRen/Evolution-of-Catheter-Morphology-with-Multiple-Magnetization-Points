"""Probe low-dimensional beta load transfer for the n1 branch."""

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
for path in (ROOT, EXPERIMENT_ROOT, P2C57_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import plotting_fonts
import run_p2c57_force_continuity_weight_repair as p2c57


jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n1_tip_z_rand15"
ALPHA = 0.10
REG_SCALE = 8.0e-4
BOUNDARY_SCALE = 0.25
TORQUE_SCALE = 0.25
LOAD_FLOOR = 1.0e-3
FORCE_SCALE = 4.0
BETA_REG_SCALES = (0.01, 0.1)
BETA_REF = 0.99
STAGES = (
    (10.0, 80),
    (20.0, 80),
    (40.0, 120),
    (60.0, 160),
)


class Args:
    """Settings for the local least-squares solve."""

    boundary_scale = BOUNDARY_SCALE
    reg_scale = REG_SCALE
    max_iter = 120
    steps_per_flex = 4
    position_scale = 60.0


def _jsonable(value: Any) -> Any:
    """Convert values to JSON-safe objects."""
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


def _split_v(v: jnp.ndarray, n_flex: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split segment wrenches and beta logits."""
    v = jnp.asarray(v, dtype=jnp.float64).reshape(-1)
    y_len = 6 * n_flex
    return v[:y_len], v[y_len:]


def _beta_from_eta(eta: jnp.ndarray) -> jnp.ndarray:
    """Map logits to load-transfer factors in [0, 1]."""
    return jax.nn.sigmoid(jnp.asarray(eta, dtype=jnp.float64))


def _eta_from_beta(beta: float, n: int) -> np.ndarray:
    """Return logits for a constant beta value."""
    beta = float(np.clip(beta, 1.0e-6, 1.0 - 1.0e-6))
    return np.full(n, np.log(beta / (1.0 - beta)), dtype=float)


def _beta_force_residual(params: Any, v: jnp.ndarray, *, steps_per_flex: int, load_floor: float) -> jnp.ndarray:
    """Return beta-scaled force continuity residuals."""
    y, eta = _split_v(v, len(params.flex))
    beta = _beta_from_eta(eta)
    blocks = y.reshape((len(params.flex), 6))
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
        pre_force = x[7:10]
        f_total, _tau_total = p2c57.p2c47._rigid_wrench(params, x, i)
        next_force = blocks[i + 1, 0:3] * f_ref
        beta_post_force = pre_force - beta[i] * f_total
        jump_force = next_force - beta_post_force
        denom = jnp.maximum(jnp.linalg.norm(f_total), floor)
        pieces.append(jump_force / denom)
        x, _mid_p, _mid_q = p2c57.p2c47._advance_rigid(params, x, i)
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate(pieces, axis=0)


def _make_functions(params_base: Any, target_p: np.ndarray, y_ref: np.ndarray, args: Args, *, beta_reg_scale: float) -> dict[str, Any]:
    """Build residual and Jacobian functions."""
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    y_ref_j = jnp.asarray(y_ref, dtype=jnp.float64).reshape(-1)
    beta_ref_j = jnp.full((max(len(params_base.flex) - 1, 0),), BETA_REF, dtype=jnp.float64)
    mask = p2c57.p2c47._mode_mask("full")

    def params_for_I(I: jnp.ndarray) -> Any:
        return p2c57.p2c47.with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_vI(v: jnp.ndarray, I: jnp.ndarray) -> Any:
        y, _eta = _split_v(v, len(params_base.flex))
        return p2c57.p2c47.forward_segment_wrench(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def continuity_vI(v: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        y, _eta = _split_v(v, len(params_base.flex))
        return p2c57.p2c47.p2c8.interface_wrench_residual(params_for_I(I), y, steps_per_flex=args.steps_per_flex)

    def beta_v(v: jnp.ndarray) -> jnp.ndarray:
        _y, eta = _split_v(v, len(params_base.flex))
        return _beta_from_eta(eta)

    def residual_vI(v: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        y, _eta = _split_v(v, len(params.flex))
        out = p2c57.p2c47.forward_segment_wrench(params, y, steps_per_flex=args.steps_per_flex)
        raw = p2c57.p2c47.p2c8.interface_wrench_residual(params, y, steps_per_flex=args.steps_per_flex).reshape((-1, 6))
        beta = beta_v(v)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_force = float(FORCE_SCALE) * _beta_force_residual(params, v, steps_per_flex=args.steps_per_flex, load_floor=LOAD_FLOOR)
        r_torque = float(TORQUE_SCALE) * raw[:, 3:6].reshape(-1)
        r_reg_y = float(args.reg_scale) * (y - y_ref_j)
        r_reg_beta = float(beta_reg_scale) * (beta - beta_ref_j)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_force.reshape(-1), r_torque.reshape(-1), r_reg_y.reshape(-1), r_reg_beta.reshape(-1)])

    return {
        "residual": jax.jit(residual_vI),
        "jacobian": jax.jit(jax.jacfwd(residual_vI, argnums=0)),
        "output": jax.jit(output_vI),
        "continuity": jax.jit(continuity_vI),
        "beta": jax.jit(beta_v),
    }


def _solve(v0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], args: Args) -> dict[str, Any]:
    """Solve one least-squares stage."""
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(v0), I_j).block_until_ready()
    t0 = time.perf_counter()
    opt = least_squares(
        lambda vv: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(vv), I_j)), dtype=float),
        v0,
        jac=lambda vv: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(vv), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(v0),
        loss="linear",
        max_nfev=int(args.max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    v = np.asarray(opt.x, dtype=float)
    v_j = jnp.asarray(v, dtype=jnp.float64)
    y_j, _eta_j = _split_v(v_j, len(funcs["params"].flex))
    r = np.asarray(jax.device_get(funcs["residual"](v_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](v_j, I_j)), dtype=float)
    out = p2c57.p2c47.output_to_numpy(funcs["output"](v_j, I_j))
    continuity = np.asarray(jax.device_get(funcs["continuity"](v_j, I_j)), dtype=float).reshape(-1)
    beta = np.asarray(jax.device_get(funcs["beta"](v_j)), dtype=float)
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "v": v,
        "y": np.asarray(jax.device_get(y_j), dtype=float),
        "beta": beta,
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


def _run_variant(beta_reg_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run staged continuation for one beta regularization weight."""
    params, current_i = p2c57.p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)
    n_interfaces = max(len(params.flex) - 1, 0)
    eta = _eta_from_beta(BETA_REF, n_interfaces)
    v = np.concatenate([np.asarray(y, dtype=float), eta], axis=0)
    y_ref = np.asarray(y, dtype=float).copy()
    rows = []
    for stage, (position_scale, max_iter) in enumerate(STAGES, start=1):
        args = Args()
        args.position_scale = float(position_scale)
        args.max_iter = int(max_iter)
        funcs = _make_functions(params, target_p, y_ref, args, beta_reg_scale=beta_reg_scale)
        funcs["params"] = params
        solve = _solve(v, current_i, funcs, args)
        load = p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
        boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
        beta = np.asarray(solve["beta"], dtype=float)
        row = {
            "case": CASE_NAME,
            "alpha": ALPHA,
            "stage": stage,
            "position_scale": position_scale,
            "beta_reg_scale": beta_reg_scale,
            "force_scale": FORCE_SCALE,
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
            "beta_min": float(np.min(beta)) if beta.size else 1.0,
            "beta_max": float(np.max(beta)) if beta.size else 1.0,
            "beta_mean": float(np.mean(beta)) if beta.size else 1.0,
            "beta_deviation_from_one": float(np.max(np.abs(beta - 1.0))) if beta.size else 0.0,
            "beta_values": beta.tolist(),
            "message": solve["message"],
        }
        row["publish_gate"] = bool(
            row["stationarity_norm"] <= 1.0e-5
            and row["jacobian_condition"] < 1.0e7
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
            and row["beta_deviation_from_one"] <= 0.1
        )
        if row["publish_gate"]:
            row["failure_class"] = "通过"
        elif row["beta_deviation_from_one"] > 0.1:
            row["failure_class"] = "载荷传递比例偏离物理闭合"
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
        v = solve["v"]
    return rows, rows[-1]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot final gate metrics for beta variants."""
    plotting_fonts.configure_matplotlib_cjk()
    finals = [row for row in rows if row["stage"] == len(STAGES)]
    labels = [f"beta正则 {row['beta_reg_scale']:g}" for row in finals]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].bar(labels, [row["position_error_m"] * 1e6 for row in finals], color="#2a9d8f")
    axes[0].set_ylabel("末端误差/微米")
    axes[0].set_title("末端位置")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [row["beta_deviation_from_one"] for row in finals], color="#e76f51")
    axes[1].axhline(0.1, color="black", linestyle="--", label="物理偏离门禁")
    axes[1].set_ylabel("max |beta-1|")
    axes[1].set_title("载荷传递偏离")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].bar(labels, [row["max_interface_force_jump_over_rigid_force"] for row in finals], color="#457b9d")
    axes[2].axhline(1.5, color="black", linestyle="--", label="载荷门禁")
    axes[2].set_ylabel("物理载荷传播比例")
    axes[2].set_title("原物理力链")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P2c111 n1 低维 beta 载荷传递诊断")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_beta_load_transfer_cn.png", dpi=180)
    plt.close(fig)


def _record(summary: dict[str, Any]) -> str:
    """Render the Chinese experiment report."""
    finals = summary["final_rows"]
    table = [
        "| beta正则 | 末端误差/m | 一阶最优性 | 条件数 | 原物理载荷比 | beta范围 | 判定 |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in finals:
        table.append(
            f"| {row['beta_reg_scale']:.3g} | {row['position_error_m']:.3e} | {row['stationarity_norm']:.3e} | "
            f"{row['jacobian_condition']:.3e} | {row['max_interface_force_jump_over_rigid_force']:.3e} | "
            f"[{row['beta_min']:.3f}, {row['beta_max']:.3f}] | {row['failure_class']} |"
        )
    best_pos = min(finals, key=lambda row: row["position_error_m"])
    best_beta = min(finals, key=lambda row: row["beta_deviation_from_one"])
    return "\n".join(
        [
            "# P2c111 n1 低维 beta 载荷传递诊断",
            "",
            "## 实验目的",
            "",
            "P2c110 说明硬力连续可以闭合力链，但会破坏末端目标或其他门禁。本轮引入低维载荷传递比例 `beta_i` 作为反事实诊断变量，判断原目标是否需要明显偏离物理力链才能拟合。",
            "",
            "## 方法",
            "",
            "扩展状态为每段近端内力/内矩加每个接口一个 `beta_i`。接口力残差为",
            "",
            "$$r_{F,i}=\\frac{n_{i+1}^{0}-\\left(n_i^{L_i}-\\beta_i F_i\\right)}{\\max(\\|F_i\\|,F_{floor})}.$$",
            "",
            "`beta_i=1` 对应原刚段外力完全传递；若求解必须让 `beta_i` 明显偏离 1，说明当前目标依赖非物理载荷传递缺口，不能发布灵敏度。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"末端误差最小的方案为 beta 正则 `{best_pos['beta_reg_scale']:.3g}`，末端误差 `{best_pos['position_error_m']:.3e} m`，beta 范围 `[{best_pos['beta_min']:.3f}, {best_pos['beta_max']:.3f}]`。"
            f"最接近物理闭合的方案为 beta 正则 `{best_beta['beta_reg_scale']:.3g}`，最大 beta 偏离 `{best_beta['beta_deviation_from_one']:.3e}`。",
            "",
            "若 beta 明显偏离 1 才能改善末端拟合，n1 当前目标应归类为物理闭合下不可发布或需要更高阶模型修复；若 beta 接近 1 仍无法过门，则应继续查条件数和接口残差分块，而不是发布灵敏度。",
            "",
            "## 下一步计划",
            "",
            "1. 若 beta 偏离明显，整理 n1 工况为不可发布边界，并给出工作空间/目标选择建议。",
            "2. 若 beta 接近 1 但仍失败，复跑分块最优性，定位剩余病态残差。",
            "3. 不把 beta 模型接入外层 IK；它只用于解释 n1 失败机制。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for beta_reg in BETA_REG_SCALES:
        rows, final = _run_variant(beta_reg)
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
        "beta_reg_scales": list(BETA_REG_SCALES),
        "stage_rows": all_rows,
        "final_rows": final_rows,
        "publishable_count": sum(row["publish_gate"] for row in final_rows),
        "best_position_error_m": min(row["position_error_m"] for row in final_rows),
        "min_beta_deviation_from_one": min(row["beta_deviation_from_one"] for row in final_rows),
        "recommended_next": "n1 boundary recommendation or residual block audit depending on beta outcome",
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
    print(json.dumps(_jsonable({"publishable_count": summary["publishable_count"], "best_position_error_m": summary["best_position_error_m"], "min_beta_deviation_from_one": summary["min_beta_deviation_from_one"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
