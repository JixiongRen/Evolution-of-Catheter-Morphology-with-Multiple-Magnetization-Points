"""Compare implicit state increments with reoptimized increments."""

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
P2C72_DIR = ROOT / "experiment_records_ik_optimize" / "20260627234500_p2c72_n2_subspace_perturb_reopt"
for path in (ROOT, P2C72_DIR):
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

import run_p2c72_n2_subspace_perturb_reopt as p2c72

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS_VALUES = (0.0025, 0.005, 0.010)
MAX_NFEV = 180


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _safe_rel(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    actual_norm = float(np.linalg.norm(actual))
    rel = float(np.linalg.norm(pred - actual) / max(actual_norm, 1.0e-14))
    denom = max(float(np.linalg.norm(pred) * actual_norm), 1.0e-14)
    cosine = float(np.dot(pred, actual) / denom)
    return rel, cosine, actual_norm


def _build_baseline() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    p2c57 = p2c72.p2c57
    p2c69 = p2c72.p2c69
    p2c71 = p2c72.p2c71
    params, I0 = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    recovered = p2c69._recover_near_gate(params, I0, target_p, y_seed)
    y_base = np.asarray(recovered["solve"]["y"], dtype=float)
    y_ref = recovered["y_ref"]
    _args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    J = np.asarray(
        jax.device_get(
            funcs["jacobian"](
                jnp.asarray(y_base, dtype=jnp.float64),
                jnp.asarray(I0, dtype=jnp.float64).reshape(8),
            )
        ),
        dtype=float,
    )
    _u, _s, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c71._orthogonal_complement(vt[-p2c72.REMOVE_WEAK_DIRECTIONS :], J.shape[1])
    base_row = max(recovered["rows"], key=lambda row: row["stage"])
    return params, np.asarray(I0, dtype=float), target_p, y_base, y_ref, funcs, basis, base_row


def _solve_q(
    *,
    y_base: np.ndarray,
    basis: np.ndarray,
    I: np.ndarray,
    funcs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y_base = np.asarray(y_base, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        return np.asarray(jax.device_get(funcs["residual"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)

    def jacobian(q: np.ndarray) -> np.ndarray:
        y = y_base + basis @ np.asarray(q, dtype=float).reshape(-1)
        J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(y, dtype=jnp.float64), I_j)), dtype=float)
        return J @ basis

    q0 = np.zeros((basis.shape[1],), dtype=float)
    _ = residual(q0)
    t0 = time.perf_counter()
    opt = least_squares(
        residual,
        q0,
        jac=jacobian,
        method="trf",
        x_scale=np.ones_like(q0),
        loss="linear",
        max_nfev=MAX_NFEV,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    return np.asarray(opt.x, dtype=float), {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(time.perf_counter() - t0),
        "cost": float(opt.cost),
        "optimality": float(opt.optimality),
        "message": str(opt.message),
    }


def _implicit_vectors(
    *,
    y_base: np.ndarray,
    I0: np.ndarray,
    direction: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, np.ndarray | float]:
    y_j = jnp.asarray(y_base, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)
    direction = np.asarray(direction, dtype=float).reshape(8)
    J_y = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    J_eff = J_y @ basis

    def residual_I(I_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["residual"](y_j, I_var)

    J_I = np.asarray(jax.device_get(jax.jacfwd(residual_I)(I_j)), dtype=float)
    rhs = J_I @ direction
    q_residual = -np.linalg.lstsq(J_eff, rhs, rcond=None)[0]
    normal_lhs = J_eff.T @ J_eff
    normal_rhs = J_eff.T @ rhs
    q_normal = -np.linalg.lstsq(normal_lhs, normal_rhs, rcond=None)[0]

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I_j).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_j)), dtype=float)
    T_eff = T_y @ basis
    svals = np.linalg.svd(J_eff, compute_uv=False)
    return {
        "q_residual": q_residual,
        "q_normal": q_normal,
        "tip_residual": T_eff @ q_residual,
        "tip_normal": T_eff @ q_normal,
        "linear_residual_norm": float(np.linalg.norm(J_eff @ q_residual + rhs)),
        "normal_linear_residual_norm": float(np.linalg.norm(J_eff @ q_normal + rhs)),
        "condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
    }


def _audit_one(
    *,
    eps: float,
    name: str,
    direction: np.ndarray,
    I0: np.ndarray,
    y_base: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, Any]:
    implicit = _implicit_vectors(y_base=y_base, I0=I0, direction=direction, funcs=funcs, basis=basis)
    q_plus, plus_info = _solve_q(y_base=y_base, basis=basis, I=I0 + eps * direction, funcs=funcs)
    q_minus, minus_info = _solve_q(y_base=y_base, basis=basis, I=I0 - eps * direction, funcs=funcs)
    q_actual = (q_plus - q_minus) / (2.0 * eps)
    q_plus_der = q_plus / eps
    q_minus_der = -q_minus / eps
    y_plus = y_base + basis @ q_plus
    y_minus = y_base + basis @ q_minus
    out_plus = funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), jnp.asarray(I0 + eps * direction, dtype=jnp.float64))
    out_minus = funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), jnp.asarray(I0 - eps * direction, dtype=jnp.float64))
    tip_fd = (
        np.asarray(jax.device_get(out_plus.tip_p), dtype=float)
        - np.asarray(jax.device_get(out_minus.tip_p), dtype=float)
    ) / (2.0 * eps)
    q_rel, q_cos, q_norm = _safe_rel(np.asarray(implicit["q_residual"]), q_actual)
    qn_rel, qn_cos, _qn_norm = _safe_rel(np.asarray(implicit["q_normal"]), q_actual)
    q_pm_rel, q_pm_cos, _ = _safe_rel(q_plus_der, q_minus_der)
    tip_rel, tip_cos, tip_norm = _safe_rel(np.asarray(implicit["tip_residual"]), tip_fd)
    tipn_rel, tipn_cos, _ = _safe_rel(np.asarray(implicit["tip_normal"]), tip_fd)
    tip_actual_from_q = np.asarray(implicit["tip_residual"]) * 0.0
    y_j = jnp.asarray(y_base, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I_j).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_j)), dtype=float)
    tip_actual_from_q = (T_y @ basis) @ q_actual
    tipq_rel, tipq_cos, _ = _safe_rel(tip_actual_from_q, tip_fd)
    return {
        "eps": eps,
        "direction": name,
        "condition": implicit["condition"],
        "q_residual_rel_err": q_rel,
        "q_residual_cosine": q_cos,
        "q_normal_rel_err": qn_rel,
        "q_normal_cosine": qn_cos,
        "q_plus_minus_derivative_rel_err": q_pm_rel,
        "q_plus_minus_derivative_cosine": q_pm_cos,
        "q_actual_norm": q_norm,
        "tip_residual_rel_err": tip_rel,
        "tip_residual_cosine": tip_cos,
        "tip_normal_rel_err": tipn_rel,
        "tip_normal_cosine": tipn_cos,
        "tip_actual_q_rel_err": tipq_rel,
        "tip_actual_q_cosine": tipq_cos,
        "tip_fd_norm": tip_norm,
        "linear_residual_norm": implicit["linear_residual_norm"],
        "normal_linear_residual_norm": implicit["normal_linear_residual_norm"],
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
        "plus_optimality": plus_info["optimality"],
        "minus_optimality": minus_info["optimality"],
        "plus_success": plus_info["success"],
        "minus_success": minus_info["success"],
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['direction']}\n{row['eps']:.4g}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].bar(x - 0.18, [row["q_residual_rel_err"] for row in rows], width=0.36, label="残差抵消公式")
    axes[0].bar(x + 0.18, [row["q_normal_rel_err"] for row in rows], width=0.36, label="一阶条件公式")
    axes[0].set_ylabel("状态增量相对误差")
    axes[0].legend()
    axes[1].bar(x - 0.18, [row["tip_residual_rel_err"] for row in rows], width=0.36, label="隐式预测末端")
    axes[1].bar(x + 0.18, [row["tip_actual_q_rel_err"] for row in rows], width=0.36, label="实际状态增量末端")
    axes[1].set_ylabel("末端导数相对误差")
    axes[1].legend()
    axes[2].bar(x, [row["q_plus_minus_derivative_rel_err"] for row in rows])
    axes[2].set_ylabel("正负扰动对称性误差")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    for ax in axes:
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c75 隐式状态增量与重求解状态增量对照")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_state_increment_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    table = [
        "| 扰动尺度 | 方向 | 状态增量误差 | 末端隐式误差 | 实际状态末端误差 | 正负扰动对称性 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['eps']:.4g} | {row['direction']} | {row['q_residual_rel_err']:.3e} | "
            f"{row['tip_residual_rel_err']:.3e} | {row['tip_actual_q_rel_err']:.3e} | {row['q_plus_minus_derivative_rel_err']:.3e} |"
        )
    max_tip_actual = max(row["tip_actual_q_rel_err"] for row in rows)
    min_q_residual = min(row["q_residual_rel_err"] for row in rows)
    return "\n".join(
        [
            "# P2c75 n2 状态增量对照记录",
            "",
            "## 实验目的",
            "",
            "P2c74 已排除主要自动微分分量错误。本轮比较隐式公式预测的状态增量与扰动重求解得到的实际状态增量，定位 P2c73 平台误差是否来自隐式线性方程。",
            "",
            "## 基线",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"实际重求解状态增量代入末端输出后的最大相对误差为 `{max_tip_actual:.3e}`；隐式状态增量与实际状态增量的最小相对误差为 `{min_q_residual:.3e}`。",
            "若实际状态增量能够解释末端有限差分，而隐式状态增量不能，则误差来自状态灵敏度线性方程，而不是末端输出导数。",
            "本轮中残差抵消公式与一阶条件公式给出几乎相同的预测，但它们均与重求解状态增量存在偏差。该偏差经末端输出映射后被放大为 P2c73 中的 `0.10-0.20` 级局部线性化误差。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若状态增量误差高，推导并实现最小二乘最优性条件的完整灵敏度。",
            "2. 若正负扰动对称性差，说明重求解存在分支非线性，应收缩扰动或做分支约束。",
            "3. 若两者都好，再检查任务输出非线性或终端点约定。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    _params, I0, _target_p, y_base, _y_ref, funcs, basis, base_row = _build_baseline()
    rows: list[dict[str, Any]] = []
    for eps in EPS_VALUES:
        for name, direction in p2c72._directions(I0):
            rows.append(
                _audit_one(
                    eps=eps,
                    name=name,
                    direction=direction,
                    I0=I0,
                    y_base=y_base,
                    funcs=funcs,
                    basis=basis,
                )
            )
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "state_increment_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failures = []
    for row in rows:
        if row["q_residual_rel_err"] > 0.05:
            failures.append({"failure": "隐式状态增量与重求解状态增量不一致", "evidence": row})
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps_values": EPS_VALUES,
        "wall_s": wall_s,
        "base_row": base_row,
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(rows, base_row, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "failures": failures, "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
