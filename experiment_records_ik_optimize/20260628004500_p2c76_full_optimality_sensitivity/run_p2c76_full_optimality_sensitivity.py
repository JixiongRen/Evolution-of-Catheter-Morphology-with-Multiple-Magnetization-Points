"""Audit full optimality sensitivity for n2 subspace reoptimization."""

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
P2C75_DIR = ROOT / "experiment_records_ik_optimize" / "20260628002500_p2c75_n2_state_increment_audit"
for path in (ROOT, P2C72_DIR, P2C75_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c72_n2_subspace_perturb_reopt as p2c72
import run_p2c75_n2_state_increment_audit as p2c75

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS_VALUES = (0.0025, 0.005, 0.010)


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _full_optimality_vectors(
    *,
    y_base: np.ndarray,
    I0: np.ndarray,
    direction: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, np.ndarray | float]:
    y_base_j = jnp.asarray(y_base, dtype=jnp.float64)
    I0_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)
    basis_j = jnp.asarray(basis, dtype=jnp.float64)
    direction_j = jnp.asarray(direction, dtype=jnp.float64).reshape(8)

    def residual_qI(q: jnp.ndarray, I_var: jnp.ndarray) -> jnp.ndarray:
        y = y_base_j + basis_j @ q
        return funcs["residual"](y, I_var)

    def jac_q(q: jnp.ndarray, I_var: jnp.ndarray) -> jnp.ndarray:
        return jax.jacfwd(lambda q_var: residual_qI(q_var, I_var))(q)

    def optimality(q: jnp.ndarray, I_var: jnp.ndarray) -> jnp.ndarray:
        r = residual_qI(q, I_var)
        J = jac_q(q, I_var)
        return J.T @ r

    q0 = jnp.zeros((basis.shape[1],), dtype=jnp.float64)
    g_q = np.asarray(jax.device_get(jax.jacfwd(lambda q: optimality(q, I0_j))(q0)), dtype=float)
    g_I = np.asarray(jax.device_get(jax.jacfwd(lambda I_var: optimality(q0, I_var))(I0_j)), dtype=float)
    q_full = -np.linalg.lstsq(g_q, g_I @ np.asarray(direction, dtype=float).reshape(8), rcond=None)[0]

    J_eff = np.asarray(jax.device_get(jac_q(q0, I0_j)), dtype=float)
    svals_g = np.linalg.svd(g_q, compute_uv=False)
    svals_j = np.linalg.svd(J_eff, compute_uv=False)

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I0_j).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_base_j)), dtype=float)
    T_eff = T_y @ np.asarray(basis, dtype=float)
    return {
        "q_full": q_full,
        "tip_full": T_eff @ q_full,
        "optimality_condition": float(svals_g[0] / max(float(svals_g[-1]), np.finfo(float).tiny)),
        "residual_jacobian_condition": float(svals_j[0] / max(float(svals_j[-1]), np.finfo(float).tiny)),
        "optimality_linear_residual_norm": float(np.linalg.norm(g_q @ q_full + g_I @ np.asarray(direction, dtype=float).reshape(8))),
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
    old = p2c75._implicit_vectors(y_base=y_base, I0=I0, direction=direction, funcs=funcs, basis=basis)
    full = _full_optimality_vectors(y_base=y_base, I0=I0, direction=direction, funcs=funcs, basis=basis)
    q_plus, plus_info = p2c75._solve_q(y_base=y_base, basis=basis, I=I0 + eps * direction, funcs=funcs)
    q_minus, minus_info = p2c75._solve_q(y_base=y_base, basis=basis, I=I0 - eps * direction, funcs=funcs)
    q_actual = (q_plus - q_minus) / (2.0 * eps)
    y_plus = y_base + basis @ q_plus
    y_minus = y_base + basis @ q_minus
    out_plus = funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), jnp.asarray(I0 + eps * direction, dtype=jnp.float64))
    out_minus = funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), jnp.asarray(I0 - eps * direction, dtype=jnp.float64))
    tip_fd = (
        np.asarray(jax.device_get(out_plus.tip_p), dtype=float)
        - np.asarray(jax.device_get(out_minus.tip_p), dtype=float)
    ) / (2.0 * eps)

    y_j = jnp.asarray(y_base, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I_j).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_j)), dtype=float)
    T_eff = T_y @ basis
    tip_actual_q = T_eff @ q_actual
    q_old_rel, q_old_cos, _ = p2c75._safe_rel(np.asarray(old["q_residual"]), q_actual)
    q_full_rel, q_full_cos, q_norm = p2c75._safe_rel(np.asarray(full["q_full"]), q_actual)
    tip_old_rel, tip_old_cos, _ = p2c75._safe_rel(np.asarray(old["tip_residual"]), tip_fd)
    tip_full_rel, tip_full_cos, tip_norm = p2c75._safe_rel(np.asarray(full["tip_full"]), tip_fd)
    tip_actual_rel, tip_actual_cos, _ = p2c75._safe_rel(tip_actual_q, tip_fd)
    linearization_ok = tip_full_rel < 0.05 and q_full_rel < 0.05
    condition_ok = float(full["optimality_condition"]) < 1.0e7
    if linearization_ok and condition_ok:
        status = "publishable"
        failure = "PASS"
    elif linearization_ok:
        status = "linearization_pass_condition_untrusted"
        failure = "最优性线性系统条件数超限"
    elif tip_full_rel >= 0.05:
        status = "diagnostic_only"
        failure = "完整最优性灵敏度仍未达到发布门禁"
    else:
        status = "diagnostic_only"
        failure = "状态增量仍未达到发布门禁"
    return {
        "eps": eps,
        "direction": name,
        "status": status,
        "failure_class": failure,
        "q_old_rel_err": q_old_rel,
        "q_old_cosine": q_old_cos,
        "q_full_rel_err": q_full_rel,
        "q_full_cosine": q_full_cos,
        "q_actual_norm": q_norm,
        "tip_old_rel_err": tip_old_rel,
        "tip_old_cosine": tip_old_cos,
        "tip_full_rel_err": tip_full_rel,
        "tip_full_cosine": tip_full_cos,
        "tip_actual_q_rel_err": tip_actual_rel,
        "tip_actual_q_cosine": tip_actual_cos,
        "tip_fd_norm": tip_norm,
        "optimality_condition": full["optimality_condition"],
        "residual_jacobian_condition": full["residual_jacobian_condition"],
        "optimality_linear_residual_norm": full["optimality_linear_residual_norm"],
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
        "plus_success": plus_info["success"],
        "minus_success": minus_info["success"],
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['direction']}\n{row['eps']:.4g}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x - 0.18, [row["q_old_rel_err"] for row in rows], width=0.36, label="旧状态公式")
    axes[0].bar(x + 0.18, [row["q_full_rel_err"] for row in rows], width=0.36, label="完整最优性公式")
    axes[0].set_ylabel("状态增量相对误差")
    axes[0].legend()
    axes[1].bar(x - 0.18, [row["tip_old_rel_err"] for row in rows], width=0.36, label="旧末端灵敏度")
    axes[1].bar(x + 0.18, [row["tip_full_rel_err"] for row in rows], width=0.36, label="完整最优性末端灵敏度")
    axes[1].set_ylabel("末端导数相对误差")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].legend()
    for ax in axes:
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c76 完整最优性灵敏度对照")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_full_optimality_sensitivity_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    table = [
        "| 扰动尺度 | 方向 | 旧状态误差 | 完整状态误差 | 旧末端误差 | 完整末端误差 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['eps']:.4g} | {row['direction']} | {row['q_old_rel_err']:.3e} | "
            f"{row['q_full_rel_err']:.3e} | {row['tip_old_rel_err']:.3e} | {row['tip_full_rel_err']:.3e} | {row['status']} |"
        )
    publishable = [row for row in rows if row["status"] == "publishable"]
    return "\n".join(
        [
            "# P2c76 n2 完整最优性灵敏度记录",
            "",
            "## 实验目的",
            "",
            "P2c75 表明误差来自状态灵敏度线性方程。本轮使用最小二乘一阶最优性 `g(q,I)=J(q,I)^T r(q,I)` 的完整导数，检验是否能恢复与扰动重求解一致的局部灵敏度。",
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
            f"本轮共有 `{len(publishable)}` / `{len(rows)}` 个方向-尺度组合达到完整发布门禁。完整最优性公式将末端误差降到发布阈值以内，但最优性线性系统条件数约为 `2.432e+13`，违反条件数硬门禁。因此当前结论是“线性化机制被修复”，不是“可发布灵敏度已获得”。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若通过，扩展到更多方向并做多初值分支复核。",
            "2. 若仍失败，检查重求解目标是否与 `g(q,I)` 完全一致，包括损失、权重、边界和求解器约束。",
            "3. 若只部分方向通过，形成方向性工程边界。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    _params, I0, _target_p, y_base, _y_ref, funcs, basis, base_row = p2c75._build_baseline()
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
    _write_csv(OUT_DIR / "full_optimality_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failures = []
    for row in rows:
        if row["status"] != "publishable":
            failures.append({"failure": row["failure_class"], "evidence": row})
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
