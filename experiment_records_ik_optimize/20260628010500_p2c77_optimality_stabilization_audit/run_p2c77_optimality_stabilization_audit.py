"""Audit stabilization strategies for the n2 optimality sensitivity system."""

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
EPS_VALUES = (0.005, 0.010)
TARGET_CONDITION = 1.0e7
RIDGE_FACTORS = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
TRUNC_CONDITIONS = (1.0e7, 1.0e8, 1.0e9)


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _build_optimality_system(
    *,
    y_base: np.ndarray,
    I0: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, Any]:
    y_base_j = jnp.asarray(y_base, dtype=jnp.float64)
    I0_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)
    basis_j = jnp.asarray(basis, dtype=jnp.float64)

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
    u, s, vt = np.linalg.svd(g_q, full_matrices=False)

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I0_j).tip_p

    t_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_base_j)), dtype=float)
    return {
        "g_q": g_q,
        "g_I": g_I,
        "u": u,
        "s": s,
        "vt": vt,
        "t_eff": t_y @ np.asarray(basis, dtype=float),
        "raw_condition": float(s[0] / max(float(s[-1]), np.finfo(float).tiny)),
    }


def _solve_methods(system: dict[str, Any], rhs: np.ndarray) -> list[dict[str, Any]]:
    u = system["u"]
    s = system["s"]
    vt = system["vt"]
    coeff = u.T @ rhs
    rows: list[dict[str, Any]] = []
    q_raw = -(vt.T @ (coeff / s))
    rows.append(
        {
            "method": "raw_svd",
            "q": q_raw,
            "effective_condition": float(s[0] / max(float(s[-1]), np.finfo(float).tiny)),
            "rank_used": int(s.size),
            "lambda": 0.0,
        }
    )
    for cond in TRUNC_CONDITIONS:
        threshold = s[0] / cond
        keep = s >= threshold
        q = -(vt[keep].T @ (coeff[keep] / s[keep]))
        eff_cond = float(s[keep][0] / s[keep][-1]) if np.any(keep) else float("inf")
        rows.append(
            {
                "method": f"truncated_cond_{cond:.0e}",
                "q": q,
                "effective_condition": eff_cond,
                "rank_used": int(np.count_nonzero(keep)),
                "lambda": 0.0,
            }
        )
    for factor in RIDGE_FACTORS:
        lam = float((s[0] ** 2) * factor)
        filt = s / (s * s + lam)
        q = -(vt.T @ (filt * coeff))
        eff_cond = float((s[0] * s[0] + lam) / (s[-1] * s[-1] + lam))
        rows.append(
            {
                "method": f"ridge_{factor:.0e}",
                "q": q,
                "effective_condition": eff_cond,
                "rank_used": int(s.size),
                "lambda": lam,
            }
        )
    return rows


def _audit_case(
    *,
    eps: float,
    direction_name: str,
    direction: np.ndarray,
    I0: np.ndarray,
    y_base: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
    system: dict[str, Any],
) -> list[dict[str, Any]]:
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
    rhs = system["g_I"] @ np.asarray(direction, dtype=float).reshape(8)
    rows = []
    for method in _solve_methods(system, rhs):
        q = np.asarray(method["q"], dtype=float)
        tip = system["t_eff"] @ q
        q_rel, q_cos, q_norm = p2c75._safe_rel(q, q_actual)
        tip_rel, tip_cos, tip_norm = p2c75._safe_rel(tip, tip_fd)
        condition_ok = method["effective_condition"] < TARGET_CONDITION
        linear_ok = q_rel < 0.05 and tip_rel < 0.05
        if condition_ok and linear_ok:
            status = "publishable"
            failure = "PASS"
        elif linear_ok:
            status = "linearization_pass_condition_untrusted"
            failure = "稳定化条件数仍未过门"
        else:
            status = "diagnostic_only"
            failure = "稳定化破坏局部线性化"
        rows.append(
            {
                "eps": eps,
                "direction": direction_name,
                "method": method["method"],
                "status": status,
                "failure_class": failure,
                "effective_condition": method["effective_condition"],
                "raw_condition": system["raw_condition"],
                "rank_used": method["rank_used"],
                "lambda": method["lambda"],
                "q_rel_err": q_rel,
                "q_cosine": q_cos,
                "q_actual_norm": q_norm,
                "tip_rel_err": tip_rel,
                "tip_cosine": tip_cos,
                "tip_fd_norm": tip_norm,
                "plus_nfev": plus_info["nfev"],
                "minus_nfev": minus_info["nfev"],
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    best_by_method = []
    for method in methods:
        part = [row for row in rows if row["method"] == method]
        best_by_method.append(
            {
                "method": method,
                "max_tip": max(row["tip_rel_err"] for row in part),
                "max_q": max(row["q_rel_err"] for row in part),
                "condition": max(row["effective_condition"] for row in part),
            }
        )
    x = np.arange(len(best_by_method))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(x, [row["max_tip"] for row in best_by_method])
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("最大末端误差")
    axes[1].bar(x, [row["max_q"] for row in best_by_method])
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("最大状态误差")
    axes[2].bar(x, [row["condition"] for row in best_by_method])
    axes[2].axhline(TARGET_CONDITION, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("有效条件数")
    axes[2].set_xticks(x, [row["method"] for row in best_by_method], rotation=35, ha="right")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c77 最优性线性系统稳定化审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_optimality_stabilization_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    table = [
        "| 方法 | 最大末端误差 | 最大状态误差 | 最大有效条件数 | 通过组合数 | 结论 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in methods:
        part = [row for row in rows if row["method"] == method]
        max_tip = max(row["tip_rel_err"] for row in part)
        max_q = max(row["q_rel_err"] for row in part)
        max_cond = max(row["effective_condition"] for row in part)
        passed = sum(row["status"] == "publishable" for row in part)
        verdict = "可发布候选" if passed == len(part) else ("部分可用" if passed else "不可发布")
        table.append(f"| {method} | `{max_tip:.3e}` | `{max_q:.3e}` | `{max_cond:.3e}` | `{passed}/{len(part)}` | {verdict} |")
    publishable = [row for row in rows if row["status"] == "publishable"]
    return "\n".join(
        [
            "# P2c77 n2 最优性线性系统稳定化审计记录",
            "",
            "## 实验目的",
            "",
            "P2c76 修复了局部线性化机制，但完整最优性线性系统条件数约 `2.432e+13`。本轮比较奇异值截断和 Tikhonov 阻尼，判断能否在不破坏状态增量和末端导数一致性的前提下降低有效条件数。",
            "",
            "## 基线",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 方法汇总",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"本轮共有 `{len(publishable)}` / `{len(rows)}` 个方法-方向-尺度组合同时通过条件数和线性化门禁。只有当某个方法在全部方向和尺度上通过，才能作为发布级稳定化方案。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在全通过方法，做多初值分支复核。",
            "2. 若只部分通过，收缩发布任务维度或方向集合。",
            "3. 若全部失败，归因为最优性系统数值病态，转向重参数化或更小有效子空间。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    _params, I0, _target_p, y_base, _y_ref, funcs, basis, base_row = p2c75._build_baseline()
    system = _build_optimality_system(y_base=y_base, I0=I0, funcs=funcs, basis=basis)
    rows: list[dict[str, Any]] = []
    for eps in EPS_VALUES:
        for direction_name, direction in p2c72._directions(I0):
            rows.extend(
                _audit_case(
                    eps=eps,
                    direction_name=direction_name,
                    direction=direction,
                    I0=I0,
                    y_base=y_base,
                    funcs=funcs,
                    basis=basis,
                    system=system,
                )
            )
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "stabilization_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failures = [{"failure": row["failure_class"], "evidence": row} for row in rows if row["status"] != "publishable"]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps_values": EPS_VALUES,
        "target_condition": TARGET_CONDITION,
        "wall_s": wall_s,
        "base_row": base_row,
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(rows, base_row, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "rows": rows, "failures": len(failures)}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
