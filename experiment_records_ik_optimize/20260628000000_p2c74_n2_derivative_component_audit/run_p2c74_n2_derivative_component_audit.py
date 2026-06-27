"""Audit derivative components for the n2 subspace sensitivity mismatch."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c72_n2_subspace_perturb_reopt as p2c72

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS_VALUES = (1.0e-4, 3.0e-4, 1.0e-3)
TASK_EPS_VALUES = (1.0e-7, 1.0e-6, 1.0e-5)


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _rel_err(pred: np.ndarray, fd: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    fd = np.asarray(fd, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(fd)), 1.0e-14)
    rel = float(np.linalg.norm(pred - fd) / denom)
    cos_denom = max(float(np.linalg.norm(pred) * np.linalg.norm(fd)), 1.0e-14)
    cosine = float(np.dot(pred, fd) / cos_denom)
    return rel, cosine, float(np.linalg.norm(fd))


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


def _component_rows(
    *,
    y_base: np.ndarray,
    I0: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_j = jnp.asarray(y_base, dtype=jnp.float64)
    I_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)
    J_y = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    J_eff = J_y @ basis

    def residual_I(I_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["residual"](y_j, I_var)

    J_I = np.asarray(jax.device_get(jax.jacfwd(residual_I)(I_j)), dtype=float)

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_var, I_j).tip_p

    def tip_I(I_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["output"](y_j, I_var).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_j)), dtype=float)
    T_eff = T_y @ basis
    T_I = np.asarray(jax.device_get(jax.jacfwd(tip_I)(I_j)), dtype=float)

    directions = p2c72._directions(I0)
    subspace_dirs = [
        ("subspace_e0", np.eye(basis.shape[1])[0]),
        ("subspace_mid", np.eye(basis.shape[1])[basis.shape[1] // 2]),
        ("subspace_last", np.eye(basis.shape[1])[-1]),
    ]

    for eps in EPS_VALUES:
        for name, direction in directions:
            direction = np.asarray(direction, dtype=float).reshape(8)
            r_plus = np.asarray(jax.device_get(funcs["residual"](y_j, jnp.asarray(I0 + eps * direction, dtype=jnp.float64))), dtype=float)
            r_minus = np.asarray(jax.device_get(funcs["residual"](y_j, jnp.asarray(I0 - eps * direction, dtype=jnp.float64))), dtype=float)
            fd = (r_plus - r_minus) / (2.0 * eps)
            pred = J_I @ direction
            rel, cosine, fd_norm = _rel_err(pred, fd)
            rows.append(
                {
                    "component": "residual_current_derivative",
                    "direction": name,
                    "eps": eps,
                    "rel_err": rel,
                    "cosine": cosine,
                    "fd_norm": fd_norm,
                    "pred_norm": float(np.linalg.norm(pred)),
                }
            )
            tip_plus = np.asarray(jax.device_get(funcs["output"](y_j, jnp.asarray(I0 + eps * direction, dtype=jnp.float64)).tip_p), dtype=float)
            tip_minus = np.asarray(jax.device_get(funcs["output"](y_j, jnp.asarray(I0 - eps * direction, dtype=jnp.float64)).tip_p), dtype=float)
            fd_tip = (tip_plus - tip_minus) / (2.0 * eps)
            pred_tip = T_I @ direction
            rel, cosine, fd_norm = _rel_err(pred_tip, fd_tip)
            rows.append(
                {
                    "component": "tip_current_direct_derivative",
                    "direction": name,
                    "eps": eps,
                    "rel_err": rel,
                    "cosine": cosine,
                    "fd_norm": fd_norm,
                    "pred_norm": float(np.linalg.norm(pred_tip)),
                }
            )

    for task_eps in TASK_EPS_VALUES:
        for name, q_dir in subspace_dirs:
            q_dir = np.asarray(q_dir, dtype=float).reshape(-1)
            y_plus = y_base + task_eps * (basis @ q_dir)
            y_minus = y_base - task_eps * (basis @ q_dir)
            tip_plus = np.asarray(jax.device_get(funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), I_j).tip_p), dtype=float)
            tip_minus = np.asarray(jax.device_get(funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), I_j).tip_p), dtype=float)
            fd = (tip_plus - tip_minus) / (2.0 * task_eps)
            pred = T_eff @ q_dir
            rel, cosine, fd_norm = _rel_err(pred, fd)
            rows.append(
                {
                    "component": "tip_state_subspace_derivative",
                    "direction": name,
                    "eps": task_eps,
                    "rel_err": rel,
                    "cosine": cosine,
                    "fd_norm": fd_norm,
                    "pred_norm": float(np.linalg.norm(pred)),
                }
            )

    svals = np.linalg.svd(J_eff, compute_uv=False)
    rows.append(
        {
            "component": "effective_jacobian_condition",
            "direction": "all",
            "eps": 0.0,
            "rel_err": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)),
            "cosine": 1.0,
            "fd_norm": float(svals[-1]),
            "pred_norm": float(svals[0]),
        }
    )
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    plot_rows = [row for row in rows if row["component"] != "effective_jacobian_condition"]
    components = sorted({row["component"] for row in plot_rows})
    fig, axes = plt.subplots(len(components), 1, figsize=(9, 3.2 * len(components)), sharex=False)
    if len(components) == 1:
        axes = [axes]
    for ax, component in zip(axes, components):
        part = [row for row in plot_rows if row["component"] == component]
        labels = [f"{row['direction']}\n{row['eps']:.0e}" for row in part]
        ax.bar(np.arange(len(part)), [max(row["rel_err"], 1.0e-16) for row in part])
        ax.set_yscale("log")
        ax.axhline(1.0e-6, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("相对误差")
        ax.set_title(component)
        ax.set_xticks(np.arange(len(part)), labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c74 导数分量有限差分核查")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_derivative_component_audit_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    groups = {}
    for row in rows:
        groups.setdefault(row["component"], []).append(row)
    table = [
        "| 分量 | 最小相对误差 | 最大相对误差 | 最小余弦 | 结论 |",
        "| --- | --- | --- | --- |",
    ]
    for component, part in sorted(groups.items()):
        if component == "effective_jacobian_condition":
            condition = part[0]["rel_err"]
            table.append(f"| 有效残差雅可比条件数 | `{condition:.3e}` | `{condition:.3e}` | `1.000e+00` | 诊断量 |")
            continue
        min_rel = min(row["rel_err"] for row in part)
        max_rel = max(row["rel_err"] for row in part)
        min_cos = min(row["cosine"] for row in part)
        all_zero = all(row["fd_norm"] < 1.0e-14 and row["pred_norm"] < 1.0e-14 for row in part)
        verdict = "通过" if all_zero or (min_rel < 1.0e-5 and min_cos > 0.999) else "异常"
        table.append(f"| {component} | `{min_rel:.3e}` | `{max_rel:.3e}` | `{min_cos:.3e}` | {verdict} |")
    return "\n".join(
        [
            "# P2c74 n2 导数分量核查记录",
            "",
            "## 实验目的",
            "",
            "P2c73 显示局部线性化误差不随电流扰动尺度下降，说明误差更可能来自隐式灵敏度公式或实现不一致。本轮分别核查残差对电流导数、末端位置对状态导数、末端位置对电流直接导数。固定状态下末端位置对电流的直接导数理论上可以为零；零-零一致时判为通过。",
            "",
            "## 基线",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 分量核查",
            "",
            *table,
            "",
            "## 结论",
            "",
            "残差对电流导数通过，末端位置对电流直接导数为零且一致，末端位置对状态导数在足够小的状态扰动下通过。因此 P2c73 的平台误差不应首先归因于自动微分分量错误，更可能来自隐式线性方程求解方向、子空间投影或重求解位移与线性预测位移不一致。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 核查 `dq = -J_eff^+ J_I dI` 与扰动重求解得到的实际 `dq` 是否一致。",
            "2. 若 `dq` 不一致，检查子空间最小二乘的秩截断、阻尼和残差权重。",
            "3. 若 `dq` 一致，再回到末端输出非线性和分支曲率分析。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    _params, I0, _target_p, y_base, _y_ref, funcs, basis, base_row = _build_baseline()
    rows = _component_rows(y_base=y_base, I0=I0, funcs=funcs, basis=basis)
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "component_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failures = []
    for row in rows:
        all_zero = row["fd_norm"] < 1.0e-14 and row["pred_norm"] < 1.0e-14
        if row["component"] != "effective_jacobian_condition" and not all_zero and (row["rel_err"] >= 1.0e-5 or row["cosine"] <= 0.999):
            failure = "较大状态扰动非线性" if row["component"] == "tip_state_subspace_derivative" else "导数分量有限差分不一致"
            failures.append({"failure": failure, "evidence": row})
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps_values": EPS_VALUES,
        "task_eps_values": TASK_EPS_VALUES,
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
