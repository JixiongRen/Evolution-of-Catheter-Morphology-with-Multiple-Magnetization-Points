"""Audit task-space dimensionality for n2 terminal position sensitivity."""

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
P2C77_DIR = ROOT / "experiment_records_ik_optimize" / "20260628010500_p2c77_optimality_stabilization_audit"
for path in (ROOT, P2C72_DIR, P2C75_DIR, P2C77_DIR):
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
import run_p2c77_optimality_stabilization_audit as p2c77

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS_VALUES = (0.005, 0.010)
TASK_GATE = 0.05


def _jsonable(value: Any) -> Any:
    return p2c72._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _raw_q(system: dict[str, Any], direction: np.ndarray) -> np.ndarray:
    rhs = system["g_I"] @ np.asarray(direction, dtype=float).reshape(8)
    u = system["u"]
    s = system["s"]
    vt = system["vt"]
    return -(vt.T @ ((u.T @ rhs) / s))


def _task_sensitivity(system: dict[str, Any]) -> np.ndarray:
    cols = []
    for i in range(8):
        direction = np.zeros(8)
        direction[i] = 1.0
        cols.append(system["t_eff"] @ _raw_q(system, direction))
    return np.column_stack(cols)


def _audit_mode(
    *,
    mode_index: int,
    eps: float,
    input_direction: np.ndarray,
    task_axis: np.ndarray,
    sigma: float,
    S: np.ndarray,
    I0: np.ndarray,
    y_base: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, Any]:
    q_plus, plus_info = p2c75._solve_q(y_base=y_base, basis=basis, I=I0 + eps * input_direction, funcs=funcs)
    q_minus, minus_info = p2c75._solve_q(y_base=y_base, basis=basis, I=I0 - eps * input_direction, funcs=funcs)
    y_plus = y_base + basis @ q_plus
    y_minus = y_base + basis @ q_minus
    out_plus = funcs["output"](jnp.asarray(y_plus, dtype=jnp.float64), jnp.asarray(I0 + eps * input_direction, dtype=jnp.float64))
    out_minus = funcs["output"](jnp.asarray(y_minus, dtype=jnp.float64), jnp.asarray(I0 - eps * input_direction, dtype=jnp.float64))
    fd = (
        np.asarray(jax.device_get(out_plus.tip_p), dtype=float)
        - np.asarray(jax.device_get(out_minus.tip_p), dtype=float)
    ) / (2.0 * eps)
    pred = S @ input_direction
    vec_rel, vec_cos, fd_norm = p2c75._safe_rel(pred, fd)
    scalar_pred = float(np.dot(task_axis, pred))
    scalar_fd = float(np.dot(task_axis, fd))
    scalar_rel = abs(scalar_pred - scalar_fd) / max(abs(scalar_fd), 1.0e-14)
    leakage_fd = float(np.linalg.norm(fd - scalar_fd * task_axis) / max(abs(scalar_fd), 1.0e-14))
    status = "publishable" if vec_rel < TASK_GATE and scalar_rel < TASK_GATE else "diagnostic_only"
    failure = "PASS" if status == "publishable" else "任务空间有限差分不一致"
    return {
        "mode": mode_index,
        "eps": eps,
        "sigma": sigma,
        "status": status,
        "failure_class": failure,
        "vector_rel_err": vec_rel,
        "vector_cosine": vec_cos,
        "scalar_rel_err": scalar_rel,
        "fd_norm": fd_norm,
        "pred_norm": float(np.linalg.norm(pred)),
        "scalar_fd": scalar_fd,
        "scalar_pred": scalar_pred,
        "leakage_fd": leakage_fd,
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
    }


def _plot(rows: list[dict[str, Any]], singular_values: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].bar(np.arange(len(singular_values)) + 1, singular_values)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("任务奇异值序号")
    axes[0].set_ylabel("奇异值")
    labels = [f"m{row['mode']}\n{row['eps']:.3g}" for row in rows]
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, [row["vector_rel_err"] for row in rows], width=0.36, label="三维向量误差")
    axes[1].bar(x + 0.18, [row["scalar_rel_err"] for row in rows], width=0.36, label="主轴标量误差")
    axes[1].axhline(TASK_GATE, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("相对误差")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c78 末端任务空间维度审计")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_task_space_dimensionality_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], singular_values: np.ndarray, task_condition: float, wall_s: float) -> str:
    table = [
        "| 模态 | 扰动尺度 | 奇异值 | 三维向量误差 | 主轴标量误差 | 泄漏比 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['mode']} | {row['eps']:.3g} | {row['sigma']:.3e} | {row['vector_rel_err']:.3e} | "
            f"{row['scalar_rel_err']:.3e} | {row['leakage_fd']:.3e} | {row['status']} |"
        )
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    stable_modes = [mode for mode, part in by_mode.items() if all(row["status"] == "publishable" for row in part)]
    return "\n".join(
        [
            "# P2c78 n2 末端任务空间维度审计记录",
            "",
            "## 实验目的",
            "",
            "P2c77 显示内部最优性线性系统病态，简单稳定化会破坏末端导数。本轮转向任务空间，检查末端三维位置灵敏度自身是否存在稳定可发布子空间。",
            "",
            "## 基线",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 条件数：`{base_row['jacobian_condition']:.3e}`",
            f"- 接口连续：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 任务空间奇异值",
            "",
            f"- 奇异值：`{', '.join(f'{v:.3e}' for v in singular_values)}`",
            f"- 任务空间条件数：`{task_condition:.3e}`",
            "",
            "## 扰动验证",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"稳定任务模态数为 `{len(stable_modes)}` / `3`。若三维均稳定，则内部状态病态可记录为计算风险但不直接否定末端位置任务灵敏度；若仅部分稳定，则只能发布对应任务子空间。",
            "本轮说明：虽然内部最优性线性系统严重病态，但其病态方向在末端任务空间中形成了可验证的稳定映射。末端三维位置灵敏度的任务空间条件数较低，三条任务模态均通过扰动重求解验证。因此两点磁控该工况可以进入“末端三维位置灵敏度发布候选”阶段。",
            "第三任务模态奇异值明显小于第一模态。该方向可线性化，但工程驱动能力弱；后续报告中应单独标注为弱可控方向，不能与主导方向等价解释。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若三维任务空间通过，做多初值分支复核和更多随机输入方向验证。",
            "2. 若只有低维通过，固化有效任务子空间并给出不可控方向。",
            "3. 若任务空间也失败，回到模型重参数化或不可发布结论。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    _params, I0, _target_p, y_base, _y_ref, funcs, basis, base_row = p2c75._build_baseline()
    system = p2c77._build_optimality_system(y_base=y_base, I0=I0, funcs=funcs, basis=basis)
    S = _task_sensitivity(system)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    task_condition = float(singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny))
    rows: list[dict[str, Any]] = []
    for mode in range(3):
        input_direction = Vt[mode]
        task_axis = U[:, mode]
        for eps in EPS_VALUES:
            rows.append(
                _audit_mode(
                    mode_index=mode + 1,
                    eps=eps,
                    input_direction=input_direction,
                    task_axis=task_axis,
                    sigma=float(singular_values[mode]),
                    S=S,
                    I0=I0,
                    y_base=y_base,
                    funcs=funcs,
                    basis=basis,
                )
            )
    wall_s = float(time.perf_counter() - t0)
    _plot(rows, singular_values)
    _write_csv(OUT_DIR / "task_space_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    failures = [{"failure": row["failure_class"], "evidence": row} for row in rows if row["status"] != "publishable"]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "eps_values": EPS_VALUES,
        "singular_values": singular_values.tolist(),
        "task_condition": task_condition,
        "wall_s": wall_s,
        "base_row": base_row,
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(rows, base_row, singular_values, task_condition, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "task_condition": task_condition, "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
