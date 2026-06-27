"""Audit n2 subspace finite-difference reoptimization."""

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
P2C69_DIR = ROOT / "experiment_records_ik_optimize" / "20260627230500_p2c69_n2_condition_precondition_audit"
P2C71_DIR = ROOT / "experiment_records_ik_optimize" / "20260627233000_p2c71_n2_effective_subspace_audit"
for path in (ROOT, P2C57_DIR, P2C69_DIR, P2C71_DIR):
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

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c69_n2_condition_precondition_audit as p2c69
import run_p2c71_n2_effective_subspace_audit as p2c71

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
EPS = 0.005
REMOVE_WEAK_DIRECTIONS = 2
MAX_NFEV = 180
PUBLISH_LINEARIZATION_GATE = 0.05
DIAGNOSTIC_LINEARIZATION_GATE = 0.30


def _jsonable(value: Any) -> Any:
    return p2c57._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _safe_rel(pred: np.ndarray, fd: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    fd = np.asarray(fd, dtype=float).reshape(-1)
    fd_norm = float(np.linalg.norm(fd))
    rel = float(np.linalg.norm(pred - fd) / max(fd_norm, 1.0e-12))
    denom = max(float(np.linalg.norm(pred) * fd_norm), 1.0e-12)
    cosine = float(np.dot(pred, fd) / denom)
    return rel, cosine, fd_norm


def _directions(I0: np.ndarray) -> list[tuple[str, np.ndarray]]:
    I0 = np.asarray(I0, dtype=float).reshape(8)
    dirs: list[tuple[str, np.ndarray]] = []
    e = np.zeros_like(I0)
    e[int(np.argmax(np.abs(I0)))] = 1.0
    dirs.append(("max_active_coil", e))
    if np.linalg.norm(I0) > 1.0e-12:
        dirs.append(("current_direction", I0 / np.linalg.norm(I0)))
    rng = np.random.default_rng(72)
    v = rng.normal(size=I0.shape)
    dirs.append(("random_unit", v / np.linalg.norm(v)))
    return dirs


def _eval_physical(
    *,
    label: str,
    params: Any,
    I: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y: np.ndarray,
    funcs: dict[str, Any],
    steady_s: float,
    nfev: int,
    success: bool,
    message: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    out = p2c57.p2c47.output_to_numpy(funcs["output"](y_j, I_j))
    continuity = np.asarray(jax.device_get(funcs["continuity"](y_j, I_j)), dtype=float).reshape(-1)
    svals = np.linalg.svd(J, compute_uv=False)
    solve = {
        "y": np.asarray(y, dtype=float),
        "out": out,
        "success": bool(success),
        "nfev": int(nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "continuity_norm": float(np.linalg.norm(continuity)),
        "message": str(message),
    }
    row = p2c57._row(
        force_scale=p2c69.FORCE_SCALE,
        stage=4,
        pos_scale=p2c69.STAGES[-1][0],
        max_iter=MAX_NFEV,
        params=p2c69.p2c57.p2c47.with_coil_currents(params, jnp.asarray(I, dtype=jnp.float64).reshape(8)),
        target_p=target_p,
        solve=solve,
    )
    row["label"] = label
    row["reg_scale"] = p2c69.REG_SCALE
    row["physical_residual_norm"] = solve["residual_norm"]
    row["message"] = str(message)
    return row, r, J


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
    return y_base + basis @ np.asarray(opt.x, dtype=float), {
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(time.perf_counter() - t0),
        "message": str(opt.message),
        "q_norm": float(np.linalg.norm(opt.x)),
    }


def _implicit_task_sensitivity(
    *,
    y: np.ndarray,
    I: np.ndarray,
    direction: np.ndarray,
    basis: np.ndarray,
    funcs: dict[str, Any],
) -> np.ndarray:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    J_y = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    J_eff = J_y @ basis

    def residual_I(I_var: jnp.ndarray) -> jnp.ndarray:
        return funcs["residual"](y_j, I_var)

    J_I = np.asarray(jax.device_get(jax.jacfwd(residual_I)(I_j)), dtype=float)
    rhs = J_I @ np.asarray(direction, dtype=float).reshape(8)
    dq = -np.linalg.lstsq(J_eff, rhs, rcond=None)[0]
    dy = basis @ dq
    output = funcs["output"]

    def tip_y(y_var: jnp.ndarray) -> jnp.ndarray:
        return output(y_var, I_j).tip_p

    def tip_I(I_var: jnp.ndarray) -> jnp.ndarray:
        return output(y_j, I_var).tip_p

    T_y = np.asarray(jax.device_get(jax.jacfwd(tip_y)(y_j)), dtype=float)
    T_I = np.asarray(jax.device_get(jax.jacfwd(tip_I)(I_j)), dtype=float)
    return T_y @ dy + T_I @ np.asarray(direction, dtype=float).reshape(8)


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [row["direction"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].bar(x, [row["rel_err"] for row in rows])
    axes[0].axhline(0.3, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("线性化相对误差")
    axes[1].bar(x, [row["branch_drift_norm"] for row in rows])
    axes[1].set_ylabel("状态漂移范数")
    axes[2].bar(x, [row["max_gate_violation"] for row in rows])
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("最大门禁倍率")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_yscale("log")
    fig.suptitle("P2c72 n2 子空间扰动重求解")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_subspace_perturb_reopt_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    table = [
        "| 方向 | 状态 | 相对误差 | 余弦 | 有限差分范数 | 分支漂移 | 最大门禁倍率 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['direction']} | {row['status']} | {row['rel_err']:.3e} | {row['cosine']:.3e} | "
            f"{row['fd_norm']:.3e} | {row['branch_drift_norm']:.3e} | {row['max_gate_violation']:.3e} | {row['failure_class']} |"
        )
    publishable = [row for row in rows if row["status"] == "publishable"]
    diagnostic = [row for row in rows if row["status"] == "diagnostic_only"]
    return "\n".join(
        [
            "# P2c72 n2 子空间扰动重求解记录",
            "",
            "## 实验目的",
            "",
            "P2c71 找到 16 维 n2 有效子空间候选。本轮在该子空间内进行 `eps=0.005` 电流扰动重求解，比较子空间隐式导数和重求解有限差分，判断该候选是否可进入更严格的分支一致性审计。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4`。从原近门分支构造移除两个最弱方向后的 16 维子空间。对最大通电线圈方向、原电流方向和一个固定随机方向分别施加 `I±0.005d` 扰动；每次只在该 16 维子空间内重求解。目标点保持不变。",
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
            f"本轮共有 `{len(publishable)}` / `{len(rows)}` 个扰动方向达到发布门禁，`{len(diagnostic)}` / `{len(rows)}` 个方向达到诊断门禁。",
            "",
            f"发布门禁要求局部线性化相对误差不超过 `{PUBLISH_LINEARIZATION_GATE}`；诊断门禁放宽到 `{DIAGNOSTIC_LINEARIZATION_GATE}`，仅用于判断分支是否可跟踪。本轮结果只能说明 16 维子空间在小扰动下可被稳定重求解，不能发布为高质量灵敏度。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 继续做 eps=0.010 与 eps=0.0025 平台审计，确认误差是否随扰动尺度收敛。",
            "2. 若误差不随扰动缩小，则优先检查隐式导数中的任务输出导数和残差电流导数。",
            "3. 若误差随扰动缩小，则保留该工况为诊断候选，继续寻找发布级方向或更小有效子空间。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, I0 = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    recovered = p2c69._recover_near_gate(params, I0, target_p, y_seed)
    y_base = np.asarray(recovered["solve"]["y"], dtype=float)
    y_ref = recovered["y_ref"]
    _args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    I_j = jnp.asarray(I0, dtype=jnp.float64).reshape(8)
    y_j = jnp.asarray(y_base, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    _u, _s, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c71._orthogonal_complement(vt[-REMOVE_WEAK_DIRECTIONS:], J.shape[1])
    base_row = max(recovered["rows"], key=lambda row: row["stage"])
    rows = []
    for name, direction in _directions(np.asarray(I0, dtype=float)):
        pred = _implicit_task_sensitivity(y=y_base, I=I0, direction=direction, basis=basis, funcs=funcs)
        plus_y, plus_info = _solve_q(y_base=y_base, basis=basis, I=np.asarray(I0) + EPS * direction, funcs=funcs)
        minus_y, minus_info = _solve_q(y_base=y_base, basis=basis, I=np.asarray(I0) - EPS * direction, funcs=funcs)
        plus_row, _rp, _jp = _eval_physical(
            label=f"{name}_plus",
            params=params,
            I=np.asarray(I0) + EPS * direction,
            target_p=target_p,
            y_ref=y_ref,
            y=plus_y,
            funcs=funcs,
            steady_s=plus_info["steady_s"],
            nfev=plus_info["nfev"],
            success=plus_info["success"],
            message=plus_info["message"],
        )
        minus_row, _rm, _jm = _eval_physical(
            label=f"{name}_minus",
            params=params,
            I=np.asarray(I0) - EPS * direction,
            target_p=target_p,
            y_ref=y_ref,
            y=minus_y,
            funcs=funcs,
            steady_s=minus_info["steady_s"],
            nfev=minus_info["nfev"],
            success=minus_info["success"],
            message=minus_info["message"],
        )
        plus_tip = np.asarray(plus_row["position_error_m"])
        out_plus = funcs["output"](jnp.asarray(plus_y), jnp.asarray(np.asarray(I0) + EPS * direction))
        out_minus = funcs["output"](jnp.asarray(minus_y), jnp.asarray(np.asarray(I0) - EPS * direction))
        tip_plus = np.asarray(jax.device_get(out_plus.tip_p), dtype=float)
        tip_minus = np.asarray(jax.device_get(out_minus.tip_p), dtype=float)
        fd = (tip_plus - tip_minus) / (2.0 * EPS)
        rel, cosine, fd_norm = _safe_rel(pred, fd)
        branch_drift = max(float(np.linalg.norm(plus_y - y_base)), float(np.linalg.norm(minus_y - y_base)))
        gate_terms = [
            plus_row["stationarity_norm"] / 1.0e-5,
            minus_row["stationarity_norm"] / 1.0e-5,
            plus_row["continuity_norm"] / 1.0e-3,
            minus_row["continuity_norm"] / 1.0e-3,
            plus_row["max_interface_force_jump_over_rigid_force"] / 1.5,
            minus_row["max_interface_force_jump_over_rigid_force"] / 1.5,
        ]
        max_gate = float(max(gate_terms))
        branch_ok = branch_drift <= 5.0
        physical_ok = max_gate <= 1.5 and plus_info["success"] and minus_info["success"]
        diagnostic_ok = rel <= DIAGNOSTIC_LINEARIZATION_GATE and cosine > 0.0 and branch_ok and physical_ok
        publish_ok = rel <= PUBLISH_LINEARIZATION_GATE and cosine > 0.0 and branch_ok and physical_ok
        if publish_ok:
            status = "publishable"
            failure = "PASS"
        elif diagnostic_ok:
            status = "diagnostic_only"
            failure = "局部线性化未达发布门禁"
        elif not branch_ok:
            status = "untrusted"
            failure = "分支漂移"
        elif rel > DIAGNOSTIC_LINEARIZATION_GATE or cosine <= 0.0:
            status = "untrusted"
            failure = "局部线性化失效"
        else:
            status = "untrusted"
            failure = "物理门禁退化"
        row = {
            "direction": name,
            "eps": EPS,
            "status": status,
            "failure_class": failure,
            "publish_ok": publish_ok,
            "diagnostic_ok": diagnostic_ok,
            "publish_linearization_gate": PUBLISH_LINEARIZATION_GATE,
            "diagnostic_linearization_gate": DIAGNOSTIC_LINEARIZATION_GATE,
            "rel_err": rel,
            "cosine": cosine,
            "fd_norm": fd_norm,
            "implicit_norm": float(np.linalg.norm(pred)),
            "branch_drift_norm": branch_drift,
            "max_gate_violation": max_gate,
            "plus_success": plus_info["success"],
            "minus_success": minus_info["success"],
            "plus_nfev": plus_info["nfev"],
            "minus_nfev": minus_info["nfev"],
            "plus_stationarity": plus_row["stationarity_norm"],
            "minus_stationarity": minus_row["stationarity_norm"],
            "plus_continuity": plus_row["continuity_norm"],
            "minus_continuity": minus_row["continuity_norm"],
            "plus_force_ratio": plus_row["max_interface_force_jump_over_rigid_force"],
            "minus_force_ratio": minus_row["max_interface_force_jump_over_rigid_force"],
            "fd_vector": fd.tolist(),
            "implicit_vector": np.asarray(pred, dtype=float).tolist(),
        }
        rows.append(row)
        if status != "publishable":
            with failures.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"failure": failure, "evidence": row}, ensure_ascii=False) + "\n")
    _plot(rows)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "perturb_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "base_row": base_row,
        "removed_weak_directions": REMOVE_WEAK_DIRECTIONS,
        "eps": EPS,
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(rows, base_row, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
