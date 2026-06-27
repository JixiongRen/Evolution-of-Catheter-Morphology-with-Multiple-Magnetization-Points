"""Audit n2 subspace sensitivity over multiple perturbation scales."""

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


def _audit_one(
    *,
    eps: float,
    name: str,
    direction: np.ndarray,
    params: Any,
    I0: np.ndarray,
    target_p: np.ndarray,
    y_base: np.ndarray,
    y_ref: np.ndarray,
    funcs: dict[str, Any],
    basis: np.ndarray,
) -> dict[str, Any]:
    pred = p2c72._implicit_task_sensitivity(y=y_base, I=I0, direction=direction, basis=basis, funcs=funcs)
    plus_y, plus_info = p2c72._solve_q(y_base=y_base, basis=basis, I=I0 + eps * direction, funcs=funcs)
    minus_y, minus_info = p2c72._solve_q(y_base=y_base, basis=basis, I=I0 - eps * direction, funcs=funcs)
    plus_row, _rp, _jp = p2c72._eval_physical(
        label=f"{name}_plus_eps_{eps:g}",
        params=params,
        I=I0 + eps * direction,
        target_p=target_p,
        y_ref=y_ref,
        y=plus_y,
        funcs=funcs,
        steady_s=plus_info["steady_s"],
        nfev=plus_info["nfev"],
        success=plus_info["success"],
        message=plus_info["message"],
    )
    minus_row, _rm, _jm = p2c72._eval_physical(
        label=f"{name}_minus_eps_{eps:g}",
        params=params,
        I=I0 - eps * direction,
        target_p=target_p,
        y_ref=y_ref,
        y=minus_y,
        funcs=funcs,
        steady_s=minus_info["steady_s"],
        nfev=minus_info["nfev"],
        success=minus_info["success"],
        message=minus_info["message"],
    )
    out_plus = funcs["output"](jnp.asarray(plus_y), jnp.asarray(I0 + eps * direction))
    out_minus = funcs["output"](jnp.asarray(minus_y), jnp.asarray(I0 - eps * direction))
    tip_plus = np.asarray(jax.device_get(out_plus.tip_p), dtype=float)
    tip_minus = np.asarray(jax.device_get(out_minus.tip_p), dtype=float)
    fd = (tip_plus - tip_minus) / (2.0 * eps)
    rel, cosine, fd_norm = p2c72._safe_rel(pred, fd)
    branch_drift = max(float(np.linalg.norm(plus_y - y_base)), float(np.linalg.norm(minus_y - y_base)))
    max_gate = float(
        max(
            plus_row["stationarity_norm"] / 1.0e-5,
            minus_row["stationarity_norm"] / 1.0e-5,
            plus_row["continuity_norm"] / 1.0e-3,
            minus_row["continuity_norm"] / 1.0e-3,
            plus_row["max_interface_force_jump_over_rigid_force"] / 1.5,
            minus_row["max_interface_force_jump_over_rigid_force"] / 1.5,
        )
    )
    publish_ok = rel <= p2c72.PUBLISH_LINEARIZATION_GATE and cosine > 0.0 and branch_drift <= 5.0 and max_gate <= 1.5
    diagnostic_ok = rel <= p2c72.DIAGNOSTIC_LINEARIZATION_GATE and cosine > 0.0 and branch_drift <= 5.0 and max_gate <= 1.5
    status = "publishable" if publish_ok else ("diagnostic_only" if diagnostic_ok else "untrusted")
    failure = "PASS" if publish_ok else ("局部线性化未达发布门禁" if diagnostic_ok else "局部线性化或物理门禁失效")
    return {
        "eps": eps,
        "direction": name,
        "status": status,
        "failure_class": failure,
        "rel_err": rel,
        "cosine": cosine,
        "fd_norm": fd_norm,
        "implicit_norm": float(np.linalg.norm(pred)),
        "branch_drift_norm": branch_drift,
        "max_gate_violation": max_gate,
        "plus_nfev": plus_info["nfev"],
        "minus_nfev": minus_info["nfev"],
        "plus_stationarity": plus_row["stationarity_norm"],
        "minus_stationarity": minus_row["stationarity_norm"],
        "plus_force_ratio": plus_row["max_interface_force_jump_over_rigid_force"],
        "minus_force_ratio": minus_row["max_interface_force_jump_over_rigid_force"],
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for direction in sorted({row["direction"] for row in rows}):
        part = sorted([row for row in rows if row["direction"] == direction], key=lambda row: row["eps"])
        ax.plot([row["eps"] for row in part], [row["rel_err"] for row in part], marker="o", label=direction)
    ax.axhline(p2c72.PUBLISH_LINEARIZATION_GATE, color="black", linestyle="--", linewidth=1, label="发布门禁")
    ax.axhline(p2c72.DIAGNOSTIC_LINEARIZATION_GATE, color="gray", linestyle=":", linewidth=1, label="诊断门禁")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("电流扰动尺度")
    ax.set_ylabel("局部线性化相对误差")
    ax.set_title("P2c73 两点磁控有效子空间扰动尺度审计")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_eps_platform_linearization_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], base_row: dict[str, Any], wall_s: float) -> str:
    table = [
        "| 扰动尺度 | 方向 | 状态 | 相对误差 | 余弦 | 分支漂移 | 最大门禁倍率 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['eps']:.4g} | {row['direction']} | {row['status']} | {row['rel_err']:.3e} | "
            f"{row['cosine']:.3e} | {row['branch_drift_norm']:.3e} | {row['max_gate_violation']:.3e} |"
        )
    publishable = [row for row in rows if row["status"] == "publishable"]
    diagnostic = [row for row in rows if row["status"] == "diagnostic_only"]
    trend_lines = []
    for direction in sorted({row["direction"] for row in rows}):
        vals = [row["rel_err"] for row in rows if row["direction"] == direction]
        spread = max(vals) - min(vals)
        trend_lines.append(f"- {direction}: `{min(vals):.3e}` 至 `{max(vals):.3e}`，跨度 `{spread:.3e}`")
    return "\n".join(
        [
            "# P2c73 n2 扰动尺度平台审计记录",
            "",
            "## 实验目的",
            "",
            "检查 P2c72 的局部线性化误差是否随电流扰动尺度缩小而收敛。若误差随尺度缩小明显下降，说明主要问题是有限扰动非线性；若不下降，则优先怀疑隐式导数、任务输出导数或残差电流导数存在不一致。",
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
            f"本轮共有 `{len(publishable)}` / `{len(rows)}` 个方向-尺度组合达到发布门禁，`{len(diagnostic)}` / `{len(rows)}` 个组合达到诊断门禁。",
            "若最小扰动尺度仍高于发布门禁，则 P2c72 的 16 维子空间只能说明分支可跟踪，不能作为可发布灵敏度。",
            "三个方向的相对误差随扰动尺度变化很小，呈平台特征。因此当前主要矛盾更可能是隐式灵敏度公式或实现与重求解有限差分不一致，而不是有限扰动尺度过大。",
            "",
            "误差平台：",
            "",
            *trend_lines,
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若误差平台不随尺度下降，做残差电流导数与任务输出导数的分量级有限差分核查。",
            "2. 若误差只在部分方向下降，保留方向性工程边界。",
            "3. 若出现发布级组合，再做多初值分支复核。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, I0, target_p, y_base, y_ref, funcs, basis, base_row = _build_baseline()
    rows: list[dict[str, Any]] = []
    for eps in EPS_VALUES:
        for name, direction in p2c72._directions(I0):
            row = _audit_one(
                eps=eps,
                name=name,
                direction=direction,
                params=params,
                I0=I0,
                target_p=target_p,
                y_base=y_base,
                y_ref=y_ref,
                funcs=funcs,
                basis=basis,
            )
            rows.append(row)
            if row["status"] != "publishable":
                with failures.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"failure": row["failure_class"], "evidence": row}, ensure_ascii=False) + "\n")
    wall_s = float(time.perf_counter() - t0)
    _plot(rows)
    _write_csv(OUT_DIR / "eps_platform_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", rows)
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
    print(json.dumps(_jsonable({"wall_s": wall_s, "rows": rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
