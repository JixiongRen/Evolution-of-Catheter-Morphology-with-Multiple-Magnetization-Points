"""Probe low-dimensional load-transfer shooting for the n2 position task."""

from __future__ import annotations

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
for path in (ROOT, P2C57_DIR):
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
from inverse_kinematics.ik_terminal_shooting_jax import _advance_rigid, _integrate_flex, initial_state, normalize_state

import run_p2c57_force_continuity_weight_repair as p2c57

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
POSITION_SCALES = (20.0, 40.0, 60.0)
BOUNDARY_SCALE = 0.25
SLACK_REG_SCALES = (0.0, 1.0e-3, 1.0e-2)
MAX_ITER = 180
STEPS_PER_FLEX = 4
LENGTH_M = 0.11


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _condition_number(J: np.ndarray) -> float:
    svals = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    if not svals.size:
        return float("inf")
    smallest = float(svals[-1])
    if smallest <= np.finfo(float).tiny:
        return float("inf")
    return float(svals[0] / smallest)


def _split_u(u: jnp.ndarray, n_interfaces: int, mode: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u = jnp.asarray(u, dtype=jnp.float64).reshape(-1)
    base = u[:6]
    if mode == "base_6d":
        zeros = jnp.zeros((n_interfaces,), dtype=jnp.float64)
        return base, zeros, zeros
    if mode == "load_scaled":
        slack = u[6:].reshape((n_interfaces, 2))
        return base, slack[:, 0], slack[:, 1]
    raise ValueError(f"Unknown mode: {mode}")


def _forward_lowdim(params: Any, u: jnp.ndarray, *, mode: str):
    n_interfaces = max(len(params.flex) - 1, 0)
    base, force_lam, torque_lam = _split_u(u, n_interfaces, mode)
    x = initial_state(params, base)
    centerline_parts = [x[0:3].reshape(1, 3)]
    blocks = []
    jumps = []
    for i in range(len(params.flex)):
        blocks.append(jnp.concatenate([x[7:10] / float(params.scales.F_ref), x[10:13] / float(params.scales.M_ref)], axis=0))
        x, flex_points = _integrate_flex(params, x, i, STEPS_PER_FLEX)
        centerline_parts.append(flex_points)
        f_rigid, tau_rigid = p2c57.p2c47._rigid_wrench(params, x, i)
        x, _mid_p, _mid_q = _advance_rigid(params, x, i)
        if i < n_interfaces:
            jump_f = force_lam[i] * f_rigid
            jump_tau = torque_lam[i] * tau_rigid
            x = x.at[7:10].add(jump_f)
            x = x.at[10:13].add(jump_tau)
            x = normalize_state(x)
            jumps.append(
                jnp.concatenate(
                    [
                        jump_f / jnp.maximum(jnp.linalg.norm(f_rigid), 1.0e-30),
                        jump_tau / jnp.maximum(jnp.linalg.norm(tau_rigid), 1.0e-30),
                    ],
                    axis=0,
                )
            )
        centerline_parts.append(x[0:3].reshape(1, 3))
    boundary = jnp.concatenate(
        [
            x[7:10] / jnp.maximum(float(params.scales.F_ref), 1.0e-12),
            x[10:13] / jnp.maximum(float(params.scales.M_ref), 1.0e-12),
        ],
        axis=0,
    )
    if jumps:
        jump_arr = jnp.stack(jumps, axis=0)
    else:
        jump_arr = jnp.zeros((0, 6), dtype=jnp.float64)
    return {
        "centerline": jnp.concatenate(centerline_parts, axis=0),
        "tip_p": x[0:3],
        "tip_q": x[3:7],
        "boundary": boundary,
        "blocks": jnp.stack(blocks, axis=0),
        "normalized_jumps": jump_arr,
    }


def _make_functions(params: Any, target_p: np.ndarray, *, mode: str, position_scale: float, slack_reg_scale: float):
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def output_u(u: jnp.ndarray):
        return _forward_lowdim(params, u, mode=mode)

    def residual_u(u: jnp.ndarray) -> jnp.ndarray:
        out = output_u(u)
        _base, force_lam, torque_lam = _split_u(u, max(len(params.flex) - 1, 0), mode)
        r_p = float(position_scale) * (out["tip_p"] - target_p_j)
        r_b = float(BOUNDARY_SCALE) * out["boundary"]
        r_lam = float(slack_reg_scale) * jnp.concatenate([force_lam.reshape(-1), torque_lam.reshape(-1)], axis=0)
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_lam.reshape(-1)], axis=0)

    return {
        "output": jax.jit(output_u),
        "residual": jax.jit(residual_u),
        "jacobian": jax.jit(jax.jacfwd(residual_u)),
    }


def _solve(u0: np.ndarray, funcs: dict[str, Any]) -> dict[str, Any]:
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(u0, dtype=jnp.float64)).block_until_ready()
    t0 = time.perf_counter()
    opt = least_squares(
        lambda uu: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(uu, dtype=jnp.float64))), dtype=float),
        u0,
        jac=lambda uu: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(uu, dtype=jnp.float64))), dtype=float),
        method="trf",
        x_scale=np.ones_like(u0),
        loss="linear",
        max_nfev=MAX_ITER,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = float(time.perf_counter() - t0)
    u = np.asarray(opt.x, dtype=float)
    u_j = jnp.asarray(u, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](u_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](u_j)), dtype=float)
    out = jax.device_get(funcs["output"](u_j))
    out_np = {key: np.asarray(value, dtype=float) for key, value in out.items()}
    return {
        "u": u,
        "out": out_np,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": steady_s,
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "jacobian_condition": _condition_number(J),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "message": str(opt.message),
    }


def _interface_metrics_from_blocks(params: Any, blocks: np.ndarray) -> dict[str, float]:
    y = np.asarray(blocks, dtype=float).reshape(-1)
    return {**p2c57.p2c47._interface_load_metrics(params, y, steps_per_flex=STEPS_PER_FLEX), **p2c57._combined_force_metrics(params, y)}


def _row(
    *,
    mode: str,
    init_name: str,
    position_scale: float,
    slack_reg_scale: float,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
) -> dict[str, Any]:
    out = solve["out"]
    load = _interface_metrics_from_blocks(params, out["blocks"])
    position_error = float(np.linalg.norm(out["tip_p"] - target_p))
    continuity_norm = float(np.linalg.norm(out["normalized_jumps"]))
    row = {
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "mode": mode,
        "init_name": init_name,
        "position_scale": float(position_scale),
        "slack_reg_scale": float(slack_reg_scale),
        "position_error_m": position_error,
        "position_error_fraction_length": position_error / LENGTH_M,
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "continuity_norm": continuity_norm,
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        "message": solve["message"],
        **load,
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["position_error_fraction_length"] <= 0.05
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )
    if row["baseline_gate"]:
        row["failure_class"] = "通过"
    elif row["stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["jacobian_condition"] >= 1.0e7:
        row["failure_class"] = "条件数病态"
    elif row["position_error_fraction_length"] > 0.05:
        row["failure_class"] = "任务残差不可达"
    elif row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "刚性磁体段载荷传播失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for mode in sorted({row["mode"] for row in rows}):
        sub = [row for row in rows if row["mode"] == mode]
        best = min(
            sub,
            key=lambda row: (
                not row["baseline_gate"],
                row["stationarity_norm"],
                row["jacobian_condition"],
                row["position_error_m"],
            ),
        )
        summaries.append(
            {
                "mode": mode,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "best_init": best["init_name"],
                "best_position_scale": best["position_scale"],
                "best_slack_reg_scale": best["slack_reg_scale"],
                "best_position_error_m": best["position_error_m"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_condition": best["jacobian_condition"],
                "best_rank": best["jacobian_rank"],
                "best_boundary_norm": best["boundary_norm"],
                "best_force_ratio_to_rigid_load": best["max_interface_force_jump_over_rigid_force"],
                "best_force_ratio_to_internal_force": best["max_force_ratio_to_internal_force"],
                "dominant_failure": best["failure_class"],
            }
        )
    return summaries


def _plot(rows: list[dict[str, Any]], best_solves: dict[str, dict[str, Any]], oracle_cl: np.ndarray) -> None:
    labels = [f"{r['mode']}\n{r['init_name']}\npos{r['position_scale']:.0f}\nreg{r['slack_reg_scale']:.0e}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(4, 1, figsize=(max(12, 0.55 * len(rows)), 11), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "position_error_m", "末端位置误差(m)"),
        (axes[1], "stationarity_norm", "一阶最优性"),
        (axes[2], "jacobian_condition", "条件数"),
        (axes[3], "max_interface_force_jump_over_rigid_force", "相对刚段外力"),
    ]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values) & (values > 0.0)]
        cap = float(np.max(finite) * 10.0) if finite.size else 1.0
        values = np.nan_to_num(values, nan=cap, posinf=cap, neginf=cap)
        ax.bar(x, values)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_yscale("log")
    axes[1].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[-1].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    fig.suptitle("n2 低维力传递参数化门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lowdim_load_transfer_metrics_cn.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.2, label="FK参照")
    for name, solve in best_solves.items():
        cl = solve["out"]["centerline"]
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], "--", lw=2.0, label=name)
        tip = solve["out"]["tip_p"]
        ax.scatter([tip[0]], [tip[1]], [tip[2]], s=35)
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=40, label="基端")
    ax.set_title("n2 低维力传递参数化最优分支")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lowdim_load_transfer_centerlines_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 参数化 | 是否过门 | 最优初值 | 位置权重 | 松弛正则 | 末端误差/m | 一阶最优性 | 条件数 | 秩 | 边界残差 | 相对刚段外力 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['mode']} | {row['any_gate_pass']} | {row['best_init']} | {row['best_position_scale']:.0f} | "
            f"{row['best_slack_reg_scale']:.0e} | {row['best_position_error_m']:.3e} | {row['best_stationarity_norm']:.3e} | "
            f"{row['best_condition']:.3e} | {row['best_rank']} | {row['best_boundary_norm']:.3e} | "
            f"{row['best_force_ratio_to_rigid_load']:.3e} | {row['dominant_failure']} |"
        )
    return "\n".join(
        [
            "# P2c61 n2 低维力传递参数化记录",
            "",
            "## 实验目的",
            "",
            "P2c60 表明简单变量尺度重整不能使 P2c59 近门样本通过条件数门禁。本轮比较更低维、更物理闭合的力传递参数化，判断 18 维独立分段内力块是否是主要病态来源。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10` 与三维末端位置任务。比较两类模型：`base_6d` 为硬连续基端内力/内矩打靶；`load_scaled` 为基端 6 维加每个接口沿刚段外力/外矩方向的标量松弛。后者保留载荷传播方向，只允许低维幅值修正。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若低维模型过门，下一步进入分支一致性和扰动重求解；若低维模型不过门但条件数显著下降，则说明原问题有自由内力块引入的坐标病态；若末端误差或边界残差明显退化，则说明当前任务需要更丰富的物理状态，而不能简单压缩自由度。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门模型，建立可信分支并做 `eps=0.005,0.010` 扰动重求解。",
            "2. 若仅接近过门，围绕该参数化做局部权重细化。",
            "3. 若低维模型不可达，形成 n2 当前模型下的低维闭合不可发布证据，并回到显式物理残差或模型耦合修复。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=STEPS_PER_FLEX)
    oracle_cl = np.asarray(p2c57.forward_segment_wrench(params, jnp.asarray(y_seed), steps_per_flex=STEPS_PER_FLEX).centerline)

    n_interfaces = max(len(params.flex) - 1, 0)
    rows: list[dict[str, Any]] = []
    solves: dict[str, dict[str, Any]] = {}
    for mode in ("base_6d", "load_scaled"):
        init_map = {
            "oracle_base": y_seed[:6],
            "half_base": 0.5 * y_seed[:6],
            "zero": np.zeros(6, dtype=float),
        }
        if mode == "load_scaled":
            init_map = {key: np.concatenate([value, np.zeros(2 * n_interfaces, dtype=float)]) for key, value in init_map.items()}
        for position_scale in POSITION_SCALES:
            reg_values = (0.0,) if mode == "base_6d" else SLACK_REG_SCALES
            for slack_reg_scale in reg_values:
                funcs = _make_functions(params, target_p, mode=mode, position_scale=position_scale, slack_reg_scale=slack_reg_scale)
                for init_name, u0 in init_map.items():
                    solve = _solve(u0, funcs)
                    row = _row(
                        mode=mode,
                        init_name=init_name,
                        position_scale=position_scale,
                        slack_reg_scale=slack_reg_scale,
                        params=params,
                        target_p=target_p,
                        solve=solve,
                    )
                    rows.append(row)
                    key = f"{mode}_{init_name}_p{position_scale:.0f}_r{slack_reg_scale:.0e}"
                    solves[key] = solve
                    if not row["baseline_gate"]:
                        _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    best_solves = {}
    for summary in summaries:
        candidates = [row for row in rows if row["mode"] == summary["mode"]]
        best = min(candidates, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["jacobian_condition"], row["position_error_m"]))
        key = f"{best['mode']}_{best['init_name']}_p{best['position_scale']:.0f}_r{best['slack_reg_scale']:.0e}"
        best_solves[summary["mode"]] = solves[key]
    _plot(rows, best_solves, oracle_cl)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "lowdim_load_transfer_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "position_scales": list(POSITION_SCALES),
        "boundary_scale": BOUNDARY_SCALE,
        "slack_reg_scales": list(SLACK_REG_SCALES),
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
    (OUT_DIR / "record.md").write_text(_record(summaries, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "summaries": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
