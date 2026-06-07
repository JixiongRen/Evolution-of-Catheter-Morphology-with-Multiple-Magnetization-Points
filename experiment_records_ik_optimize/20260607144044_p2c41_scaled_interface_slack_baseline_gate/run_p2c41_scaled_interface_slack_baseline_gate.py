"""Evaluate load-scaled interface slack for single- and double-magnet cases."""

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
P2C10T_DIR = ROOT / "experiment_records_ik_optimize" / "20260605102000_p2c10t_interface_slack_shooting"
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
for path in (ROOT, FK_OPT_DIR, P2C10T_DIR, P2C13_DIR, P2C24_DIR):
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
from inverse_kinematics.ik_terminal_shooting_jax import _advance_rigid, _integrate_flex, initial_state, output_to_numpy

import run_p2c10t_interface_slack_shooting as p2c10t
import run_p2c13_rigid_load_propagation_audit as p2c13
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _parse_specs(text: str) -> list[tuple[str, float]]:
    specs = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            name, alpha = raw.split(":", 1)
            specs.append((name.strip(), float(alpha)))
    return specs


def _parse_schedule(text: str) -> list[tuple[str, float, float]]:
    stages = []
    for raw in text.split(";"):
        raw = raw.strip()
        if raw:
            label, boundary, slack = raw.split(":", 2)
            stages.append((label.strip(), float(boundary), float(slack)))
    return stages


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


def load_scaled_slack_residual(params: Any, y: jnp.ndarray, *, variant: str, load_floor: float, steps_per_flex: int) -> jnp.ndarray:
    """Return interface slack normalized by local rigid magnetic load."""
    n_interfaces = max(len(params.flex) - 1, 0)
    base, force_slack, torque_slack = p2c10t._split_state(jnp.asarray(y, dtype=jnp.float64), variant, n_interfaces)
    x = initial_state(params, base)
    pieces = []
    f_ref = float(params.scales.F_ref)
    m_ref = float(params.scales.M_ref)
    floor_f = float(load_floor) * max(f_ref, 1.0e-12)
    floor_m = float(load_floor) * max(m_ref, 1.0e-12)
    for i in range(len(params.flex)):
        x, _points = _integrate_flex(params, x, i, steps_per_flex)
        x_prox = x
        f_total, tau_total = _rigid_wrench(params, x_prox, i)
        x, _mid_p, _mid_q = _advance_rigid(params, x_prox, i)
        if i < n_interfaces:
            slack_f_dim = force_slack[i] * f_ref
            slack_m_dim = torque_slack[i] * m_ref
            denom_f = jnp.maximum(jnp.linalg.norm(f_total), floor_f)
            denom_m = jnp.maximum(jnp.linalg.norm(tau_total), floor_m)
            pieces.extend([slack_f_dim / denom_f, slack_m_dim / denom_m])
            x = x.at[7:10].add(slack_f_dim)
            x = x.at[10:13].add(slack_m_dim)
    if not pieces:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate([piece.reshape(-1) for piece in pieces], axis=0)


def _make_functions(params_base: Any, target_p: np.ndarray, args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray, boundary_scale: jnp.ndarray, slack_scale: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = p2c10t.forward_interface_slack(params, y, variant="wrench", steps_per_flex=args.steps_per_flex)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = jnp.asarray(boundary_scale, dtype=jnp.float64).reshape(()) * out.boundary
        y_flat = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        if mode == "raw":
            r_slack = jnp.asarray(slack_scale, dtype=jnp.float64).reshape(()) * y_flat[6:]
        elif mode == "load_scaled":
            scaled = load_scaled_slack_residual(params, y, variant="wrench", load_floor=args.load_floor, steps_per_flex=args.steps_per_flex)
            r_slack = jnp.asarray(slack_scale, dtype=jnp.float64).reshape(()) * scaled
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_slack.reshape(-1)], axis=0)

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], *, boundary_scale: float, slack_scale: float, max_iter: int) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    b_j = jnp.asarray(boundary_scale, dtype=jnp.float64)
    s_j = jnp.asarray(slack_scale, dtype=jnp.float64)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j, b_j, s_j).block_until_ready()
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy), I_j, b_j, s_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=int(max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j, b_j, s_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j, b_j, s_j)), dtype=float)
    out = output_to_numpy(funcs["output"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "message": str(opt.message),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
    }


def _load_audit(params: Any, y: np.ndarray, I: np.ndarray) -> dict[str, float]:
    params_i = with_coil_currents(params, jnp.asarray(I, dtype=jnp.float64).reshape(8))
    trace = p2c13._trace_state(params_i, y, steps_per_flex=4)
    rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "max_slack_force_over_rigid_force": max((float(row["slack_force_over_rigid_force"]) for row in rows), default=0.0),
        "max_slack_torque_over_rigid_torque": max((float(row["slack_torque_over_rigid_torque"]) for row in rows), default=0.0),
        "min_slack_force_cosine_rigid_force": min((float(row["slack_force_cosine_rigid_force"]) for row in rows), default=float("nan")),
        "min_slack_torque_cosine_rigid_torque": min((float(row["slack_torque_cosine_rigid_torque"]) for row in rows), default=float("nan")),
        "trace_boundary_norm": float(np.linalg.norm(trace["boundary"])),
    }


def _row(case_name: str, alpha: float, mode: str, stage: str, boundary_scale: float, slack_scale: float, target_p: np.ndarray, params: Any, I: np.ndarray, solve: dict[str, Any]) -> dict[str, Any]:
    load = _load_audit(params, solve["y"], I)
    pos_err = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "case": case_name,
        "alpha": alpha,
        "mode": mode,
        "stage": stage,
        "boundary_scale": boundary_scale,
        "slack_scale": slack_scale,
        "success": solve["success"],
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "position_error_m": pos_err,
        "residual_norm": solve["residual_norm"],
        "stationarity_norm": solve["stationarity_norm"],
        "boundary_norm": solve["boundary_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        **load,
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["max_slack_force_over_rigid_force"] <= 1.5
    )
    return row


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes = []
    if not row["success"]:
        modes.append("求解器未报告成功")
    if row["position_error_m"] > 0.05 * 0.11:
        modes.append("末端位置约束失配")
    if row["stationarity_norm"] > 1.0e-5:
        modes.append("一阶最优性退化")
    if row["jacobian_condition"] >= 1.0e9:
        modes.append("条件数严重病态")
    elif row["jacobian_condition"] >= 1.0e7:
        modes.append("条件数偏高")
    if row["max_slack_force_over_rigid_force"] > 2.0:
        modes.append("刚性磁体段载荷传播失配")
    elif row["max_slack_force_over_rigid_force"] > 1.5:
        modes.append("刚性磁体段载荷传播临界")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    for item in ["刚性磁体段载荷传播失配", "一阶最优性退化", "条件数严重病态", "条件数偏高", "末端位置约束失配"]:
        if item in modes:
            return item
    return modes[0]


def _run_case(case_name: str, alpha: float, args: argparse.Namespace, heartbeat: Path, failures: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    all_rows: list[dict[str, Any]] = []

    for mode in ["raw", "load_scaled"]:
        funcs = _make_functions(params, target_p, args, mode=mode)
        y = p2c24._seed_from_reference(y_ref, eta=args.start_eta, base_scale=args.start_base_scale)
        for stage, boundary_scale, slack_scale in _parse_schedule(args.schedule):
            _append_jsonl(heartbeat, {"ts": time.time(), "case": case_name, "alpha": alpha, "mode": mode, "stage": stage})
            solve = _solve(y, I0, funcs, boundary_scale=boundary_scale, slack_scale=slack_scale, max_iter=args.max_iter)
            row = _row(case_name, alpha, mode, stage, boundary_scale, slack_scale, target_p, params, I0, solve)
            all_rows.append(row)
            y = solve["y"]

    best = min(all_rows, key=lambda r: (not r["baseline_gate"], r["max_slack_force_over_rigid_force"], r["stationarity_norm"], r["position_error_m"]))
    modes = [] if best["baseline_gate"] else _failure_modes(best)
    summary = {
        "case": case_name,
        "alpha": alpha,
        "best_mode": best["mode"],
        "best_stage": best["stage"],
        "baseline_gate": best["baseline_gate"],
        "best_position_error_m": best["position_error_m"],
        "best_stationarity_norm": best["stationarity_norm"],
        "best_jacobian_condition": best["jacobian_condition"],
        "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
        "best_slack_torque_ratio": best["max_slack_torque_over_rigid_torque"],
        "best_boundary_norm": best["boundary_norm"],
        "raw_best_slack_force_ratio": min(r["max_slack_force_over_rigid_force"] for r in all_rows if r["mode"] == "raw"),
        "load_scaled_best_slack_force_ratio": min(r["max_slack_force_over_rigid_force"] for r in all_rows if r["mode"] == "load_scaled"),
        "primary_failure_mode": "" if best["baseline_gate"] else _primary_mode(modes),
        "failure_modes": "" if best["baseline_gate"] else ";".join(modes),
        "total_steady_s": sum(r["steady_s"] for r in all_rows),
    }
    if not best["baseline_gate"]:
        _append_jsonl(failures, {"ts": time.time(), "case": case_name, "alpha": alpha, "failure": "baseline_gate_failed", "evidence": summary})
    return summary, all_rows


def _plot_summary(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    mode_cn = {"raw": "原始松弛", "load_scaled": "载荷定标松弛"}
    labels = [f"{s['case']}\n{mode_cn.get(s['best_mode'], s['best_mode'])}" for s in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    specs = [
        ("best_stationarity_norm", "一阶最优性残差", 1.0e-5),
        ("best_jacobian_condition", "条件数", 1.0e7),
        ("best_slack_force_ratio", "界面力松弛/刚段外力", 1.5),
        ("best_position_error_m", "末端位置误差 m", 0.05 * 0.11),
    ]
    colors = ["green" if s["baseline_gate"] else "orange" for s in summaries]
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [s[key] for s in summaries], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=10, ha="right")
    fig.suptitle("P2c41 基线门禁结果")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c41_baseline_gate_summary.png", dpi=180)
    plt.close(fig)

    for case_name in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case_name]
        labels = [f"{mode_cn.get(r['mode'], r['mode'])}\n{r['stage']}" for r in sub]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.9 * len(labels)), 9), sharex=True)
        for ax, key, ylabel, gate in [
            (axes[0], "stationarity_norm", "一阶最优性", 1.0e-5),
            (axes[1], "max_slack_force_over_rigid_force", "载荷传播比", 1.5),
            (axes[2], "position_error_m", "末端位置误差 m", 0.05 * 0.11),
        ]:
            ax.bar(x, [r[key] for r in sub], color=["green" if r["baseline_gate"] else "orange" for r in sub])
            ax.axhline(gate, color="black", ls="--", lw=1.0)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        fig.suptitle(f"{case_name} 接口模型阶段指标")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case_name}_stage_metrics.png", dpi=180)
        plt.close(fig)


def _build_record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    mode_cn = {"raw": "原始松弛", "load_scaled": "载荷定标松弛"}
    table = ["| 工况 | 最优模型 | 基线门禁 | 一阶最优性 | 条件数 | 载荷传播比 | 主失败类型 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summaries:
        table.append(
            f"| {row['case']} | {mode_cn.get(row['best_mode'], row['best_mode'])} | {row['baseline_gate']} | "
            f"{row['best_stationarity_norm']:.3e} | {row['best_jacobian_condition']:.3e} | "
            f"{row['best_slack_force_ratio']:.3e} | {row['primary_failure_mode'] or '通过'} |"
        )
    return "\n".join(
        [
            "# P2c41 物理尺度化接口松弛基线门禁记录",
            "",
            "## 实验目的",
            "",
            "本轮实验检验 P2c40 推荐的物理尺度化接口松弛是否能改善单点与两点磁控打靶 IK 的刚性磁体段载荷传播失配。实验只做三维末端位置任务的基线门禁；若基线门禁不通过，不进入扰动灵敏度扫描。",
            "",
            "## 方法",
            "",
            "对每个工况比较两类残差：第一类是原始接口松弛罚项，直接惩罚界面松弛变量；第二类是物理尺度化接口松弛罚项，将界面松弛力和松弛力矩分别按本段刚性磁体外力和外力矩尺度归一化。这样可直接抑制“人工松弛量替代真实载荷链”的通道。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 阶段调度：`{args.schedule}`",
            f"- 最大函数调用次数：`{args.max_iter}`",
            f"- 位置残差尺度：`{args.position_scale}`",
            f"- 载荷尺度下限：`{args.load_floor}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "本轮结果只用于判定接口闭合修复方向是否有效。只有当一阶最优性、条件数、载荷传播比和末端位置误差同时过门禁时，才允许进入下一轮扰动重求解灵敏度审计。",
            "",
            "## 下一步计划",
            "",
            "1. 若 `load_scaled` 模型通过基线门禁，则对该工况执行 eps=0.005 与 0.010 的扰动重求解。",
            "2. 若载荷传播比下降但一阶最优性失败，则做一阶最优性残差分块归因。",
            "3. 若载荷传播比仍大于 2，则判定简单接口松弛尺度化不足，进入显式刚柔段载荷耦合模型。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--schedule", default="loose:0.10:0.50;mid:0.25:1.00;strict:0.50:2.00")
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--load_floor", type=float, default=0.05)
    parser.add_argument("--start_eta", type=float, default=1.0)
    parser.add_argument("--start_base_scale", type=float, default=1.0)
    args = parser.parse_args()

    heartbeat = OUT_DIR / "heartbeat.jsonl"
    failures = OUT_DIR / "failure_events.jsonl"
    heartbeat.write_text("", encoding="utf-8")
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case_name, alpha in _parse_specs(args.specs):
        summary, case_rows = _run_case(case_name, alpha, args, heartbeat, failures)
        summaries.append(summary)
        rows.extend(case_rows)
    _plot_summary(rows, summaries)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    _write_csv(OUT_DIR / "stage_rows.csv", rows)
    summary_payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
        "summaries": summaries,
        "baseline_pass_count": sum(1 for row in summaries if row["baseline_gate"]),
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_build_record(args, summaries), encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python "
        "experiment_records_ik_optimize/20260607144044_p2c41_scaled_interface_slack_baseline_gate/"
        "run_p2c41_scaled_interface_slack_baseline_gate.py\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary_payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
