"""Audit force and torque components of terminal boundary closure."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C41_DIR = ROOT / "experiment_records_ik_optimize" / "20260607144044_p2c41_scaled_interface_slack_baseline_gate"
for path in (ROOT, FK_OPT_DIR, P2C24_DIR, P2C41_DIR):
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
from inverse_kinematics.ik_terminal_shooting_jax import output_to_numpy

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c41_scaled_interface_slack_baseline_gate as p2c41

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
            case_name, alpha = raw.split(":", 1)
            specs.append((case_name.strip(), float(alpha)))
    return specs


def _mode_mask(mode: str) -> jnp.ndarray:
    if mode == "none":
        return jnp.asarray([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
    if mode == "force":
        return jnp.asarray([1, 1, 1, 0, 0, 0], dtype=jnp.float64)
    if mode == "torque":
        return jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.float64)
    if mode == "full":
        return jnp.asarray([1, 1, 1, 1, 1, 1], dtype=jnp.float64)
    raise ValueError(f"Unknown boundary mode: {mode}")


def _make_functions(params_base: Any, target_p: np.ndarray, args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    mask = _mode_mask(mode)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c41.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = p2c41.p2c10t.forward_interface_slack(params, y, variant="wrench", steps_per_flex=args.steps_per_flex)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_s = float(args.slack_scale) * p2c41.load_scaled_slack_residual(
            params,
            y,
            variant="wrench",
            load_floor=args.load_floor,
            steps_per_flex=args.steps_per_flex,
        )
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_s.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _solve(y0: np.ndarray, I: np.ndarray, funcs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    I_j = jnp.asarray(I, dtype=jnp.float64).reshape(8)
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    _ = funcs["residual"](jnp.asarray(y0), I_j).block_until_ready()
    t0 = time.perf_counter()
    opt = least_squares(
        lambda yy: np.asarray(jax.device_get(funcs["residual"](jnp.asarray(yy), I_j)), dtype=float),
        y0,
        jac=lambda yy: np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(yy), I_j)), dtype=float),
        method="trf",
        x_scale=np.ones_like(y0),
        loss="linear",
        max_nfev=int(args.max_iter),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    steady_s = time.perf_counter() - t0
    y = np.asarray(opt.x, dtype=float)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    r = np.asarray(jax.device_get(funcs["residual"](y_j, I_j)), dtype=float)
    J = np.asarray(jax.device_get(funcs["jacobian"](y_j, I_j)), dtype=float)
    out = output_to_numpy(funcs["output"](y_j, I_j))
    svals = np.linalg.svd(J, compute_uv=False)
    return {
        "y": y,
        "out": out,
        "success": bool(opt.success),
        "nfev": int(opt.nfev),
        "steady_s": float(steady_s),
        "residual_norm": float(np.linalg.norm(r)),
        "stationarity_norm": float(np.linalg.norm(J.T @ r)),
        "jacobian_condition": float(svals[0] / max(float(svals[-1]), np.finfo(float).tiny)) if svals.size else float("inf"),
        "jacobian_rank": int(np.linalg.matrix_rank(J)),
        "message": str(opt.message),
    }


def _run_one(case_name: str, alpha: float, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    params, I0 = p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y0 = p2c24._seed_from_reference(np.asarray(y_ref, dtype=float), eta=1.0, base_scale=1.0)
    funcs = _make_functions(params, target_p, args, mode=mode)
    solve = _solve(y0, I0, funcs, args)
    load = p2c41._load_audit(params, solve["y"], I0)
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    position_error = float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p))
    row = {
        "case": case_name,
        "alpha": alpha,
        "boundary_mode": mode,
        "success": solve["success"],
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "position_error_m": position_error,
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "boundary_norm": float(np.linalg.norm(boundary)),
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "max_slack_force_over_rigid_force": load["max_slack_force_over_rigid_force"],
        "max_slack_torque_over_rigid_torque": load["max_slack_torque_over_rigid_torque"],
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["position_error_m"] <= 0.05 * 0.11
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["max_slack_force_over_rigid_force"] <= 1.5
    )
    if row["baseline_gate"]:
        row["failure_class"] = "通过"
    elif row["stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["max_slack_force_over_rigid_force"] > 2.0:
        row["failure_class"] = "载荷传播失配"
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summary(case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(rows, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]))
    best_force = min(rows, key=lambda row: row["terminal_force_norm"])
    best_torque = min(rows, key=lambda row: row["terminal_torque_norm"])
    return {
        "case": case_name,
        "any_gate_pass": any(row["baseline_gate"] for row in rows),
        "best_mode": best["boundary_mode"],
        "best_stationarity_norm": best["stationarity_norm"],
        "best_jacobian_condition": best["jacobian_condition"],
        "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
        "best_position_error_m": best["position_error_m"],
        "min_terminal_force_mode": best_force["boundary_mode"],
        "min_terminal_force_norm": best_force["terminal_force_norm"],
        "min_terminal_torque_mode": best_torque["boundary_mode"],
        "min_terminal_torque_norm": best_torque["terminal_torque_norm"],
        "pass_modes": ",".join(row["boundary_mode"] for row in rows if row["baseline_gate"]),
        "row_count": len(rows),
    }


def _plot(rows: list[dict[str, Any]]) -> None:
    mode_cn = {"none": "无边界", "force": "仅力", "torque": "仅矩", "full": "力矩全闭合"}
    for case in sorted({row["case"] for row in rows}):
        sub = [row for row in rows if row["case"] == case]
        x = np.arange(len(sub))
        labels = [mode_cn[row["boundary_mode"]] for row in sub]
        fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
        specs = [
            ("stationarity_norm", "一阶最优性残差", 1.0e-5),
            ("terminal_force_norm", "末端内力残差", None),
            ("terminal_torque_norm", "末端内矩残差", None),
            ("max_slack_force_over_rigid_force", "载荷传播比", 1.5),
        ]
        for ax, (key, ylabel, gate) in zip(axes, specs):
            ax.bar(x, [row[key] for row in sub], color=["green" if row["baseline_gate"] else "orange" for row in sub])
            if gate is not None:
                ax.axhline(gate, color="black", ls="--", lw=1.0)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=10, ha="right")
        fig.suptitle(f"{case} 末端边界分量审计")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case}_boundary_component_audit_cn.png", dpi=180)
        plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 是否有通过模式 | 最优模式 | 最优一阶最优性 | 最优载荷传播比 | 最小内力模式 | 最小内矩模式 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    mode_cn = {"none": "无边界", "force": "仅力", "torque": "仅矩", "full": "力矩全闭合"}
    for summary in summaries:
        if summary["any_gate_pass"]:
            conclusion = "存在可进入扰动重求解的边界模式。"
        else:
            conclusion = "边界分量拆分仍不能通过门禁，边界条件与任务/接口模型不相容。"
        table.append(
            f"| {summary['case']} | {summary['any_gate_pass']} | {mode_cn[summary['best_mode']]} | "
            f"{summary['best_stationarity_norm']:.3e} | {summary['best_slack_force_ratio']:.3e} | "
            f"{mode_cn[summary['min_terminal_force_mode']]} | {mode_cn[summary['min_terminal_torque_mode']]} | {conclusion} |"
        )
    return "\n".join(
        [
            "# P2c44 末端边界分量审计记录",
            "",
            "## 实验目的",
            "",
            "P2c43 表明边界平衡失败不是单纯权重尺度误设。本轮将末端边界残差拆分为末端内力和末端内矩，分别测试无边界、仅力闭合、仅矩闭合和力矩全闭合四种模式，判断病态来自哪一类终端物理约束。",
            "",
            "## 实验方案",
            "",
            "仍使用载荷定标接口松弛模型和三维末端位置任务。每个边界模式独立求解一次，并记录一阶最优性、条件数、末端内力残差、末端内矩残差、载荷传播比和末端位置误差。基线不过门时不做扰动灵敏度。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 边界权重：`{args.boundary_scale}`",
            f"- 接口松弛权重：`{args.slack_scale}`",
            f"- 最大函数调用次数：`{args.max_iter}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若无边界或单分量边界模式可显著降低一阶最优性，但力矩全闭合失败，则说明完整末端自由边界与目标形态存在相容性问题；若所有模式均失败，则说明问题更可能来自接口载荷传播参数化或当前目标工作点本身。",
            "",
            "## 下一步计划",
            "",
            "1. 若某一边界分量模式过门禁，则基于该模式做小扰动重求解灵敏度审计。",
            "2. 若所有模式不过门，则停止边界权重与边界分量扫描，进入显式刚柔段载荷耦合模型。",
            "3. 若单点持续失败，应转向有效子空间位置灵敏度或工程不可发布结论。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--modes", default="none,force,torque,full")
    parser.add_argument("--boundary_scale", type=float, default=0.25)
    parser.add_argument("--slack_scale", type=float, default=2.0)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--load_floor", type=float, default=0.05)
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for case_name, alpha in _parse_specs(args.specs):
        case_rows = []
        for mode in modes:
            row = _run_one(case_name, alpha, mode, args)
            rows.append(row)
            case_rows.append(row)
            if not row["baseline_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "case": case_name, "mode": mode, "failure": row["failure_class"], "evidence": row})
        summaries.append(_summary(case_name, case_rows))
    _plot(rows)
    _write_csv(OUT_DIR / "boundary_component_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "args": vars(args),
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
    (OUT_DIR / "record.md").write_text(_record(args, summaries), encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points\n"
        "XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n evolution_of_mcrs python "
        "experiment_records_ik_optimize/20260607154050_p2c44_terminal_boundary_component_audit/"
        "run_p2c44_terminal_boundary_component_audit.py\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
