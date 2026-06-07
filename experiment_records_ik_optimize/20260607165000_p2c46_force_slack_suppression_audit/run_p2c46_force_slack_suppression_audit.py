"""Audit force-channel suppression for interface slack."""

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
P2C13_DIR = ROOT / "experiment_records_ik_optimize" / "20260605123000_p2c13_rigid_load_propagation_audit"
P2C41_DIR = ROOT / "experiment_records_ik_optimize" / "20260607144044_p2c41_scaled_interface_slack_baseline_gate"
P2C44_DIR = ROOT / "experiment_records_ik_optimize" / "20260607154050_p2c44_terminal_boundary_component_audit"
for path in (ROOT, FK_OPT_DIR, P2C13_DIR, P2C41_DIR, P2C44_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
from forward_kinematics_optimized.equilibrium_solver_nondim import with_coil_currents

import run_p2c13_rigid_load_propagation_audit as p2c13
import run_p2c41_scaled_interface_slack_baseline_gate as p2c41
import run_p2c44_terminal_boundary_component_audit as p2c44

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


def _parse_floats(text: str) -> list[float]:
    return [float(raw.strip()) for raw in text.split(",") if raw.strip()]


def _scaled_slack_components(params: Any, y: jnp.ndarray, args: argparse.Namespace) -> tuple[jnp.ndarray, jnp.ndarray]:
    scaled = p2c41.load_scaled_slack_residual(
        params,
        y,
        variant="wrench",
        load_floor=args.load_floor,
        steps_per_flex=args.steps_per_flex,
    )
    n_interfaces = max(len(params.flex) - 1, 0)
    if n_interfaces == 0:
        z = jnp.zeros((0,), dtype=jnp.float64)
        return z, z
    blocks = scaled.reshape((n_interfaces, 6))
    return blocks[:, :3].reshape(-1), blocks[:, 3:].reshape(-1)


def _make_functions(
    params_base: Any,
    target_p: np.ndarray,
    args: argparse.Namespace,
    *,
    boundary_mode: str,
    force_slack_scale: float,
) -> dict[str, Any]:
    target_p_j = jnp.asarray(target_p, dtype=jnp.float64).reshape(3)
    mask = p2c44._mode_mask(boundary_mode)

    def params_for_I(I: jnp.ndarray) -> Any:
        return with_coil_currents(params_base, jnp.asarray(I, dtype=jnp.float64).reshape(8))

    def output_yI(y: jnp.ndarray, I: jnp.ndarray) -> Any:
        return p2c41.p2c10t.forward_interface_slack(params_for_I(I), y, variant="wrench", steps_per_flex=args.steps_per_flex)

    def residual_yI(y: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        params = params_for_I(I)
        out = p2c41.p2c10t.forward_interface_slack(params, y, variant="wrench", steps_per_flex=args.steps_per_flex)
        r_p = float(args.position_scale) * (out.tip_p - target_p_j)
        r_b = float(args.boundary_scale) * mask * out.boundary
        r_sf, r_sm = _scaled_slack_components(params, y, args)
        r_sf = float(force_slack_scale) * r_sf
        r_sm = float(args.torque_slack_scale) * r_sm
        return jnp.concatenate([r_p.reshape(-1), r_b.reshape(-1), r_sf.reshape(-1), r_sm.reshape(-1)])

    return {
        "residual": jax.jit(residual_yI),
        "jacobian": jax.jit(jax.jacfwd(residual_yI, argnums=0)),
        "output": jax.jit(output_yI),
    }


def _trace_metrics(params: Any, y: np.ndarray) -> dict[str, float]:
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    interface_rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    return {
        "min_slack_force_cosine_natural_delta": min(
            (float(row["slack_force_cosine_natural_delta"]) for row in interface_rows),
            default=float("nan"),
        ),
        "max_slack_force_over_natural_force_delta": max(
            (
                float(row["slack_force_norm_dim"] / max(row["natural_force_delta_norm_dim"], 1.0e-30))
                for row in interface_rows
            ),
            default=float("nan"),
        ),
    }


def _run_one(case_name: str, alpha: float, boundary_mode: str, force_slack_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    params, I0 = p2c44.p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c44.p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y0 = p2c44.p2c24._seed_from_reference(np.asarray(y_ref, dtype=float), eta=1.0, base_scale=1.0)
    funcs = _make_functions(params, target_p, args, boundary_mode=boundary_mode, force_slack_scale=force_slack_scale)
    solve = p2c44._solve(y0, I0, funcs, args)
    load = p2c41._load_audit(params, solve["y"], I0)
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    row = {
        "case": case_name,
        "alpha": alpha,
        "boundary_mode": boundary_mode,
        "force_slack_scale": force_slack_scale,
        "torque_slack_scale": args.torque_slack_scale,
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "max_slack_force_over_rigid_force": load["max_slack_force_over_rigid_force"],
        "max_slack_torque_over_rigid_torque": load["max_slack_torque_over_rigid_torque"],
        "steady_s": solve["steady_s"],
        "nfev": solve["nfev"],
        "success": solve["success"],
    }
    row.update(_trace_metrics(params, solve["y"]))
    row["baseline_gate"] = bool(
        row["success"]
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


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for case in sorted({row["case"] for row in rows}):
        sub = [row for row in rows if row["case"] == case]
        best_gate = [row for row in sub if row["baseline_gate"]]
        best = min(sub, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_slack_force_over_rigid_force"]))
        min_slack = min(sub, key=lambda row: row["max_slack_force_over_rigid_force"])
        min_torque = min(sub, key=lambda row: row["terminal_torque_norm"])
        summaries.append(
            {
                "case": case,
                "any_gate_pass": bool(best_gate),
                "best_mode": best["boundary_mode"],
                "best_force_slack_scale": best["force_slack_scale"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_slack_force_ratio": best["max_slack_force_over_rigid_force"],
                "best_condition": best["jacobian_condition"],
                "min_slack_mode": min_slack["boundary_mode"],
                "min_slack_force_scale": min_slack["force_slack_scale"],
                "min_slack_force_ratio": min_slack["max_slack_force_over_rigid_force"],
                "min_torque_mode": min_torque["boundary_mode"],
                "min_torque_force_scale": min_torque["force_slack_scale"],
                "min_terminal_torque_norm": min_torque["terminal_torque_norm"],
                "row_count": len(sub),
            }
        )
    return summaries


def _plot(rows: list[dict[str, Any]]) -> None:
    mode_cn = {"torque": "仅矩", "full": "力矩全闭合"}
    for case in sorted({row["case"] for row in rows}):
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for mode in sorted({row["boundary_mode"] for row in rows}):
            sub = sorted([row for row in rows if row["case"] == case and row["boundary_mode"] == mode], key=lambda row: row["force_slack_scale"])
            x = [row["force_slack_scale"] for row in sub]
            axes[0].plot(x, [row["max_slack_force_over_rigid_force"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
            axes[1].plot(x, [row["stationarity_norm"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
            axes[2].plot(x, [row["terminal_torque_norm"] for row in sub], marker="o", label=mode_cn.get(mode, mode))
        axes[0].axhline(1.5, color="black", ls="--", lw=1.0)
        axes[0].axhline(2.0, color="red", ls=":", lw=1.0)
        axes[1].axhline(1.0e-5, color="black", ls="--", lw=1.0)
        axes[0].set_ylabel("载荷传播比")
        axes[1].set_ylabel("一阶最优性残差")
        axes[2].set_ylabel("末端内矩残差")
        axes[2].set_xlabel("接口力通道权重")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"{case} 接口力通道抑制审计")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case}_force_slack_suppression_cn.png", dpi=180)
        plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 是否过门 | 最优模式 | 最优接口力权重 | 最优一阶最优性 | 最优载荷传播比 | 最小载荷传播比 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    mode_cn = {"torque": "仅矩", "full": "力矩全闭合"}
    for summary in summaries:
        conclusion = "可进入扰动复核" if summary["any_gate_pass"] else "不可发布，需模型修复"
        table.append(
            f"| {summary['case']} | {summary['any_gate_pass']} | {mode_cn.get(summary['best_mode'], summary['best_mode'])} | "
            f"{summary['best_force_slack_scale']:.3g} | {summary['best_stationarity_norm']:.3e} | "
            f"{summary['best_slack_force_ratio']:.3e} | {summary['min_slack_force_ratio']:.3e} | {conclusion} |"
        )
    return "\n".join(
        [
            "# P2c46 接口力通道抑制审计记录",
            "",
            "## 实验目的",
            "",
            "P2c45 显示接口力松弛是单点和两点基线失败的重要通道。本轮将接口松弛拆成力通道和矩通道，只提高力通道惩罚，观察是否能在保留末端内矩闭合的同时降低非物理接口力补偿。",
            "",
            "## 实验方案",
            "",
            "保持位置任务、边界权重、矩通道权重和最大迭代次数不变，仅扫描接口力通道权重。边界模式只保留 P2c45 中相对有意义的 `torque` 和 `full`，避免重复无效的大范围边界扫描。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 边界模式：`{args.modes}`",
            f"- 接口力通道权重：`{args.force_slack_scales}`",
            f"- 接口矩通道权重：`{args.torque_slack_scale}`",
            f"- 边界权重：`{args.boundary_scale}`",
            f"- 最大函数调用次数：`{args.max_iter}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若提高接口力通道权重仍不能同时压低载荷传播比和一阶最优性残差，则说明当前问题不是单一权重尺度错误，而是接口力/矩传播方程与末端边界闭合存在结构性不相容。此时应停止继续扫松弛权重，转向显式刚柔段载荷耦合模型。",
            "",
            "## 下一步计划",
            "",
            "1. 若某个工况过基线门禁，则只对该工况做小扰动重求解。",
            "2. 若无工况过门禁，则实现显式接口载荷传播残差，避免自由松弛变量承担等效外力。",
            "3. 单点若持续有效秩不足，应转入有效子空间位置灵敏度路线。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="n1_tip_z_rand15:0.10,n2_z_pair_rand12:0.10")
    parser.add_argument("--modes", default="torque,full")
    parser.add_argument("--force_slack_scales", default="2,4,8,16")
    parser.add_argument("--torque_slack_scale", type=float, default=2.0)
    parser.add_argument("--boundary_scale", type=float, default=0.25)
    parser.add_argument("--max_iter", type=int, default=120)
    parser.add_argument("--steps_per_flex", type=int, default=4)
    parser.add_argument("--position_scale", type=float, default=60.0)
    parser.add_argument("--load_floor", type=float, default=0.05)
    args = parser.parse_args()

    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    force_scales = _parse_floats(args.force_slack_scales)
    for case_name, alpha in _parse_specs(args.specs):
        for mode in modes:
            for force_scale in force_scales:
                row = _run_one(case_name, alpha, mode, force_scale, args)
                rows.append(row)
                if not row["baseline_gate"]:
                    _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})
    summaries = _summaries(rows)
    _plot(rows)
    _write_csv(OUT_DIR / "force_slack_rows.csv", rows)
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
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
