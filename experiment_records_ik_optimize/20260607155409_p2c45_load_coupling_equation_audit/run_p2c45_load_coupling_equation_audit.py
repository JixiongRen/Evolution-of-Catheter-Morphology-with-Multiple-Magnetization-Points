"""Audit load-coupling terms behind terminal boundary failures."""

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
P2C44_DIR = ROOT / "experiment_records_ik_optimize" / "20260607154050_p2c44_terminal_boundary_component_audit"
for path in (ROOT, FK_OPT_DIR, P2C13_DIR, P2C44_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c13_rigid_load_propagation_audit as p2c13
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


def _trace_summary(params: Any, y: np.ndarray, mode_row: dict[str, Any]) -> dict[str, Any]:
    trace = p2c13._trace_state(params, y, steps_per_flex=4)
    interface_rows = [row for row in trace["rows"] if "slack_force_over_rigid_force" in row]
    natural_torque_ratios = [
        float(row["natural_torque_delta_norm_dim"] / max(row["rigid_torque_norm_dim"], 1.0e-30))
        for row in trace["rows"]
    ]
    natural_force_ratios = [
        float(row["natural_force_delta_norm_dim"] / max(row["rigid_force_norm_dim"], 1.0e-30))
        for row in trace["rows"]
    ]
    return {
        **mode_row,
        "max_natural_force_delta_over_rigid_force": max(natural_force_ratios, default=float("nan")),
        "max_natural_torque_delta_over_rigid_torque": max(natural_torque_ratios, default=float("nan")),
        "max_slack_force_over_natural_force_delta": max(
            (
                float(row["slack_force_norm_dim"] / max(row["natural_force_delta_norm_dim"], 1.0e-30))
                for row in interface_rows
            ),
            default=float("nan"),
        ),
        "max_slack_torque_over_natural_torque_delta": max(
            (
                float(row["slack_torque_norm_dim"] / max(row["natural_torque_delta_norm_dim"], 1.0e-30))
                for row in interface_rows
            ),
            default=float("nan"),
        ),
        "min_slack_force_cosine_natural_delta": min(
            (float(row["slack_force_cosine_natural_delta"]) for row in interface_rows),
            default=float("nan"),
        ),
        "min_slack_torque_cosine_natural_delta": min(
            (float(row["slack_torque_cosine_natural_delta"]) for row in interface_rows),
            default=float("nan"),
        ),
    }


def _run_case_mode(case_name: str, alpha: float, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    params, I0 = p2c44.p2c24._params_from_saved_case(case_name, alpha)
    target_p, y_ref = p2c44.p2c24._load_saved_target_and_seed(case_name, alpha)
    target_p = np.asarray(target_p, dtype=float)
    y0 = p2c44.p2c24._seed_from_reference(np.asarray(y_ref, dtype=float), eta=1.0, base_scale=1.0)
    funcs = p2c44._make_functions(params, target_p, args, mode=mode)
    solve = p2c44._solve(y0, I0, funcs, args)
    load = p2c44.p2c41._load_audit(params, solve["y"], I0)
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    row = {
        "case": case_name,
        "alpha": alpha,
        "boundary_mode": mode,
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "max_slack_force_over_rigid_force": load["max_slack_force_over_rigid_force"],
        "max_slack_torque_over_rigid_torque": load["max_slack_torque_over_rigid_torque"],
        "steady_s": solve["steady_s"],
        "nfev": solve["nfev"],
    }
    return _trace_summary(params, solve["y"], row)


def _summarize(case_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    none = next(row for row in rows if row["boundary_mode"] == "none")
    full = next(row for row in rows if row["boundary_mode"] == "full")
    torque = next(row for row in rows if row["boundary_mode"] == "torque")
    return {
        "case": case_name,
        "terminal_torque_reduction_none_to_full": float(none["terminal_torque_norm"] / max(full["terminal_torque_norm"], 1.0e-30)),
        "slack_force_increase_none_to_full": float(full["max_slack_force_over_rigid_force"] / max(none["max_slack_force_over_rigid_force"], 1.0e-30)),
        "best_stationarity_mode": min(rows, key=lambda row: row["stationarity_norm"])["boundary_mode"],
        "best_stationarity_norm": min(row["stationarity_norm"] for row in rows),
        "min_slack_force_mode": min(rows, key=lambda row: row["max_slack_force_over_rigid_force"])["boundary_mode"],
        "min_slack_force_ratio": min(row["max_slack_force_over_rigid_force"] for row in rows),
        "torque_mode_slack_force_ratio": torque["max_slack_force_over_rigid_force"],
        "full_mode_slack_force_ratio": full["max_slack_force_over_rigid_force"],
        "mechanism": "末端内矩闭合与接口力松弛替代" if full["max_slack_force_over_rigid_force"] > none["max_slack_force_over_rigid_force"] else "接口力松弛未随内矩闭合升高",
    }


def _plot(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    mode_cn = {"none": "无边界", "force": "仅力", "torque": "仅矩", "full": "力矩全闭合"}
    for case in sorted({row["case"] for row in rows}):
        sub = [row for row in rows if row["case"] == case]
        x = np.arange(len(sub))
        labels = [mode_cn[row["boundary_mode"]] for row in sub]
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        specs = [
            ("terminal_torque_norm", "末端内矩残差", None),
            ("max_slack_force_over_rigid_force", "接口力松弛/刚段外力", 1.5),
            ("max_slack_force_over_natural_force_delta", "接口力松弛/刚段自然力变化", 1.5),
        ]
        for ax, (key, ylabel, gate) in zip(axes, specs):
            ax.bar(x, [row[key] for row in sub], color="#f97316")
            if gate is not None:
                ax.axhline(gate, color="black", ls="--", lw=1.0)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=10, ha="right")
        fig.suptitle(f"{case} 载荷耦合方程审计")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{case}_load_coupling_audit_cn.png", dpi=180)
        plt.close(fig)


def _record(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    table = [
        "| 工况 | 内矩残差降低倍数 | 接口力松弛升高倍数 | 最优一阶最优性模式 | 最小接口力模式 | 机制判断 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    mode_cn = {"none": "无边界", "force": "仅力", "torque": "仅矩", "full": "力矩全闭合"}
    for summary in summaries:
        table.append(
            f"| {summary['case']} | {summary['terminal_torque_reduction_none_to_full']:.3e} | "
            f"{summary['slack_force_increase_none_to_full']:.3e} | {mode_cn[summary['best_stationarity_mode']]} | "
            f"{mode_cn[summary['min_slack_force_mode']]} | {summary['mechanism']} |"
        )
    return "\n".join(
        [
            "# P2c45 载荷耦合方程审计记录",
            "",
            "## 实验目的",
            "",
            "P2c44 表明末端内矩闭合与接口载荷传播存在替代关系。本轮进一步追踪刚段自然力/矩变化、接口松弛量和末端边界残差之间的量级关系，判断下一步显式刚柔段载荷耦合模型应约束哪条物理链路。",
            "",
            "## 实验方案",
            "",
            "复用 P2c44 的四种边界模式，对每个求解结果调用刚段载荷传播审计。重点比较从无边界到力矩全闭合时，末端内矩残差是否下降，以及接口力松弛相对于刚段外力和刚段自然力变化是否升高。",
            "",
            "## 参数",
            "",
            f"- 工况：`{args.specs}`",
            f"- 边界模式：`{args.modes}`",
            f"- 边界权重：`{args.boundary_scale}`",
            f"- 接口松弛权重：`{args.slack_scale}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若力矩全闭合显著降低末端内矩残差，但同步抬高接口力松弛比例，则说明当前接口松弛变量正在承担末端内矩闭合所需的等效力通道。此时继续在松弛权重上扫参并不能形成物理可信灵敏度；应显式把刚段外力、力臂项和相邻柔性段近端内力内矩耦合起来。",
            "",
            "## 下一步计划",
            "",
            "1. 实现显式刚柔段载荷耦合残差：将刚段外力、外矩和力臂项传播到下一柔性段近端内力内矩，而不是允许任意接口跳变。",
            "2. 对单点、两点仅运行基线门禁；若仍失败，分别给出有效子空间或不可发布结论。",
            "3. 若基线门禁通过，再做 eps=0.005 和 0.010 的扰动重求解。",
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
            row = _run_case_mode(case_name, alpha, mode, args)
            rows.append(row)
            case_rows.append(row)
            _append_jsonl(failures, {"ts": time.time(), "case": case_name, "mode": mode, "failure": "load_coupling_diagnostic", "evidence": row})
        summaries.append(_summarize(case_name, case_rows))
    _plot(rows, summaries)
    _write_csv(OUT_DIR / "load_coupling_rows.csv", rows)
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
        "experiment_records_ik_optimize/20260607155409_p2c45_load_coupling_equation_audit/"
        "run_p2c45_load_coupling_equation_audit.py\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
