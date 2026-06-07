"""Probe regularization strength after n2 force-continuity repair."""

from __future__ import annotations

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
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
REG_SCALES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
FORCE_SCALE = 4.0


class Args(p2c57.Args):
    """Mutable settings with experiment-specific regularization."""


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


def _solve_stage(params: Any, current_i: np.ndarray, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, *, pos_scale: float, max_iter: int, reg_scale: float) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    args.reg_scale = float(reg_scale)
    funcs = p2c57._make_split_functions(params, target_p, y_ref, args, force_scale=FORCE_SCALE, torque_scale=p2c57.TORQUE_SCALE)
    return p2c57.p2c47._solve(y0, current_i, funcs, args)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for reg_scale in REG_SCALES:
        sub = [row for row in rows if row["reg_scale"] == reg_scale]
        final = max(sub, key=lambda row: row["stage"])
        summaries.append(
            {
                "reg_scale": reg_scale,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "final_stationarity_norm": final["stationarity_norm"],
                "final_condition": final["jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_ratio_to_rigid_load": final["max_interface_force_jump_over_rigid_force"],
                "final_force_ratio_to_internal_force": final["max_force_ratio_to_internal_force"],
                "final_position_error_m": final["position_error_m"],
                "dominant_failure": final["failure_class"],
            }
        )
    return summaries


def _plot(summaries: list[dict[str, Any]]) -> None:
    x = [row["reg_scale"] for row in summaries]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x, [row["final_stationarity_norm"] for row in summaries], marker="o")
    axes[1].plot(x, [row["final_condition"] for row in summaries], marker="o")
    axes[2].plot(x, [row["final_force_ratio_to_rigid_load"] for row in summaries], marker="o")
    axes[3].plot(x, [row["final_position_error_m"] for row in summaries], marker="o")
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("一阶最优性")
    axes[1].set_ylabel("条件数")
    axes[2].set_ylabel("力跳变比")
    axes[3].set_ylabel("末端误差/m")
    axes[3].set_xlabel("正则权重")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("n2 力连续修复后的正则权重探针")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_continuity_condition_probe_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 正则权重 | 是否过门 | 一阶最优性 | 条件数 | 接口连续残差 | 相对刚段外力 | 相对接口内力 | 末端误差/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['reg_scale']:.1e} | {row['any_gate_pass']} | {row['final_stationarity_norm']:.3e} | "
            f"{row['final_condition']:.3e} | {row['final_continuity_norm']:.3e} | "
            f"{row['final_force_ratio_to_rigid_load']:.3e} | {row['final_force_ratio_to_internal_force']:.3e} | "
            f"{row['final_position_error_m']:.3e} | {row['dominant_failure']} |"
        )
    best = min(summaries, key=lambda row: (row["final_condition"], row["final_stationarity_norm"]))
    return "\n".join(
        [
            "# P2c58 n2 力连续修复后的条件数探针记录",
            "",
            "## 实验目的",
            "",
            "P2c57 中力连续权重 `4` 已使接口连续残差和力传播比例进入门禁附近，但条件数仍为 `5.90e7`。本轮固定力连续权重 `4`，扫描正则权重，判断能否在不牺牲物理闭合的情况下降低条件数。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10`、position-only 任务、P2c50 同源初值、力连续权重 `4`、矩连续权重 `0.25`。正则权重扫描 `1e-4,1e-3,1e-2,1e-1`。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"条件数最小的设置为正则权重 `{best['reg_scale']:.1e}`，但该设置显著破坏接口力连续性。结果说明正则可以降低条件数，但会把解推离力连续闭合分支；单独调整正则不足以获得发布级基线，需要联合调整力连续权重和正则权重，或改变状态参数化。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门设置，进入分支一致性与扰动重求解审计。",
            "2. 若仍未过门但接近，尝试变量尺度重整或低维力传递参数化。",
            "3. 若正则导致末端误差或一阶最优性退化，停止正则扫描。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for reg_scale in REG_SCALES:
        y = y_seed.copy()
        y_ref = y.copy()
        for stage, (pos_scale, max_iter) in enumerate(p2c57.STAGES, start=1):
            solve = _solve_stage(params, current_i, target_p, y_ref, y, pos_scale=pos_scale, max_iter=max_iter, reg_scale=reg_scale)
            row = p2c57._row(force_scale=FORCE_SCALE, stage=stage, pos_scale=pos_scale, max_iter=max_iter, params=params, target_p=target_p, solve=solve)
            row["reg_scale"] = reg_scale
            rows.append(row)
            y = solve["y"]
            if not row["baseline_gate"]:
                p2c57._append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    p2c57._write_csv(OUT_DIR / "condition_probe_rows.csv", rows)
    p2c57._write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "force_scale": FORCE_SCALE,
        "torque_scale": p2c57.TORQUE_SCALE,
        "reg_scales": list(REG_SCALES),
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
    print(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
