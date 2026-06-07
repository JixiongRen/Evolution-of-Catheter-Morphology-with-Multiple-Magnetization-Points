"""Run a local force-continuity and regularization grid for n2 shooting."""

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
P2C58_DIR = ROOT / "experiment_records_ik_optimize" / "20260607202500_p2c58_force_continuity_condition_probe"
for path in (ROOT, P2C57_DIR, P2C58_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c58_force_continuity_condition_probe as p2c58

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
FORCE_SCALES = (4.0, 8.0, 12.0)
REG_SCALES = (3.0e-4, 1.0e-3, 3.0e-3)
STAGES = p2c57.STAGES


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


def _solve_stage(params: Any, current_i: np.ndarray, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, *, pos_scale: float, max_iter: int, force_scale: float, reg_scale: float) -> dict[str, Any]:
    args = p2c58.Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    args.reg_scale = float(reg_scale)
    funcs = p2c57._make_split_functions(params, target_p, y_ref, args, force_scale=force_scale, torque_scale=p2c57.TORQUE_SCALE)
    return p2c57.p2c47._solve(y0, current_i, funcs, args)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for force_scale in FORCE_SCALES:
        for reg_scale in REG_SCALES:
            sub = [row for row in rows if row["force_continuity_scale"] == force_scale and row["reg_scale"] == reg_scale]
            final = max(sub, key=lambda row: row["stage"])
            score = (
                max(0.0, np.log10(final["jacobian_condition"] / 1.0e7))
                + max(0.0, np.log10(final["max_interface_force_jump_over_rigid_force"] / 1.5))
                + max(0.0, np.log10(final["continuity_norm"] / 1.0e-3))
                + max(0.0, np.log10(final["stationarity_norm"] / 1.0e-5))
            )
            summaries.append(
                {
                    "force_continuity_scale": force_scale,
                    "reg_scale": reg_scale,
                    "any_gate_pass": any(row["baseline_gate"] for row in sub),
                    "gate_violation_score": float(score),
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
    force_vals = list(FORCE_SCALES)
    reg_vals = list(REG_SCALES)
    grids = {
        "gate_violation_score": np.zeros((len(force_vals), len(reg_vals))),
        "final_condition": np.zeros((len(force_vals), len(reg_vals))),
        "final_force_ratio_to_rigid_load": np.zeros((len(force_vals), len(reg_vals))),
    }
    for i, force_scale in enumerate(force_vals):
        for j, reg_scale in enumerate(reg_vals):
            row = next(r for r in summaries if r["force_continuity_scale"] == force_scale and r["reg_scale"] == reg_scale)
            for key in grids:
                grids[key][i, j] = row[key]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["门禁违背评分", "条件数", "相对刚段外力"]
    for ax, key, title in zip(axes, grids, titles):
        data = np.log10(np.maximum(grids[key], 1.0e-30))
        im = ax.imshow(data, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xticks(range(len(reg_vals)), [f"{v:.0e}" for v in reg_vals])
        ax.set_yticks(range(len(force_vals)), [f"{v:.0f}" for v in force_vals])
        ax.set_xlabel("正则权重")
        ax.set_ylabel("力连续权重")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("n2 力连续与正则局部网格")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_reg_local_grid_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 力权重 | 正则权重 | 是否过门 | 门禁违背评分 | 一阶最优性 | 条件数 | 接口连续 | 相对刚段外力 | 末端误差/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['force_continuity_scale']:.0f} | {row['reg_scale']:.0e} | {row['any_gate_pass']} | {row['gate_violation_score']:.3e} | "
            f"{row['final_stationarity_norm']:.3e} | {row['final_condition']:.3e} | {row['final_continuity_norm']:.3e} | "
            f"{row['final_force_ratio_to_rigid_load']:.3e} | {row['final_position_error_m']:.3e} | {row['dominant_failure']} |"
        )
    best = min(summaries, key=lambda row: row["gate_violation_score"])
    return "\n".join(
        [
            "# P2c59 n2 力连续与正则局部网格记录",
            "",
            "## 实验目的",
            "",
            "P2c57 显示力连续权重 `4` 能修复载荷传播但条件数偏高；P2c58 显示正则可降低条件数但破坏力连续。本轮做小范围二维网格，寻找二者折中点。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12, alpha=0.10`、position-only 任务、P2c50 同源初值。扫描力连续权重 `4,8,12` 与正则权重 `3e-4,1e-3,3e-3`。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"门禁违背评分最低的组合为力连续权重 `{best['force_continuity_scale']:.0f}`、正则权重 `{best['reg_scale']:.0e}`。若该组合仍未过门，则当前残差尺度调节只能形成诊断候选，不能发布。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门组合，进入分支一致性与扰动重求解审计。",
            "2. 若无过门但出现近门组合，围绕该点做变量尺度重整。",
            "3. 若网格整体退化，停止权重扫描，转入低维力传递参数化。",
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
    for force_scale in FORCE_SCALES:
        for reg_scale in REG_SCALES:
            y = y_seed.copy()
            y_ref = y.copy()
            for stage, (pos_scale, max_iter) in enumerate(STAGES, start=1):
                solve = _solve_stage(params, current_i, target_p, y_ref, y, pos_scale=pos_scale, max_iter=max_iter, force_scale=force_scale, reg_scale=reg_scale)
                row = p2c57._row(force_scale=force_scale, stage=stage, pos_scale=pos_scale, max_iter=max_iter, params=params, target_p=target_p, solve=solve)
                row["reg_scale"] = reg_scale
                rows.append(row)
                y = solve["y"]
                if not row["baseline_gate"]:
                    p2c57._append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    p2c57._write_csv(OUT_DIR / "force_reg_grid_rows.csv", rows)
    p2c57._write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": p2c57.CASE_NAME,
        "alpha": p2c57.ALPHA,
        "force_scales": list(FORCE_SCALES),
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
