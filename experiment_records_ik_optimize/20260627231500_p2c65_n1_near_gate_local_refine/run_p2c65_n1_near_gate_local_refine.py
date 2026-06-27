"""Refine the near-gate n1 force-continuity baseline."""

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
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n1_tip_z_rand15"
ALPHA = 0.10
FORCE_SCALES = (3.0, 4.0, 5.0)
REG_SCALES = (6.0e-4, 8.0e-4, 1.2e-3)
STAGES = p2c57.STAGES


class Args(p2c57.Args):
    """Mutable settings for the n1 physical-closure probe."""


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


def _solve_stage(params: Any, current_i: np.ndarray, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, *, pos_scale: float, max_iter: int, force_scale: float, reg_scale: float) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    args.reg_scale = float(reg_scale)
    funcs = p2c57._make_split_functions(
        params,
        target_p,
        y_ref,
        args,
        force_scale=force_scale,
        torque_scale=p2c57.TORQUE_SCALE,
    )
    return p2c57.p2c47._solve(y0, current_i, funcs, args)


def _row(*, force_scale: float, reg_scale: float, stage: int, pos_scale: float, max_iter: int, params: Any, target_p: np.ndarray, solve: dict[str, Any]) -> dict[str, Any]:
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c57.p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
    combined = p2c57._combined_force_metrics(params, solve["y"])
    row = {
        "case": CASE_NAME,
        "alpha": ALPHA,
        "force_continuity_scale": float(force_scale),
        "reg_scale": float(reg_scale),
        "torque_continuity_scale": p2c57.TORQUE_SCALE,
        "stage": int(stage),
        "position_scale": float(pos_scale),
        "max_iter": int(max_iter),
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        **load,
        **combined,
    }
    row["baseline_gate"] = bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["continuity_norm"] <= 1.0e-3
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
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "接口力连续性失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for force_scale in FORCE_SCALES:
        for reg_scale in REG_SCALES:
            sub = [row for row in rows if row["force_continuity_scale"] == force_scale and row["reg_scale"] == reg_scale]
            final = max(sub, key=lambda row: row["stage"])
            score = (
                max(0.0, np.log10(final["stationarity_norm"] / 1.0e-5))
                + max(0.0, np.log10(final["jacobian_condition"] / 1.0e7))
                + max(0.0, np.log10(final["continuity_norm"] / 1.0e-3))
                + max(0.0, np.log10(final["max_interface_force_jump_over_rigid_force"] / 1.5))
            )
            summaries.append(
                {
                    "force_continuity_scale": force_scale,
                    "reg_scale": reg_scale,
                    "any_gate_pass": any(row["baseline_gate"] for row in sub),
                    "gate_violation_score": float(score),
                    "final_position_error_m": final["position_error_m"],
                    "final_stationarity_norm": final["stationarity_norm"],
                    "final_condition": final["jacobian_condition"],
                    "final_continuity_norm": final["continuity_norm"],
                    "final_force_ratio_to_rigid_load": final["max_interface_force_jump_over_rigid_force"],
                    "final_force_ratio_to_internal_force": final["max_force_ratio_to_internal_force"],
                    "dominant_failure": final["failure_class"],
                }
            )
    return summaries


def _plot(summaries: list[dict[str, Any]]) -> None:
    labels = [f"F{row['force_continuity_scale']:.0f}\\nR{row['reg_scale']:.0e}" for row in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(4, 1, figsize=(max(10, 0.7 * len(labels)), 11), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "final_stationarity_norm", "一阶最优性"),
        (axes[1], "final_condition", "条件数"),
        (axes[2], "final_continuity_norm", "接口连续残差"),
        (axes[3], "final_force_ratio_to_rigid_load", "相对刚段外力"),
    ]:
        ax.bar(x, [row[key] for row in summaries])
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[-1].set_xticks(x, labels, rotation=0)
    fig.suptitle("n1 近门力连续与正则局部细化")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_near_gate_local_refine_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 力权重 | 正则权重 | 是否过门 | 评分 | 末端误差/m | 一阶最优性 | 条件数 | 接口连续 | 相对刚段外力 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['force_continuity_scale']:.0f} | {row['reg_scale']:.0e} | {row['any_gate_pass']} | {row['gate_violation_score']:.3e} | "
            f"{row['final_position_error_m']:.3e} | {row['final_stationarity_norm']:.3e} | {row['final_condition']:.3e} | "
            f"{row['final_continuity_norm']:.3e} | {row['final_force_ratio_to_rigid_load']:.3e} | {row['dominant_failure']} |"
        )
    best = min(summaries, key=lambda row: row["gate_violation_score"])
    return "\n".join(
        [
            "# P2c65 单点近门力连续与正则局部细化记录",
            "",
            "## 实验目的",
            "",
            "P2c64 已找到单点近门组合：力权重 `4`、正则 `1e-3`，一阶最优性和条件数已过门，剩余失败集中在接口连续和载荷传播。本轮只围绕该近门点做小范围细化。",
            "",
            "## 实验方案",
            "",
            "固定 `n1_tip_z_rand15, alpha=0.10`、position-only 任务和 P2c47 显式分段内力模型。扫描力连续权重 `3,4,5` 与正则权重 `6e-4,8e-4,1.2e-3`，每组使用相同分阶段位置权重。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            (
                f"最低门禁违背评分组合为力权重 `{best['force_continuity_scale']:.0f}`、"
                f"正则 `{best['reg_scale']:.0e}`。该组合的一阶最优性为 "
                f"`{best['final_stationarity_norm']:.2e}`，条件数为 `{best['final_condition']:.2e}`，"
                f"末端误差为 `{best['final_position_error_m']:.2e} m`，但接口连续误差为 "
                f"`{best['final_continuity_norm']:.2e}`，相对刚段外力跳变为 "
                f"`{best['final_force_ratio_to_rigid_load']:.2f}`。局部细化没有突破载荷传播瓶颈，"
                "单点模型当前仍不能进入扰动灵敏度审计。"
            ),
            "",
            "多个数值可行组合都停留在相近的接口连续误差和相对刚段外力跳变范围内，说明当前失败更可能来自模型耦合或残差定义，而不是力连续权重和正则权重的局部调参不足。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 停止继续围绕力连续权重和正则权重做局部细化。",
            "2. 优先复核显式刚段外力传播公式和接口载荷连续门禁定义。",
            "3. 若门禁定义无误，则将该现象归入单点模型当前不可发布证据链，并转向更稳定的物理参数化方式。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, current_i = p2c57.p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for force_scale in FORCE_SCALES:
        for reg_scale in REG_SCALES:
            y = y_seed.copy()
            y_ref = y.copy()
            for stage, (pos_scale, max_iter) in enumerate(STAGES, start=1):
                solve = _solve_stage(params, current_i, target_p, y_ref, y, pos_scale=pos_scale, max_iter=max_iter, force_scale=force_scale, reg_scale=reg_scale)
                row = _row(force_scale=force_scale, reg_scale=reg_scale, stage=stage, pos_scale=pos_scale, max_iter=max_iter, params=params, target_p=target_p, solve=solve)
                rows.append(row)
                y = solve["y"]
                if not row["baseline_gate"]:
                    _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})
    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    _write_csv(OUT_DIR / "n1_force_reg_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
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
    print(json.dumps(_jsonable({"wall_s": wall_s, "summaries": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
