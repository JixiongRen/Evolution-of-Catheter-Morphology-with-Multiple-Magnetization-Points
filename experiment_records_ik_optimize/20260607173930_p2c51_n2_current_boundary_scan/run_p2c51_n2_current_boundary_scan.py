"""Scan current-amplitude boundaries for the n2 position shooting gate."""

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
P2C10T_DIR = ROOT / "experiment_records_ik_optimize" / "20260605102000_p2c10t_interface_slack_shooting"
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
for path in (ROOT, P2C10T_DIR, P2C47_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c10t_interface_slack_shooting as p2c10t
import run_p2c47_explicit_continuity_position_gate as p2c47

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n2_z_pair_rand12"
ALPHAS = (0.02, 0.05, 0.08, 0.10)
STAGES = (
    (10.0, 0.25, 120),
    (20.0, 0.25, 120),
    (30.0, 0.25, 120),
    (40.0, 0.25, 120),
    (50.0, 0.25, 120),
    (60.0, 0.25, 240),
)


class Args:
    """Solve settings for the explicit-continuity boundary scan."""

    boundary_scale = 0.25
    reg_scale = 1.0e-4
    max_iter = 120
    steps_per_flex = 4
    position_scale = 60.0


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


def _solve_stage(
    params: Any,
    current_i: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y0: np.ndarray,
    *,
    pos_scale: float,
    cont_scale: float,
    max_iter: int,
) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    funcs = p2c47._make_functions(
        params,
        target_p,
        y_ref,
        args,
        continuity_scale=float(cont_scale),
        boundary_mode="full",
    )
    return p2c47._solve(y0, current_i, funcs, args)


def _row_from_solve(
    *,
    alpha: float,
    stage: int,
    pos_scale: float,
    cont_scale: float,
    max_iter: int,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
) -> dict[str, Any]:
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
    row = {
        "case": CASE_NAME,
        "alpha": float(alpha),
        "stage": int(stage),
        "position_scale": float(pos_scale),
        "continuity_scale": float(cont_scale),
        "max_iter": int(max_iter),
        "position_error_m": float(np.linalg.norm(np.asarray(solve["out"]["tip_p"], dtype=float) - target_p)),
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "continuity_norm": solve["continuity_norm"],
        "terminal_force_norm": float(np.linalg.norm(boundary[:3])),
        "terminal_torque_norm": float(np.linalg.norm(boundary[3:])),
        "steady_s": solve["steady_s"],
        "nfev": solve["nfev"],
        "success": solve["success"],
        **load,
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
    elif row["continuity_norm"] > 1.0e-3 or row["max_interface_force_jump_over_rigid_force"] > 1.5:
        row["failure_class"] = "刚性磁体段载荷传播失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for alpha in sorted({row["alpha"] for row in rows}):
        sub = [row for row in rows if row["alpha"] == alpha]
        final = max(sub, key=lambda row: row["stage"])
        best = min(
            sub,
            key=lambda row: (
                not row["baseline_gate"],
                row["stationarity_norm"],
                row["max_interface_force_jump_over_rigid_force"],
            ),
        )
        summaries.append(
            {
                "case": CASE_NAME,
                "alpha": alpha,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "final_stage": final["stage"],
                "final_position_error_m": final["position_error_m"],
                "final_stationarity_norm": final["stationarity_norm"],
                "final_condition": final["jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_jump_ratio": final["max_interface_force_jump_over_rigid_force"],
                "final_terminal_force_norm": final["terminal_force_norm"],
                "best_stage": best["stage"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_force_jump_ratio": best["max_interface_force_jump_over_rigid_force"],
                "dominant_failure": final["failure_class"],
            }
        )
    return summaries


def _plot(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for alpha in sorted({row["alpha"] for row in rows}):
        sub = sorted([row for row in rows if row["alpha"] == alpha], key=lambda row: row["stage"])
        x = [row["stage"] for row in sub]
        label = f"alpha={alpha:.2f}"
        axes[0].plot(x, [row["stationarity_norm"] for row in sub], marker="o", label=label)
        axes[1].plot(x, [row["jacobian_condition"] for row in sub], marker="o", label=label)
        axes[2].plot(x, [row["continuity_norm"] for row in sub], marker="o", label=label)
        axes[3].plot(x, [row["max_interface_force_jump_over_rigid_force"] for row in sub], marker="o", label=label)
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("一阶最优性残差")
    axes[1].set_ylabel("条件数")
    axes[2].set_ylabel("接口连续残差")
    axes[3].set_ylabel("接口力跳变比")
    axes[3].set_xlabel("延拓阶段")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("两点磁控电流幅值边界扫描")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_current_boundary_stage_metrics_cn.png", dpi=180)
    plt.close(fig)

    x = [row["alpha"] for row in summaries]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x, [row["final_stationarity_norm"] for row in summaries], marker="o")
    axes[1].plot(x, [row["final_condition"] for row in summaries], marker="o")
    axes[2].plot(x, [row["final_continuity_norm"] for row in summaries], marker="o")
    axes[3].plot(x, [row["final_force_jump_ratio"] for row in summaries], marker="o")
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e7, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[3].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("末阶段一阶最优性")
    axes[1].set_ylabel("末阶段条件数")
    axes[2].set_ylabel("末阶段接口连续")
    axes[3].set_ylabel("末阶段力跳变比")
    axes[3].set_xlabel("电流幅值系数 alpha")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("电流幅值对两点基线门禁的影响")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_current_boundary_alpha_summary_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| alpha | 是否过门 | 一阶最优性 | 条件数 | 接口连续残差 | 接口力跳变比 | 末端误差/m | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['alpha']:.2f} | {row['any_gate_pass']} | {row['final_stationarity_norm']:.3e} | "
            f"{row['final_condition']:.3e} | {row['final_continuity_norm']:.3e} | "
            f"{row['final_force_jump_ratio']:.3e} | {row['final_position_error_m']:.3e} | {row['dominant_failure']} |"
        )
    return "\n".join(
        [
            "# P2c51 两点电流幅值边界扫描记录",
            "",
            "## 实验目的",
            "",
            "P2c50 表明两点工况在 alpha=0.10 下已经能达到很小的一阶最优性残差，但条件数、接口连续残差和刚性磁体段载荷传播比例仍未通过发布门禁。本轮降低电流幅值，判断失败是否主要来自强加载造成的载荷传播失配，还是当前物理打靶表述本身存在结构性不闭合。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12`、position-only 任务、18 维分段内力/内矩状态、力矩全闭合边界、显式接口连续残差和 P2c50 的 task-first 细化延拓路径。对 `alpha=0.02,0.05,0.08,0.10` 分别重新生成 FK 目标和分段初值，不复用 alpha=0.10 的目标。",
            "",
            "## 门禁",
            "",
            "发布级基线需要同时满足：一阶最优性残差不大于 `1e-5`，条件数小于 `1e7`，接口连续残差不大于 `1e-3`，接口力跳变比不大于 `1.5`。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若低 alpha 仍不能降低接口力跳变比和条件数，则两点工况失败不能简单归因于电流过大，而应转向刚性磁体段载荷传播模型、接口连续约束尺度或任务可辨识性修复。若存在过门 alpha，则该幅值附近可进入扰动重求解灵敏度审计。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 对过门或最接近过门的 alpha 做小扰动重求解。",
            "2. 若全部失败且失败模式一致，停止同类 alpha 扫描，转入载荷传播模型修复。",
            "3. 将本轮结果纳入两点磁控工作区边界图。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        params, current_i, target_p, _target_q, y_seed, _oracle = p2c10t._build_case(CASE_NAME, alpha)
        y = np.asarray(y_seed, dtype=float).reshape(-1).copy()
        y_ref = y.copy()
        for idx, (pos_scale, cont_scale, max_iter) in enumerate(STAGES, start=1):
            solve = _solve_stage(
                params,
                current_i,
                target_p,
                y_ref,
                y,
                pos_scale=pos_scale,
                cont_scale=cont_scale,
                max_iter=max_iter,
            )
            row = _row_from_solve(
                alpha=alpha,
                stage=idx,
                pos_scale=pos_scale,
                cont_scale=cont_scale,
                max_iter=max_iter,
                params=params,
                target_p=target_p,
                solve=solve,
            )
            rows.append(row)
            y = solve["y"]
            if not row["baseline_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summaries(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(rows, summaries)
    _write_csv(OUT_DIR / "boundary_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alphas": list(ALPHAS),
        "stages": list(STAGES),
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
