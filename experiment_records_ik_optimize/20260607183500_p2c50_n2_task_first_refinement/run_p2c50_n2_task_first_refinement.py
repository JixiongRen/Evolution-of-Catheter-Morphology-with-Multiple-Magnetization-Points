"""Refine the n2 task-first explicit-continuity path."""

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
P2C49_DIR = ROOT / "experiment_records_ik_optimize" / "20260607181500_p2c49_n2_explicit_continuity_continuation"
for path in (ROOT, P2C49_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c49_n2_explicit_continuity_continuation as p2c49

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


class Args(p2c49.Args):
    """Solve settings for refined stages."""

    max_iter = 160


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


def _schedules() -> dict[str, list[tuple[float, float, int]]]:
    return {
        "task_first_fine": [
            (10.0, 0.25, 120),
            (20.0, 0.25, 120),
            (30.0, 0.25, 120),
            (40.0, 0.25, 120),
            (50.0, 0.25, 120),
            (60.0, 0.25, 240),
        ],
        "task_first_low_cont": [
            (10.0, 0.25, 120),
            (20.0, 0.25, 120),
            (40.0, 0.25, 120),
            (60.0, 0.25, 160),
            (60.0, 0.10, 240),
        ],
        "task_first_high_cont_finish": [
            (10.0, 0.25, 120),
            (20.0, 0.25, 120),
            (40.0, 0.25, 120),
            (60.0, 0.25, 160),
            (60.0, 0.50, 240),
        ],
    }


def _solve_stage(params: Any, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, pos_scale: float, cont_scale: float, max_iter: int) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    args.max_iter = int(max_iter)
    funcs = p2c49.p2c47._make_functions(params, target_p, y_ref, args, continuity_scale=float(cont_scale), boundary_mode="full")
    return p2c49.p2c47._solve(y0, p2c49.CURRENT_I, funcs, args)


def _row_from_solve(
    *,
    schedule: str,
    stage: int,
    pos_scale: float,
    cont_scale: float,
    max_iter: int,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
) -> dict[str, Any]:
    row = p2c49._row_from_solve(
        schedule=schedule,
        stage=stage,
        pos_scale=pos_scale,
        cont_scale=cont_scale,
        params=params,
        target_p=target_p,
        solve=solve,
    )
    row["max_iter"] = max_iter
    return row


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for schedule in sorted({row["schedule"] for row in rows}):
        sub = [row for row in rows if row["schedule"] == schedule]
        final = max(sub, key=lambda row: row["stage"])
        best = min(sub, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_interface_force_jump_over_rigid_force"]))
        summaries.append(
            {
                "schedule": schedule,
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "final_stationarity_norm": final["stationarity_norm"],
                "final_condition": final["jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_jump_ratio": final["max_interface_force_jump_over_rigid_force"],
                "final_position_error_m": final["position_error_m"],
                "best_stage": best["stage"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_force_jump_ratio": best["max_interface_force_jump_over_rigid_force"],
                "dominant_failure": final["failure_class"],
            }
        )
    return summaries


def _plot(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for schedule in sorted({row["schedule"] for row in rows}):
        sub = sorted([row for row in rows if row["schedule"] == schedule], key=lambda row: row["stage"])
        x = [row["stage"] for row in sub]
        axes[0].plot(x, [row["stationarity_norm"] for row in sub], marker="o", label=schedule)
        axes[1].plot(x, [row["continuity_norm"] for row in sub], marker="o", label=schedule)
        axes[2].plot(x, [row["max_interface_force_jump_over_rigid_force"] for row in sub], marker="o", label=schedule)
    axes[0].axhline(1.0e-5, color="black", ls="--", lw=1.0)
    axes[1].axhline(1.0e-3, color="black", ls="--", lw=1.0)
    axes[2].axhline(1.5, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("一阶最优性残差")
    axes[1].set_ylabel("接口连续残差")
    axes[2].set_ylabel("接口力跳变比")
    axes[2].set_xlabel("细化阶段")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("两点工况 task-first 细化延拓")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_task_first_refinement_cn.png", dpi=180)
    plt.close(fig)


def _record(summaries: list[dict[str, Any]], wall_s: float) -> str:
    table = [
        "| 路径 | 是否过门 | 末阶段一阶最优性 | 末阶段条件数 | 末阶段接口连续残差 | 末阶段接口力跳变比 | 最优阶段 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['schedule']} | {row['any_gate_pass']} | {row['final_stationarity_norm']:.3e} | "
            f"{row['final_condition']:.3e} | {row['final_continuity_norm']:.3e} | "
            f"{row['final_force_jump_ratio']:.3e} | {row['best_stage']} | {row['dominant_failure']} |"
        )
    return "\n".join(
        [
            "# P2c50 两点 task-first 细化延拓记录",
            "",
            "## 实验目的",
            "",
            "P2c49 显示 `task_first` 路径可显著改善两点工况的一阶最优性和接口力跳变。本轮围绕该路径做细化阶段、长迭代精修和末端接口连续权重探针，判断是否能进一步接近发布门禁。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12:0.10`、position-only 任务、力矩全闭合边界和 18 维分段内力/内矩状态。比较三条路径：更细任务权重递增、低接口连续权重末端精修、高接口连续权重末端精修。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若细化后仍无法通过门禁，则说明当前两点工作点不能靠同一路径继续温和修复；应转向更温和电流范围或磁体配置边界，而不是继续扩大同类延拓扫描。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若过门，进入小扰动重求解。",
            "2. 若只改善未过门，记录当前路径为诊断分支，不发布。",
            "3. 若无实质改善，转入两点电流幅值边界扫描，寻找可发布工作空间。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")

    params, p2c49.CURRENT_I = p2c49.p2c47.p2c24._params_from_saved_case("n2_z_pair_rand12", 0.10)
    target_p, y_slack = p2c49.p2c47.p2c24._load_saved_target_and_seed("n2_z_pair_rand12", 0.10)
    y_seed = p2c49.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for schedule, stages in _schedules().items():
        y = y_seed.copy()
        for idx, (pos_scale, cont_scale, max_iter) in enumerate(stages, start=1):
            solve = _solve_stage(params, target_p, y_seed, y, pos_scale, cont_scale, max_iter)
            row = _row_from_solve(
                schedule=schedule,
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
    _plot(rows)
    _write_csv(OUT_DIR / "refinement_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
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
