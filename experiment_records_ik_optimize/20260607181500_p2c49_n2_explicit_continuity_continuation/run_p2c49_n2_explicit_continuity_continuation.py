"""Run n2 explicit-continuity continuation gates."""

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
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
FK_OPT_DIR = ROOT / "forward_kinematics_optimized"
for path in (ROOT, FK_OPT_DIR, P2C47_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c47_explicit_continuity_position_gate as p2c47

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent


class Args:
    """Mutable solve settings for staged continuation."""

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


def _schedules() -> dict[str, list[tuple[float, float]]]:
    return {
        "continuity_first": [(10.0, 16.0), (20.0, 16.0), (40.0, 4.0), (60.0, 0.25)],
        "balanced": [(10.0, 4.0), (20.0, 4.0), (40.0, 1.0), (60.0, 0.25)],
        "task_first": [(10.0, 0.25), (20.0, 0.25), (40.0, 0.25), (60.0, 0.25)],
    }


def _solve_stage(params: Any, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, pos_scale: float, cont_scale: float) -> dict[str, Any]:
    args = Args()
    args.position_scale = float(pos_scale)
    funcs = p2c47._make_functions(params, target_p, y_ref, args, continuity_scale=float(cont_scale), boundary_mode="full")
    return p2c47._solve(y0, CURRENT_I, funcs, args)


def _row_from_solve(
    *,
    schedule: str,
    stage: int,
    pos_scale: float,
    cont_scale: float,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
) -> dict[str, Any]:
    boundary = np.asarray(solve["out"]["boundary"], dtype=float).reshape(6)
    load = p2c47._interface_load_metrics(params, solve["y"], steps_per_flex=Args.steps_per_flex)
    row = {
        "case": "n2_z_pair_rand12",
        "alpha": 0.10,
        "schedule": schedule,
        "stage": stage,
        "position_scale": pos_scale,
        "continuity_scale": cont_scale,
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
    out = []
    for schedule in sorted({row["schedule"] for row in rows}):
        sub = [row for row in rows if row["schedule"] == schedule]
        final = max(sub, key=lambda row: row["stage"])
        best = min(sub, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_interface_force_jump_over_rigid_force"]))
        out.append(
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
    return out


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
    axes[2].set_xlabel("延拓阶段")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("两点工况显式接口连续延拓门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_explicit_continuity_continuation_cn.png", dpi=180)
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
            "# P2c49 两点显式接口连续延拓记录",
            "",
            "## 实验目的",
            "",
            "P2c47 中两点工况的一阶最优性相较自由接口松弛路径已有改善，但仍未过基线门禁。本轮使用分阶段延拓，检验更平滑的任务权重和接口连续权重路径是否能获得可信基线。",
            "",
            "## 实验方案",
            "",
            "固定工况 `n2_z_pair_rand12:0.10` 和 position-only 任务。采用 18 维分段内力/内矩状态，保留力矩全闭合边界。每条路径从 P2c22 保存的状态转换初值出发，逐阶段 warm start；只记录基线门禁，不做扰动灵敏度。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若延拓后仍不能同时满足一阶最优性、条件数、接口连续和载荷传播门禁，则说明两点工况当前工作点仍不能接入发布级三维末端位置灵敏度。若某条路径显著降低一阶最优性或接口力跳变，则该路径可作为下一轮更细延拓或分支库构建的入口。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若任一路径过门，则进入小扰动重求解复核。",
            "2. 若无路径过门但指标改善，细化该路径的延拓阶段。",
            "3. 若无明显改善，给出两点当前工作点不可发布边界，并转向更温和电流范围或磁体配置建议。",
            "",
        ]
    )


CURRENT_I: np.ndarray


def main() -> None:
    global CURRENT_I
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")

    params, CURRENT_I = p2c47.p2c24._params_from_saved_case("n2_z_pair_rand12", 0.10)
    target_p, y_slack = p2c47.p2c24._load_saved_target_and_seed("n2_z_pair_rand12", 0.10)
    y_seed = p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=Args.steps_per_flex)

    rows: list[dict[str, Any]] = []
    for schedule, stages in _schedules().items():
        y = y_seed.copy()
        for idx, (pos_scale, cont_scale) in enumerate(stages, start=1):
            solve = _solve_stage(params, target_p, y_seed, y, pos_scale, cont_scale)
            row = _row_from_solve(
                schedule=schedule,
                stage=idx,
                pos_scale=pos_scale,
                cont_scale=cont_scale,
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
    _write_csv(OUT_DIR / "continuation_rows.csv", rows)
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
