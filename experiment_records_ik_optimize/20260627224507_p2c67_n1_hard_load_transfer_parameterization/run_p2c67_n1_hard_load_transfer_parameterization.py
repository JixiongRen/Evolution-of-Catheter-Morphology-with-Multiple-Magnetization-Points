"""Run n1 hard load-transfer shooting as a physical-closure test."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C61_DIR = ROOT / "experiment_records_ik_optimize" / "20260627210200_p2c61_lowdim_load_transfer_parameterization"
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
for path in (ROOT, P2C61_DIR, P2C57_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c61_lowdim_load_transfer_parameterization as p2c61

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n1_tip_z_rand15"
ALPHA = 0.10
POSITION_SCALES = (20.0, 40.0, 60.0)
INIT_SCALES = {"zero": 0.0, "half_base": 0.5, "oracle_base": 1.0}
LENGTH_M = 0.11


def _jsonable(value: Any) -> Any:
    return p2c61._jsonable(value)


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


def _row(
    *,
    init_name: str,
    position_scale: float,
    params: Any,
    target_p: np.ndarray,
    solve: dict[str, Any],
) -> dict[str, Any]:
    out = solve["out"]
    load = p2c61._interface_metrics_from_blocks(params, out["blocks"])
    position_error = float(np.linalg.norm(out["tip_p"] - target_p))
    row = {
        "case": CASE_NAME,
        "alpha": ALPHA,
        "mode": "hard_load_transfer_6d",
        "init_name": init_name,
        "position_scale": float(position_scale),
        "position_error_m": position_error,
        "position_error_fraction_length": position_error / LENGTH_M,
        "stationarity_norm": solve["stationarity_norm"],
        "jacobian_condition": solve["jacobian_condition"],
        "jacobian_rank": solve["jacobian_rank"],
        "boundary_norm": float(np.linalg.norm(out["boundary"])),
        "continuity_norm": 0.0,
        "nfev": solve["nfev"],
        "steady_s": solve["steady_s"],
        "success": solve["success"],
        "message": solve["message"],
        **load,
    }
    row["publish_gate"] = bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["jacobian_condition"] < 1.0e7
        and row["position_error_fraction_length"] <= 0.05
        and row["max_force_ratio_to_rigid_load"] <= 1.5
    )
    if row["publish_gate"]:
        row["failure_class"] = "通过"
    elif row["stationarity_norm"] > 1.0e-5:
        row["failure_class"] = "一阶最优性退化"
    elif row["jacobian_condition"] >= 1.0e9:
        row["failure_class"] = "条件数严重病态"
    elif row["jacobian_condition"] >= 1.0e7:
        row["failure_class"] = "条件数偏高"
    elif row["position_error_fraction_length"] > 0.05:
        row["failure_class"] = "任务残差不可达"
    elif row["max_force_ratio_to_rigid_load"] > 1.5:
        row["failure_class"] = "刚性磁体段载荷传播失配"
    else:
        row["failure_class"] = "综合失败"
    return row


def _plot(rows: list[dict[str, Any]], best_solve: dict[str, Any], oracle_cl: np.ndarray) -> None:
    labels = [f"{row['init_name']}\nP{row['position_scale']:.0f}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(4, 1, figsize=(max(10, 0.8 * len(labels)), 11), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "position_error_m", "末端位置误差(m)"),
        (axes[1], "stationarity_norm", "一阶最优性"),
        (axes[2], "jacobian_condition", "条件数"),
        (axes[3], "boundary_norm", "终端边界残差"),
    ]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values) & (values > 0.0)]
        cap = float(np.max(finite) * 10.0) if finite.size else 1.0
        values = np.nan_to_num(values, nan=cap, posinf=cap, neginf=cap)
        ax.bar(x, values)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes[1].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(1.0e7, color="black", linestyle="--", linewidth=1.0)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    fig.suptitle("P2c67 单点硬载荷传播参数化门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_hard_load_transfer_metrics_cn.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(oracle_cl[:, 0], oracle_cl[:, 1], oracle_cl[:, 2], "k-", lw=2.2, label="独立分段近门参照")
    cl = best_solve["out"]["centerline"]
    ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], "r--", lw=2.0, label="硬载荷传播最优")
    ax.scatter([oracle_cl[0, 0]], [oracle_cl[0, 1]], [oracle_cl[0, 2]], c="blue", s=45, label="基端")
    ax.scatter([cl[-1, 0]], [cl[-1, 1]], [cl[-1, 2]], c="red", s=45, label="硬传播末端")
    ax.set_title("P2c67 单点硬载荷传播姿态对比")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_hard_load_transfer_centerline_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]], wall_s: float) -> str:
    best = min(
        rows,
        key=lambda row: (
            not row["publish_gate"],
            row["stationarity_norm"],
            row["position_error_m"],
            row["jacobian_condition"],
        ),
    )
    table = [
        "| 初值 | 位置权重 | 是否过门 | 末端误差/m | 一阶最优性 | 条件数 | 秩 | 边界残差 | 失败分类 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['init_name']} | {row['position_scale']:.0f} | {row['publish_gate']} | "
            f"{row['position_error_m']:.3e} | {row['stationarity_norm']:.3e} | "
            f"{row['jacobian_condition']:.3e} | {row['jacobian_rank']} | {row['boundary_norm']:.3e} | {row['failure_class']} |"
        )
    return "\n".join(
        [
            "# P2c67 单点硬载荷传播参数化记录",
            "",
            "## 实验目的",
            "",
            "P2c66 显示单点近门样本的主要问题不是力连续权重局部调参不足，而是独立分段内力变量与刚段外载传播之间存在闭合缺口。本轮将相邻柔段初始内力显式设为前一段经过刚段传播后的结果，只保留基端 6 维内力/内矩变量，检验硬物理闭合是否仍能完成末端位置任务。",
            "",
            "## 实验方案",
            "",
            "固定 `n1_tip_z_rand15, alpha=0.10` 和 position-only 任务。采用 6 维硬载荷传播参数化，扫描 `zero`、`half_base`、`oracle_base` 三个初值和位置权重 `20,40,60`。接口载荷跳变由参数化消元，不再作为软惩罚残差。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"最优样本为 `{best['init_name']}`、位置权重 `{best['position_scale']:.0f}`。其末端误差为 `{best['position_error_m']:.3e} m`，一阶最优性为 `{best['stationarity_norm']:.3e}`，条件数为 `{best['jacobian_condition']:.3e}`，边界残差为 `{best['boundary_norm']:.3e}`。",
            "",
            "若该样本不过门，说明单点 position-only 任务在硬载荷传播约束下仍缺少可发布证据；此前软约束近门结果更可能依赖独立分段内力自由度，而非稳定物理解。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若硬传播样本过门，进入多初值分支一致性和小扰动重求解。",
            "2. 若末端任务不可达或一阶最优性退化，形成单点当前模型下硬闭合不可发布证据。",
            "3. 若硬传播接近过门但边界残差主导，复核终端边界残差尺度和任务权重。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    params, _current_i = p2c57.p2c24._params_from_saved_case(CASE_NAME, ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(CASE_NAME, ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c61.STEPS_PER_FLEX)
    oracle_cl = np.asarray(p2c57.forward_segment_wrench(params, jnp.asarray(y_seed), steps_per_flex=p2c61.STEPS_PER_FLEX).centerline)

    rows: list[dict[str, Any]] = []
    solves: dict[str, dict[str, Any]] = {}
    for init_name, scale in INIT_SCALES.items():
        u0 = scale * np.asarray(y_seed[:6], dtype=float)
        for position_scale in POSITION_SCALES:
            funcs = p2c61._make_functions(params, target_p, mode="base_6d", position_scale=position_scale, slack_reg_scale=0.0)
            solve = p2c61._solve(u0, funcs)
            row = _row(init_name=init_name, position_scale=position_scale, params=params, target_p=target_p, solve=solve)
            rows.append(row)
            solves[f"{init_name}_p{position_scale:.0f}"] = solve
            if not row["publish_gate"]:
                _append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    best = min(rows, key=lambda row: (not row["publish_gate"], row["stationarity_norm"], row["position_error_m"], row["jacobian_condition"]))
    best_solve = solves[f"{best['init_name']}_p{best['position_scale']:.0f}"]
    _plot(rows, best_solve, oracle_cl)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "hard_load_transfer_rows.csv", rows)
    _write_csv(OUT_DIR / "summary.csv", [best])
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alpha": ALPHA,
        "mode": "hard_load_transfer_6d",
        "position_scales": list(POSITION_SCALES),
        "init_scales": INIT_SCALES,
        "best": best,
        "rows": rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "best": best}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
