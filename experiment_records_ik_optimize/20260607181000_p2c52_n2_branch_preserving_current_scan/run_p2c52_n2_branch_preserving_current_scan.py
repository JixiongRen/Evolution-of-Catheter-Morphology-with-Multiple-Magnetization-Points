"""Run a branch-preserving current scan for the n2 position shooting gate."""

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
P2C24_DIR = ROOT / "experiment_records_ik_optimize" / "20260606160747_p2c24_interface_weight_continuation_repair"
P2C47_DIR = ROOT / "experiment_records_ik_optimize" / "20260607172000_p2c47_explicit_continuity_position_gate"
P2C51_DIR = ROOT / "experiment_records_ik_optimize" / "20260607173930_p2c51_n2_current_boundary_scan"
for path in (ROOT, P2C24_DIR, P2C47_DIR, P2C51_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c24_interface_weight_continuation_repair as p2c24
import run_p2c47_explicit_continuity_position_gate as p2c47
import run_p2c51_n2_current_boundary_scan as p2c51

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
CASE_NAME = "n2_z_pair_rand12"
ALPHAS = (0.02, 0.05, 0.08, 0.10)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for alpha in sorted({row["alpha"] for row in rows}):
        sub = [row for row in rows if row["alpha"] == alpha]
        final = max(sub, key=lambda row: row["stage"])
        best = min(sub, key=lambda row: (not row["baseline_gate"], row["stationarity_norm"], row["max_interface_force_jump_over_rigid_force"]))
        summaries.append(
            {
                "case": CASE_NAME,
                "alpha": alpha,
                "seed_mode": "p2c50_saved_branch",
                "any_gate_pass": any(row["baseline_gate"] for row in sub),
                "final_stage": final["stage"],
                "final_position_error_m": final["position_error_m"],
                "final_stationarity_norm": final["stationarity_norm"],
                "final_condition": final["jacobian_condition"],
                "final_continuity_norm": final["continuity_norm"],
                "final_force_jump_ratio": final["max_interface_force_jump_over_rigid_force"],
                "best_stage": best["stage"],
                "best_stationarity_norm": best["stationarity_norm"],
                "best_force_jump_ratio": best["max_interface_force_jump_over_rigid_force"],
                "dominant_failure": final["failure_class"],
            }
        )
    return summaries


def _plot(summaries: list[dict[str, Any]]) -> None:
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
    axes[0].set_ylabel("一阶最优性残差")
    axes[1].set_ylabel("条件数")
    axes[2].set_ylabel("接口连续残差")
    axes[3].set_ylabel("接口力跳变比")
    axes[3].set_xlabel("电流幅值系数 alpha")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("两点磁控同源分支电流边界扫描")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_branch_preserving_current_scan_cn.png", dpi=180)
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
            "# P2c52 两点同源分支电流边界扫描记录",
            "",
            "## 实验目的",
            "",
            "P2c51 显示，直接使用 FK oracle 分段内力/内矩初值会在 alpha=0.10 落入不同分支，不能复现 P2c50 的优质一阶最优性。本轮改用 P2c50 同源保存分支初值，对不同电流幅值重新生成目标并进行同一路径精修，判断两点工况是否存在可发布的电流幅值窗口。",
            "",
            "## 实验方案",
            "",
            "固定 `n2_z_pair_rand12`、position-only 任务、18 维分段内力/内矩状态、力矩全闭合边界和显式接口连续残差。初值来自 P2c24/P2c50 同源接口松弛分支，并在每个 alpha 下转换为分段内力/内矩初值。alpha 扫描为 `0.02,0.05,0.08,0.10`。",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            "若同源分支下仍无 alpha 通过门禁，则两点当前配置不能靠降低电流幅值获得发布级三维末端位置灵敏度；应转入刚性磁体段载荷传播模型或任务可辨识性修复。若 alpha=0.10 能复现 P2c50 而低 alpha 不能改善载荷传播，则说明失败与分支和模型闭合共同相关。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若存在过门 alpha，进入扰动重求解灵敏度审计。",
            "2. 若全部失败，停止当前 alpha 扫描，转入载荷传播模型修复。",
            "3. 对比 P2c51 与 P2c52，分离初值分支效应与电流幅值效应。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")

    _target_saved, y_slack_saved = p2c24._load_saved_target_and_seed(CASE_NAME, 0.10)
    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        params, current_i = p2c24._params_from_saved_case(CASE_NAME, alpha)
        target_p, _target_q, _y_oracle, _oracle = p2c51.p2c10t._build_case(CASE_NAME, alpha)[2:]
        y = p2c47._base_slack_to_segment_y(params, y_slack_saved, steps_per_flex=p2c51.Args.steps_per_flex)
        y_ref = y.copy()
        for idx, (pos_scale, cont_scale, max_iter) in enumerate(p2c51.STAGES, start=1):
            solve = p2c51._solve_stage(
                params,
                current_i,
                target_p,
                y_ref,
                y,
                pos_scale=pos_scale,
                cont_scale=cont_scale,
                max_iter=max_iter,
            )
            row = p2c51._row_from_solve(
                alpha=alpha,
                stage=idx,
                pos_scale=pos_scale,
                cont_scale=cont_scale,
                max_iter=max_iter,
                params=params,
                target_p=target_p,
                solve=solve,
            )
            row["seed_mode"] = "p2c50_saved_branch"
            rows.append(row)
            y = solve["y"]
            if not row["baseline_gate"]:
                p2c51._append_jsonl(failures, {"ts": time.time(), "failure": row["failure_class"], "evidence": row})

    summaries = _summary_rows(rows)
    wall_s = float(time.perf_counter() - t0)
    _plot(summaries)
    p2c51._write_csv(OUT_DIR / "branch_scan_rows.csv", rows)
    p2c51._write_csv(OUT_DIR / "summary.csv", summaries)
    payload = {
        "experiment": OUT_DIR.name,
        "wall_s": wall_s,
        "case": CASE_NAME,
        "alphas": list(ALPHAS),
        "summaries": summaries,
        "rows": rows,
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c51._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(summaries, wall_s), encoding="utf-8")
    print(json.dumps(p2c51._jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
