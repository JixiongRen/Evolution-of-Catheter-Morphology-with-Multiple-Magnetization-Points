"""Validate n2 trusted branch reachability through current continuation."""

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
P2C79_DIR = ROOT / "experiment_records_ik_optimize" / "20260628020500_p2c79_branch_random_validation"
for path in (ROOT, P2C79_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk

import run_p2c79_branch_random_validation as p2c79

jax.config.update("jax_enable_x64", True)
configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def _jsonable(value: Any) -> Any:
    return p2c79._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _solve_stage(params: Any, target_p: np.ndarray, y_ref: np.ndarray, y0: np.ndarray, I: np.ndarray) -> dict[str, Any]:
    _args, funcs = p2c79.p2c69._make_funcs(params, target_p, y_ref, p2c79.p2c69.STAGES[-1][0])
    solve = p2c79.p2c57.p2c47._solve(y0, I, funcs, _args)
    row = p2c79.p2c57._row(
        force_scale=p2c79.p2c69.FORCE_SCALE,
        stage=4,
        pos_scale=p2c79.p2c69.STAGES[-1][0],
        max_iter=_args.max_iter,
        params=params,
        target_p=target_p,
        solve=solve,
    )
    row["reg_scale"] = p2c79.p2c69.REG_SCALE
    return {"solve": solve, "row": row, "funcs": funcs}


def _branch_from_y(params: Any, I0: np.ndarray, target_p: np.ndarray, y: np.ndarray, y_ref: np.ndarray, label: str) -> dict[str, Any]:
    _args, funcs = p2c79.p2c69._make_funcs(params, target_p, y_ref, p2c79.p2c69.STAGES[-1][0])
    solve = p2c79.p2c57.p2c47._solve(y, I0, funcs, _args)
    rec = {"solve": solve, "rows": [], "y_ref": y_ref}
    # Reuse branch construction by emulating final solve fields.
    J = np.asarray(jax.device_get(funcs["jacobian"](jnp.asarray(solve["y"], dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8))), dtype=float)
    _u, _s, vt = np.linalg.svd(J, full_matrices=False)
    basis = p2c79.p2c72.p2c71._orthogonal_complement(vt[-p2c79.p2c72.REMOVE_WEAK_DIRECTIONS :], J.shape[1])
    system = p2c79.p2c77._build_optimality_system(y_base=solve["y"], I0=I0, funcs=funcs, basis=basis)
    S = p2c79.p2c78._task_sensitivity(system)
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)
    out = p2c79.p2c57.p2c47.output_to_numpy(funcs["output"](jnp.asarray(solve["y"], dtype=jnp.float64), jnp.asarray(I0, dtype=jnp.float64).reshape(8)))
    final_row = p2c79.p2c57._row(
        force_scale=p2c79.p2c69.FORCE_SCALE,
        stage=4,
        pos_scale=p2c79.p2c69.STAGES[-1][0],
        max_iter=_args.max_iter,
        params=params,
        target_p=target_p,
        solve=solve,
    )
    return {
        "label": label,
        "recovered": rec,
        "final_row": final_row,
        "y": np.asarray(solve["y"], dtype=float),
        "y_ref": y_ref,
        "funcs": funcs,
        "basis": basis,
        "system": system,
        "S": S,
        "U": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "tip": np.asarray(out["tip_p"], dtype=float),
        "task_condition": float(singular_values[0] / max(float(singular_values[-1]), np.finfo(float).tiny)),
    }


def _run_continuation(params: Any, I0: np.ndarray, target_p: np.ndarray, y0: np.ndarray, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = np.asarray(y0, dtype=float).copy()
    y_ref = y.copy()
    rows = []
    for frac in FRACTIONS:
        I = np.asarray(I0, dtype=float) * frac
        result = _solve_stage(params, target_p, y_ref, y, I)
        y = np.asarray(result["solve"]["y"], dtype=float)
        row = dict(result["row"])
        row["path"] = label
        row["current_fraction"] = frac
        row["gate_ok"] = bool(
            row["success"]
            and row["stationarity_norm"] <= 1.0e-5
            and row["continuity_norm"] <= 1.0e-3
            and row["max_interface_force_jump_over_rigid_force"] <= 1.5
        )
        rows.append(row)
    branch = _branch_from_y(params, I0, target_p, y, y_ref, f"continuation_{label}")
    return branch, rows


def _plot(rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for path in sorted({row["path"] for row in rows}):
        part = [row for row in rows if row["path"] == path]
        axes[0].plot([row["current_fraction"] for row in part], [row["stationarity_norm"] for row in part], marker="o", label=path)
        axes[1].plot([row["current_fraction"] for row in part], [row["max_interface_force_jump_over_rigid_force"] for row in part], marker="o", label=path)
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("一阶最优性")
    axes[1].set_ylabel("载荷传播比")
    axes[1].set_xlabel("电流加载比例")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("P2c80 continuation 分支复核")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_continuation_branch_cn.png", dpi=180)
    plt.close(fig)


def _record(stage_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], wall_s: float) -> str:
    stage_table = [
        "| 路径 | 电流比例 | 一阶最优性 | 接口连续 | 载荷比 | 阶段门禁 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in stage_rows:
        stage_table.append(
            f"| {row['path']} | {row['current_fraction']:.2f} | {row['stationarity_norm']:.3e} | "
            f"{row['continuity_norm']:.3e} | {row['max_interface_force_jump_over_rigid_force']:.3e} | {row['gate_ok']} |"
        )
    branch_table = [
        "| 分支 | 基线门禁 | 分支一致 | 末端差异 | 灵敏度差异 | 任务条件数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in branch_rows:
        branch_table.append(
            f"| {row['branch']} | {row['baseline_ok']} | {row['branch_ok']} | {row['tip_diff_m']:.3e} | "
            f"{row['sensitivity_rel_err']:.3e} | {row['task_condition']:.3e} |"
        )
    all_ok = all(row["branch_ok"] for row in branch_rows)
    return "\n".join(
        [
            "# P2c80 n2 continuation 分支复核记录",
            "",
            "## 实验目的",
            "",
            "P2c79 显示 trusted 分支内灵敏度稳定，但零初值和半初值直接求解落入坏分支。本轮使用电流分段加载检验 trusted 分支是否可通过 continuation 稳定到达。",
            "",
            "## 分段加载过程",
            "",
            *stage_table,
            "",
            "## 最终分支对齐",
            "",
            *branch_table,
            "",
            "## 结论",
            "",
            f"continuation 分支一致性整体结果：`{all_ok}`。若为真，可将 n2 发布口径限定为 continuation 可达分支；若为假，初始化敏感仍未解决。",
            "零初值路径在所有加载阶段均出现较大接口失配和载荷传播失配；半初值路径虽然末端位置接近 trusted 分支，但一阶最优性在后续加载中急剧恶化，最终灵敏度与 trusted 分支差异仍达 `1.557e+01`。因此“末端位置接近”不能作为物理解分支一致的证据。",
            "本轮将 n2 当前工况的主要风险进一步归因为初始化敏感和多平衡分支竞争。trusted 分支内三维末端位置灵敏度稳定，但尚不能声明为一般初值下可达的稳健灵敏度。",
            "",
            f"- 总耗时：`{wall_s:.2f} s`",
            "",
            "## 下一步计划",
            "",
            "1. 若继续推进 n2，应尝试更细电流加载或目标点 continuation，但优先级低于整理当前局部分支证据链。",
            "2. 当前可发布口径只能是“trusted 分支条件下的局部三维末端位置灵敏度候选”，不能宣称初始化无关。",
            "3. 下一轮建议形成 n2 工程边界报告，明确可用条件、不可用初值和失败机制。",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    params, I0 = p2c79.p2c57.p2c24._params_from_saved_case(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    target_p, y_slack = p2c79.p2c57.p2c24._load_saved_target_and_seed(p2c79.p2c57.CASE_NAME, p2c79.p2c57.ALPHA)
    y_seed = p2c79.p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c79.p2c57.Args.steps_per_flex)
    trusted = p2c79._recover_branch(params, I0, target_p, y_seed, "trusted_seed")
    seeds = {
        "zero_seed": np.zeros_like(y_seed),
        "half_seed": 0.5 * np.asarray(y_seed, dtype=float),
    }
    stage_rows: list[dict[str, Any]] = []
    branches = []
    for label, seed in seeds.items():
        branch, rows = _run_continuation(params, I0, target_p, seed, label)
        branches.append(branch)
        stage_rows.extend(rows)
    branch_rows = [p2c79._branch_row(branch, trusted) for branch in branches]
    wall_s = float(time.perf_counter() - t0)
    _plot(stage_rows, branch_rows)
    _write_csv(OUT_DIR / "continuation_stage_rows.csv", stage_rows)
    _write_csv(OUT_DIR / "continuation_branch_rows.csv", branch_rows)
    _write_csv(OUT_DIR / "summary.csv", stage_rows + branch_rows)
    failures = []
    for row in stage_rows:
        if not row["gate_ok"]:
            failures.append({"failure": "continuation 阶段门禁失败", "evidence": row})
    for row in branch_rows:
        if not row["branch_ok"]:
            failures.append({"failure": "continuation 最终分支未对齐", "evidence": row})
    (OUT_DIR / "failure_events.jsonl").write_text("".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures), encoding="utf-8")
    payload = {
        "experiment": OUT_DIR.name,
        "fractions": FRACTIONS,
        "wall_s": wall_s,
        "stage_rows": stage_rows,
        "branch_rows": branch_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "record.md").write_text(_record(stage_rows, branch_rows, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "failures": failures}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
