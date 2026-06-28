"""Run a local target-point continuation audit for the n2 trusted branch."""

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
P2C57_DIR = ROOT / "experiment_records_ik_optimize" / "20260607200500_p2c57_force_continuity_weight_repair"
P2C69_DIR = ROOT / "experiment_records_ik_optimize" / "20260627230500_p2c69_n2_condition_precondition_audit"
for path in (ROOT, P2C57_DIR, P2C69_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import run_p2c57_force_continuity_weight_repair as p2c57
import run_p2c69_n2_condition_precondition_audit as p2c69

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent
TARGET_STEP_M = 5.0e-6
FRACTIONS = (0.0, 0.5, 1.0, 0.5, 0.0)
MAX_ITER_PER_STEP = 80


def _set_chinese_font() -> None:
    """Use a local CJK font for plots."""
    for font_file in [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]:
        if font_file.exists():
            fm.fontManager.addfont(str(font_file))
            prop = fm.FontProperties(fname=str(font_file))
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _jsonable(value: Any) -> Any:
    """Convert values into JSON-safe objects."""
    return p2c57._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _recover_trusted() -> dict[str, Any]:
    """Recover the trusted n2 branch used by P2c69 and P2c79."""
    params, current_i = p2c57.p2c24._params_from_saved_case(p2c57.CASE_NAME, p2c57.ALPHA)
    target_p, y_slack = p2c57.p2c24._load_saved_target_and_seed(p2c57.CASE_NAME, p2c57.ALPHA)
    y_seed = p2c57.p2c47._base_slack_to_segment_y(params, y_slack, steps_per_flex=p2c57.Args.steps_per_flex)
    recovered = p2c69._recover_near_gate(params, current_i, target_p, y_seed)
    return {
        "params": params,
        "current_i": np.asarray(current_i, dtype=float),
        "target_p": np.asarray(target_p, dtype=float),
        "y_ref": recovered["y_ref"],
        "y": np.asarray(recovered["solve"]["y"], dtype=float),
        "base_row": max(recovered["rows"], key=lambda row: row["stage"]),
    }


def _target_dirs() -> list[tuple[str, np.ndarray]]:
    """Return deterministic target perturbation directions."""
    return [
        ("x_plus", np.array([1.0, 0.0, 0.0])),
        ("y_plus", np.array([0.0, 1.0, 0.0])),
        ("z_plus", np.array([0.0, 0.0, 1.0])),
    ]


def _solve_target_step(
    *,
    params: Any,
    current_i: np.ndarray,
    target_p: np.ndarray,
    y_ref: np.ndarray,
    y0: np.ndarray,
) -> dict[str, Any]:
    """Solve one continuation target step."""
    args, funcs = p2c69._make_funcs(params, target_p, y_ref, p2c69.STAGES[-1][0])
    args.max_iter = MAX_ITER_PER_STEP
    solve = p2c57.p2c47._solve(y0, current_i, funcs, args)
    row = p2c57._row(
        force_scale=p2c69.FORCE_SCALE,
        stage=4,
        pos_scale=p2c69.STAGES[-1][0],
        max_iter=MAX_ITER_PER_STEP,
        params=params,
        target_p=target_p,
        solve=solve,
    )
    row["reg_scale"] = p2c69.REG_SCALE
    return {"solve": solve, "row": row}


def _gate_ok(row: dict[str, Any]) -> bool:
    """Return whether a continuation step satisfies the physical gates."""
    return bool(
        row["success"]
        and row["stationarity_norm"] <= 1.0e-5
        and row["continuity_norm"] <= 1.0e-3
        and row["max_interface_force_jump_over_rigid_force"] <= 1.5
    )


def _run_loop(base: dict[str, Any], direction_name: str, direction: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one out-and-back local target continuation loop."""
    params = base["params"]
    current_i = base["current_i"]
    target0 = base["target_p"]
    y_ref = base["y_ref"]
    y = base["y"].copy()
    y0 = base["y"].copy()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for step_idx, frac in enumerate(FRACTIONS):
        target = target0 + float(frac) * TARGET_STEP_M * direction
        out = _solve_target_step(params=params, current_i=current_i, target_p=target, y_ref=y_ref, y0=y)
        y = np.asarray(out["solve"]["y"], dtype=float)
        row = dict(out["row"])
        row.update(
            {
                "direction": direction_name,
                "step_index": step_idx,
                "target_fraction": float(frac),
                "target_offset_m": float(np.linalg.norm(target - target0)),
                "gate_ok": _gate_ok(row),
                "state_drift_from_base": float(np.linalg.norm(y - y0)),
                "wall_elapsed_s": float(time.perf_counter() - t0),
            }
        )
        rows.append(row)
    final = rows[-1]
    summary = {
        "direction": direction_name,
        "all_steps_gate_ok": all(row["gate_ok"] for row in rows),
        "return_state_drift": final["state_drift_from_base"],
        "return_position_error_m": final["position_error_m"],
        "return_stationarity_norm": final["stationarity_norm"],
        "return_force_ratio": final["max_interface_force_jump_over_rigid_force"],
        "return_ok": bool(
            all(row["gate_ok"] for row in rows)
            and final["state_drift_from_base"] <= 1.0e-3
            and final["position_error_m"] <= 5.0e-6
        ),
    }
    return rows, summary


def _plot(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    """Plot target-continuation diagnostics."""
    _set_chinese_font()
    sci = mticker.FuncFormatter(lambda value, _pos: f"{value:.0e}" if value > 0 else "0")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for direction in sorted({row["direction"] for row in rows}):
        part = [row for row in rows if row["direction"] == direction]
        axes[0].plot([row["step_index"] for row in part], [row["stationarity_norm"] for row in part], marker="o", label=direction)
        axes[1].plot([row["step_index"] for row in part], [row["max_interface_force_jump_over_rigid_force"] for row in part], marker="o")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(sci)
    axes[0].set_title("一阶最优性")
    axes[0].legend()
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("载荷传播比例")
    axes[2].bar([row["direction"] for row in summaries], [row["return_state_drift"] for row in summaries], color="#457b9d")
    axes[2].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("闭环返回状态漂移")
    axes[2].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c89 n2 目标点闭环延拓")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_target_point_continuation_cn.png", dpi=180)
    plt.close(fig)


def _record(base_row: dict[str, Any], summaries: list[dict[str, Any]], wall_s: float) -> str:
    """Render the Chinese report."""
    table = [
        "| 方向 | 全步骤过门 | 返回状态漂移 | 返回末端误差/m | 返回一阶最优性 | 返回载荷比 | 闭环结论 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        table.append(
            f"| {row['direction']} | {row['all_steps_gate_ok']} | {row['return_state_drift']:.3e} | "
            f"{row['return_position_error_m']:.3e} | {row['return_stationarity_norm']:.3e} | "
            f"{row['return_force_ratio']:.3e} | {'通过' if row['return_ok'] else '未通过'} |"
        )
    passed = sum(row["return_ok"] for row in summaries)
    return "\n".join(
        [
            "# P2c89 n2 目标点闭环延拓记录",
            "",
            "## 实验目的",
            "",
            "P2c88 显示 n2 受信任分支内三维末端位置灵敏度较强，但普通初值和电流延拓不能稳定回到该分支。本轮做局部目标点闭环延拓：从受信任目标出发，沿 x、y、z 三个方向做小目标扰动，再返回原目标，检验分支是否在局部任务路径内可跟踪。",
            "",
            "## 实验设置",
            "",
            f"- 目标扰动幅值：`{TARGET_STEP_M:.3e} m`",
            f"- 路径比例：`{FRACTIONS}`",
            f"- 每步最大迭代：`{MAX_ITER_PER_STEP}`",
            "- 判定：所有路径点通过一阶最优性、接口连续和载荷传播门禁，且返回状态漂移不超过 `1e-3`。",
            "",
            "## 基线",
            "",
            f"- 末端位置误差：`{base_row['position_error_m']:.3e} m`",
            f"- 一阶最优性：`{base_row['stationarity_norm']:.3e}`",
            f"- 接口连续：`{base_row['continuity_norm']:.3e}`",
            f"- 载荷传播比例：`{base_row['max_interface_force_jump_over_rigid_force']:.3e}`",
            "",
            "## 结果",
            "",
            *table,
            "",
            "## 结论",
            "",
            f"共有 `{passed}` / `{len(summaries)}` 个目标点闭环路径通过。若全部通过，只能说明 trusted 分支在局部目标邻域内可跟踪；它仍不能替代 zero/half 初值或全局电流延拓失败的证据。若存在失败路径，则 n2 当前发布口径必须进一步收窄为静态 trusted 分支条件导数。",
            "",
            "本轮结果用于区分两类风险：若闭环通过，主要风险是全局分支可达性；若闭环失败，主要风险是局部分支不稳定。两者都不能直接支持初始化无关的外层 IK 灵敏度接入。",
            "",
            "## 下一步计划",
            "",
            "1. 若闭环通过，下一轮做更长目标路径或多目标点延拓，寻找 trusted 分支可达域。",
            "2. 若闭环失败，停止 n2 发布级推进，给出受信任分支条件导数的工程边界。",
            "3. 无论结果如何，暂不推进完整末端位姿灵敏度。",
            "",
            f"- 本轮耗时：`{wall_s:.2f} s`",
            "",
        ]
    )


def main() -> None:
    t0 = time.perf_counter()
    base = _recover_trusted()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, direction in _target_dirs():
        rows, summary = _run_loop(base, name, direction)
        all_rows.extend(rows)
        summaries.append(summary)
    wall_s = float(time.perf_counter() - t0)
    _write_csv(OUT_DIR / "continuation_rows.csv", all_rows)
    _write_csv(OUT_DIR / "summary.csv", summaries)
    failures = [
        {"failure": "目标点闭环延拓未通过", "evidence": row}
        for row in summaries
        if not row["return_ok"]
    ]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(_jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "status": "target_continuation_diagnostic",
        "target_step_m": TARGET_STEP_M,
        "fractions": list(FRACTIONS),
        "wall_s": wall_s,
        "base_row": base["base_row"],
        "summaries": summaries,
        "rows": all_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(all_rows, summaries)
    (OUT_DIR / "record.md").write_text(_record(base["base_row"], summaries, wall_s), encoding="utf-8")
    print(json.dumps(_jsonable({"wall_s": wall_s, "summaries": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
