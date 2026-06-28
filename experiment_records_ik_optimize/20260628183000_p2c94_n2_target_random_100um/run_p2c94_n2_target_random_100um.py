"""Run 100 um random-direction target-continuation audits for n2."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
P2C89_DIR = ROOT / "experiment_records_ik_optimize" / "20260628150600_p2c89_n2_target_point_continuation"
for path in (ROOT, P2C89_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import run_p2c89_n2_target_point_continuation as p2c89


OUT_DIR = Path(__file__).resolve().parent
TARGET_STEP_M = 1.0e-4
MAX_ITER_PER_STEP = 160
RANDOM_SEED = 20260628
N_DIRECTIONS = 6


def _target_dirs() -> list[tuple[str, np.ndarray]]:
    """Return deterministic random target perturbation directions."""
    rng = np.random.default_rng(RANDOM_SEED)
    dirs: list[tuple[str, np.ndarray]] = []
    for idx in range(N_DIRECTIONS):
        vec = rng.normal(size=3)
        vec = vec / np.linalg.norm(vec)
        dirs.append((f"rand_{idx:02d}", vec))
    return dirs


def _plot(rows: list[dict], summaries: list[dict]) -> None:
    """Plot random-direction continuation diagnostics."""
    p2c89._set_chinese_font()
    sci = mticker.FuncFormatter(lambda value, _pos: f"{value:.0e}" if value > 0 else "0")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for direction in sorted({row["direction"] for row in rows}):
        part = [row for row in rows if row["direction"] == direction]
        x = [row["step_index"] for row in part]
        axes[0].plot(x, [row["stationarity_norm"] for row in part], marker="o", label=direction)
        axes[1].plot(x, [row["max_interface_force_jump_over_rigid_force"] for row in part], marker="o")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(sci)
    axes[0].set_title("一阶最优性")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("载荷传播比例")
    axes[2].bar([row["direction"] for row in summaries], [row["return_state_drift"] for row in summaries], color="#457b9d")
    axes[2].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("闭环返回状态漂移")
    axes[2].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c94 n2 目标点 100 微米随机方向闭环延拓")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_target_random_100um_cn.png", dpi=180)
    plt.close(fig)


def _record(base_row: dict, summaries: list[dict], directions: list[tuple[str, np.ndarray]], wall_s: float) -> str:
    """Render the Chinese experiment report."""
    dir_lines = [f"- `{name}`: `{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}`" for name, vec in directions]
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
            "# P2c94 n2 目标点 100 微米随机方向闭环延拓记录",
            "",
            "## 实验目的",
            "",
            "P2c92 和 P2c93 已证明 n2 受信任分支在坐标轴方向与组合方向 100 微米目标扰动下可闭环返回。本轮使用固定随机方向，检验该结论是否依赖人工选定方向。",
            "",
            "## 实验设置",
            "",
            f"- 目标扰动范数：`{TARGET_STEP_M:.3e} m`",
            f"- 路径比例：`{p2c89.FRACTIONS}`",
            f"- 每步最大迭代：`{MAX_ITER_PER_STEP}`",
            f"- 随机种子：`{RANDOM_SEED}`",
            "",
            "## 随机方向",
            "",
            *dir_lines,
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
            f"共有 `{passed}` / `{len(summaries)}` 个随机方向闭环路径通过。若全通过，说明 n2 受信任分支在 100 微米局部任务邻域内不明显依赖人工方向选择；若失败，失败方向应作为局部工作空间边界候选。",
            "",
            "该结果仍是分支条件证据，只能支持 position-only 局部任务灵敏度继续推进，不能支持初始化无关或完整末端位姿灵敏度发布。",
            "",
            "## 下一步计划",
            "",
            "1. 若全通过，进入更大半径或电流扰动重求解一致性审计。",
            "2. 若失败，沿失败方向二分半径，定位可发布边界。",
            "3. 对 n1 保持不可发布机制归因路线，不与 n2 结论混用。",
            "",
            f"- 本轮耗时：`{wall_s:.2f} s`",
            "",
        ]
    )


def main() -> None:
    p2c89.OUT_DIR = OUT_DIR
    p2c89.TARGET_STEP_M = TARGET_STEP_M
    p2c89.MAX_ITER_PER_STEP = MAX_ITER_PER_STEP
    directions = _target_dirs()
    t0 = time.perf_counter()
    base = p2c89._recover_trusted()
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for name, direction in directions:
        rows, summary = p2c89._run_loop(base, name, direction)
        all_rows.extend(rows)
        summaries.append(summary)
    wall_s = float(time.perf_counter() - t0)
    p2c89._write_csv(OUT_DIR / "continuation_rows.csv", all_rows)
    p2c89._write_csv(OUT_DIR / "summary.csv", summaries)
    failures = [{"failure": "100微米随机方向闭环延拓未通过", "evidence": row} for row in summaries if not row["return_ok"]]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(p2c89._jsonable(item), ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    payload = {
        "experiment": OUT_DIR.name,
        "status": "target_random_100um_diagnostic",
        "target_step_m": TARGET_STEP_M,
        "fractions": list(p2c89.FRACTIONS),
        "random_seed": RANDOM_SEED,
        "directions": {name: vec.tolist() for name, vec in directions},
        "wall_s": wall_s,
        "base_row": base["base_row"],
        "summaries": summaries,
        "rows": all_rows,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(p2c89._jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(all_rows, summaries)
    (OUT_DIR / "record.md").write_text(_record(base["base_row"], summaries, directions, wall_s), encoding="utf-8")
    print(json.dumps(p2c89._jsonable({"wall_s": wall_s, "summaries": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
