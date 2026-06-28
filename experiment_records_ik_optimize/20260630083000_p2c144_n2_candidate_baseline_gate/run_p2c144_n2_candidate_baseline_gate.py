"""Prepare and optionally run n2 candidate baseline gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
P2C89_DIR = ROOT / "experiment_records_ik_optimize/20260628150600_p2c89_n2_target_point_continuation"
for path in (ROOT, P2C89_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk
import run_p2c89_n2_target_point_continuation as p2c89


configure_matplotlib_cjk()
P2C94_SUMMARY = ROOT / "experiment_records_ik_optimize/20260628183000_p2c94_n2_target_random_100um/summary.json"
P2C143_CANDIDATES = ROOT / "experiment_records_ik_optimize/20260630080000_p2c143_n2_workpoint_candidate_design/n2_workpoint_candidate_rows.csv"
TARGET_STEP_M = 1.0e-4
MAX_ITER_PER_STEP = 160
RUN_WARNING = "This command can take roughly 30-45 minutes on the current environment."


def jsonable(value: Any) -> Any:
    """Convert arrays and scalar values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def read_candidate_rows() -> list[dict[str, str]]:
    """Read candidate rows selected by P2c143."""
    with P2C143_CANDIDATES.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def candidate_directions() -> list[tuple[str, np.ndarray]]:
    """Return candidate direction vectors for target continuation."""
    random_dirs = json.loads(P2C94_SUMMARY.read_text(encoding="utf-8"))["directions"]
    fixed = {
        "y_plus": np.array([0.0, 1.0, 0.0]),
        "z_plus": np.array([0.0, 0.0, 1.0]),
        "yz_plus": np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0),
    }
    rows = read_candidate_rows()
    directions: list[tuple[str, np.ndarray]] = []
    for row in rows:
        name = row["direction"]
        if name in fixed:
            vec = fixed[name]
        else:
            vec = np.asarray(random_dirs[name], dtype=float)
            vec /= max(float(np.linalg.norm(vec)), 1.0e-18)
        directions.append((name, vec))
    return directions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read rows from CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_executed_summary(rows: list[dict[str, Any]]) -> None:
    """Plot executed baseline-gate results."""
    labels = [str(row["direction"]) for row in rows]
    stationarity = [float(row["return_stationarity_norm"]) for row in rows]
    drift = [float(row["return_state_drift"]) for row in rows]
    force = [float(row["return_force_ratio"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    axes[0].bar(labels, stationarity, color="#f4a261")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("一阶最优性")
    axes[1].bar(labels, drift, color="#457b9d")
    axes[1].axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("返回状态漂移")
    axes[2].bar(labels, force, color="#2a9d8f")
    axes[2].axhline(1.5, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("载荷传播比例")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("P2c144 n2 候选工作点真实基线门禁")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n2_candidate_baseline_gate_cn.png", dpi=200)
    plt.close(fig)


def run_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run out-and-back target continuation for selected candidates."""
    p2c89.OUT_DIR = OUT_DIR
    p2c89.TARGET_STEP_M = TARGET_STEP_M
    p2c89.MAX_ITER_PER_STEP = MAX_ITER_PER_STEP
    base = p2c89._recover_trusted()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for name, direction in candidate_directions():
        case_rows, summary = p2c89._run_loop(base, name, direction)
        rows.extend(case_rows)
        summaries.append(summary)
    return rows, summaries, float(time.perf_counter() - t0)


def build_plan_rows() -> list[dict[str, Any]]:
    """Build planned candidate rows without running solvers."""
    rows = []
    for rank, (name, vec) in enumerate(candidate_directions(), start=1):
        rows.append(
            {
                "rank": rank,
                "direction": name,
                "direction_vector": vec.tolist(),
                "target_step_m": TARGET_STEP_M,
                "fractions": list(p2c89.FRACTIONS),
                "max_iter_per_step": MAX_ITER_PER_STEP,
                "planned_gate": "一阶最优性<=1e-5; 载荷传播<=1.5; 返回状态漂移<=1e-3 m; 返回位置误差<=5e-6 m",
            }
        )
    return rows


def write_record(summary: dict[str, Any]) -> None:
    """Write the Chinese experiment record."""
    if summary["execution_status"] == "planned_not_run":
        result = "本轮只生成可复现实验包，未执行长耗时求解。"
        result_table = "- 尚无运行结果。"
    else:
        result = (
            f"本轮已运行 {summary['candidate_count']} 个候选，闭环通过 "
            f"{summary['return_pass_count']}/{summary['candidate_count']}。"
        )
        result_table = "\n".join(
            [
                "| 方向 | 一阶最优性 | 返回状态漂移/m | 返回位置误差/m | 载荷传播比例 | 结论 |",
                "|---|---:|---:|---:|---:|---|",
                *[
                    (
                        f"| {row['direction']} | {float(row['return_stationarity_norm']):.3e} | "
                        f"{float(row['return_state_drift']):.3e} | "
                        f"{float(row['return_position_error_m']):.3e} | "
                        f"{float(row['return_force_ratio']):.3f} | "
                        f"{'通过' if row['return_ok'] else '未通过'} |"
                    )
                    for row in summary.get("candidate_summaries", [])
                ],
            ]
        )
    text = f"""# P2c144 两点磁控候选工作点真实基线门禁包

## 实验目的

P2c143 从既有目标闭环诊断中筛出 5 个 n2 新工作点候选。本轮建立真实基线门禁执行包，用于下一步无人值守运行。该包只验证候选是否能形成可信物理解分支，不直接发布灵敏度。

## 执行状态

{result}

## 运行结果

{result_table}

## 候选与门禁

- 目标偏移：`{TARGET_STEP_M:.1e} m`
- 延拓路径：`{p2c89.FRACTIONS}`
- 每步最大迭代：`{MAX_ITER_PER_STEP}`
- 通过门禁：一阶最优性残差不超过 `1e-5`，载荷传播比例不超过 `1.5`，返回状态漂移不超过 `1e-3 m`，返回位置误差不超过 `5e-6 m`。

## 长跑命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python experiment_records_ik_optimize/20260630083000_p2c144_n2_candidate_baseline_gate/run_p2c144_n2_candidate_baseline_gate.py --run
```

预计耗时：{RUN_WARNING}

## 解释边界

该门禁即使通过，也只能说明候选工作点具备进一步做奇异值谱和第一主方向扰动重求解的资格。它不能直接证明三维末端位置灵敏度可发布。

## 下一步计划

1. 对通过候选计算末端位置灵敏度矩阵奇异值谱。
2. 只对条件数和有效秩可接受的候选做第一主方向扰动重求解。
3. 若候选条件数仍处于谨慎诊断区，应形成工作空间分区建议，而不是直接发布三维灵敏度。
"""
    (OUT_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Prepare or run the candidate baseline gate package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the expensive candidate gates.")
    parser.add_argument("--postprocess", action="store_true", help="Rebuild plots and record from existing results.")
    args = parser.parse_args()
    plan_rows = build_plan_rows()
    write_csv(OUT_DIR / "planned_candidate_rows.csv", plan_rows)

    failure_events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "experiment_id": OUT_DIR.name,
        "purpose": "prepare_or_run_n2_candidate_baseline_gates",
        "execution_status": "planned_not_run",
        "candidate_count": len(plan_rows),
        "target_step_m": TARGET_STEP_M,
        "max_iter_per_step": MAX_ITER_PER_STEP,
        "fractions": list(p2c89.FRACTIONS),
        "publishable_now": False,
        "outer_ik_allowed": False,
        "run_warning": RUN_WARNING,
        "jax_devices": [str(device) for device in jax.devices()],
    }

    if args.postprocess:
        summary = json.loads((OUT_DIR / "summary.json").read_text(encoding="utf-8"))
        if summary.get("execution_status") == "executed":
            candidate_summaries = read_csv(OUT_DIR / "candidate_summary.csv")
            summary["candidate_summaries"] = [
                {
                    **row,
                    "return_ok": row["return_ok"] == "True",
                    "all_steps_gate_ok": row["all_steps_gate_ok"] == "True",
                }
                for row in candidate_summaries
            ]
            plot_executed_summary(summary["candidate_summaries"])
        write_record(summary)
        print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
        return

    if args.run:
        rows, candidate_summaries, wall_s = run_candidates()
        write_csv(OUT_DIR / "continuation_rows.csv", rows)
        write_csv(OUT_DIR / "candidate_summary.csv", candidate_summaries)
        plot_executed_summary(candidate_summaries)
        failure_events = [
            {"failure_class": "候选真实基线门禁未通过", "evidence": row}
            for row in candidate_summaries
            if not row["return_ok"]
        ]
        summary.update(
            {
                "execution_status": "executed",
                "wall_s": wall_s,
                "return_pass_count": sum(row["return_ok"] for row in candidate_summaries),
                "candidate_summaries": candidate_summaries,
            }
        )

    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for event in failure_events:
            f.write(json.dumps(jsonable(event), ensure_ascii=False) + "\n")
    write_record(summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
