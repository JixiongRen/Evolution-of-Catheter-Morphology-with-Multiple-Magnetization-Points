"""Audit hidden overhead in steady n2 cache replay gates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(ROOT))
from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write a UTF-8 JSON file."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Create hidden-overhead audit artifacts from P2c187."""
    configure_matplotlib_cjk()
    source = load_json(
        ROOT / "experiment_records_ik_optimize/20260705060000_p2c187_n2_independent_cache_steady_replay/summary.json"
    )
    rows = []
    for item in source["workpoint_summaries"]:
        steady = item["steady"]
        hidden = steady["gate_s"] - steady["solve_steady_s"]
        rows.append(
            {
                "workpoint": item["workpoint"],
                "steady_system_s": steady["system_build_s"],
                "steady_gate_s": steady["gate_s"],
                "steady_solve_s": steady["solve_steady_s"],
                "hidden_gate_overhead_s": hidden,
                "hidden_gate_overhead_ratio": hidden / max(steady["gate_s"], 1.0e-12),
                "steady_pass": f"{steady['pass_count']}/{steady['tested_count']}",
                "max_task_error": steady["max_task_error"],
            }
        )
    total_gate = sum(row["steady_gate_s"] for row in rows)
    total_solve = sum(row["steady_solve_s"] for row in rows)
    total_hidden = total_gate - total_solve
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "steady_replay_hidden_overhead_dominates",
        "decision": "稳态门禁已经可靠，但门禁耗时主要不在优化器计入的净重求解时间内；下一步应优化 solve_reduced 外部的残差预评估、输出评估和 Python 循环。",
        "source_experiment": source["experiment_id"],
        "steady_pass": f"{source['steady_pass_count']}/{source['steady_tested_count']}",
        "steady_total_s": source["steady_total_s"],
        "steady_gate_s": total_gate,
        "steady_solve_s": total_solve,
        "hidden_gate_overhead_s": total_hidden,
        "hidden_gate_overhead_ratio": total_hidden / max(total_gate, 1.0e-12),
        "jax_devices": source["jax_devices"],
        "rows": rows,
        "recommended_next_steps": [
            "Move residual warm-up timing inside solve_reduced accounting to expose true per-solve cost.",
            "Pre-evaluate and cache perturbation currents and predicted states before the gate loop.",
            "Batch or vectorize output evaluations for the four initial states after each perturbation.",
            "Avoid repeated return-loop solves when multi-start states are numerically identical within gate tolerance, but only after proving equivalence.",
            "Restore GPU backend before final production timing.",
        ],
    }
    write_json(EXP_DIR / "summary.json", summary)

    failures = [
        {
            "case": "steady_gate_runtime",
            "failure_class": "hidden_gate_overhead_dominates",
            "evidence": f"hidden_gate_overhead_ratio={summary['hidden_gate_overhead_ratio']:.6f}",
            "decision": "optimize_gate_loop_before_claiming_final_efficiency",
        },
        {
            "case": "gpu_backend",
            "failure_class": "cuda_backend_unavailable",
            "evidence": ",".join(source["jax_devices"]),
            "decision": "gpu_efficiency_not_claimed",
        },
    ]
    (EXP_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    labels = [row["workpoint"] for row in rows]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(x, [row["steady_solve_s"] for row in rows], label="净重求解", color="#2a9d8f")
    axes[0].bar(
        x,
        [row["hidden_gate_overhead_s"] for row in rows],
        bottom=[row["steady_solve_s"] for row in rows],
        label="隐藏门禁开销",
        color="#e76f51",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("稳态门禁耗时构成")
    axes[0].set_xticks(list(x), labels, rotation=30, ha="right")
    axes[0].legend()
    axes[1].bar(x, [row["hidden_gate_overhead_ratio"] for row in rows], color="#f4a261")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("隐藏开销占比")
    axes[1].set_xticks(list(x), labels, rotation=30, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("n2稳态缓存回放隐藏开销归因")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_steady_replay_hidden_overhead_cn.png", dpi=200)
    plt.close(fig)

    table = [
        "| 工作点 | 稳态门禁/s | 净重求解/s | 隐藏开销/s | 隐藏占比 | 稳态通过 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['workpoint']} | {row['steady_gate_s']:.2f} | {row['steady_solve_s']:.4f} | "
            f"{row['hidden_gate_overhead_s']:.2f} | {row['hidden_gate_overhead_ratio']:.2%} | {row['steady_pass']} |"
        )
    record = f"""# P2c188 两点磁控稳态缓存回放隐藏开销归因

## 实验目的

P2c187 证明预热后稳态回放保持 `40/40` 通过，稳态总耗时约 `53.53 s`。但稳态重求解净耗时只有 `0.91 s`。本轮不重跑求解，而是基于 P2c187 的行级计时数据，归因稳态门禁中的隐藏耗时。

## 结果

{chr(10).join(table)}

## 数据分析

稳态门禁总耗时为 `{total_gate:.2f} s`，其中优化器记录的净重求解时间为 `{total_solve:.4f} s`，隐藏门禁开销为 `{total_hidden:.2f} s`，占 `{summary['hidden_gate_overhead_ratio']:.2%}`。这说明即使完成编译预热，当前效率瓶颈仍主要来自 `solve_reduced` 外部的残差预评估、输出评估、返回分支检查、Python 循环和 JAX 调度，而不是最小二乘迭代本身。

## 结论

{summary['decision']}

## 下一步计划

1. 将 `solve_reduced` 中残差预评估纳入计时，避免低估单次重求解真实成本。
2. 将正负扰动、多初值和回环检查中的输出评估批量化。
3. 先验证多初值解在数值上一致后，再考虑减少重复回环求解。
4. GPU 后端恢复前，当前耗时只能视为 CPU 保守参考。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c188_steady_replay_hidden_overhead_audit.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
