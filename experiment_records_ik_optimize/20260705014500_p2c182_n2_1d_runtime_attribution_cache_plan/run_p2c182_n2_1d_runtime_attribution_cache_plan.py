"""Create a runtime attribution and cache plan for the n2 one-dimensional gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(ROOT))
from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write a JSON file."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build runtime attribution artifacts from P2c179."""
    configure_matplotlib_cjk()
    p2c179_dir = ROOT / "experiment_records_ik_optimize/20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop"
    summary179 = load_json(p2c179_dir / "summary.json")
    rows179 = list(csv.DictReader((p2c179_dir / "branch_loop_rows.csv").open(encoding="utf-8-sig")))

    wall_s = float(summary179["wall_s"])
    steady_s = float(summary179["total_solve_steady_s"])
    overhead_s = wall_s - steady_s
    solve_calls = len(rows179) * 2
    mean_solve_s = steady_s / max(solve_calls, 1)
    overhead_ratio = overhead_s / max(wall_s, 1e-12)

    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "runtime_dominated_by_branch_recovery_and_compile_overhead",
        "decision": "n2一维门禁数值重求解本身很快；当前效率瓶颈主要来自重复分支恢复、函数构造、JAX编译和CPU后端。",
        "source_experiment": summary179["experiment_id"],
        "wall_s": wall_s,
        "steady_s": steady_s,
        "overhead_s": overhead_s,
        "overhead_ratio": overhead_ratio,
        "solve_call_count": solve_calls,
        "mean_solve_s": mean_solve_s,
        "jax_devices": summary179["jax_devices"],
        "gate_pass_count": summary179["pass_count"],
        "gate_tested_count": summary179["tested_count"],
        "cache_plan": [
            "Cache recovered workpoint branches after baseline gate.",
            "Reuse local functions and reduced bases across positive and negative perturbations.",
            "Batch four initial states per perturbation where shapes are identical.",
            "Separate first-call compilation timing from steady-state gate timing.",
            "Restore GPU runtime before claiming production efficiency.",
        ],
        "production_readiness": "not_ready_until_cached_gpu_gate_is_verified",
    }
    write_json(EXP_DIR / "summary.json", summary)

    failures = [
        {
            "case": "runtime_efficiency",
            "failure_class": "compile_and_recovery_overhead_dominates",
            "evidence": f"overhead_ratio={overhead_ratio:.6f}",
            "decision": "must_cache_branch_and_reuse_compiled_paths",
        },
        {
            "case": "gpu_efficiency",
            "failure_class": "cuda_backend_unavailable",
            "evidence": ",".join(summary179["jax_devices"]),
            "decision": "cannot_claim_gpu_efficiency",
        },
    ]
    (EXP_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].pie(
        [steady_s, overhead_s],
        labels=["净重求解", "恢复/构造/编译等开销"],
        autopct="%1.2f%%",
        colors=["#2a9d8f", "#e76f51"],
        startangle=90,
    )
    axes[0].set_title("P2c179 耗时构成")
    labels = [item["workpoint"] for item in summary179["workpoint_summaries"]]
    task_errors = [item["max_task_error"] for item in summary179["workpoint_summaries"]]
    axes[1].bar(labels, task_errors, color="#457b9d")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("各工作点最大任务线性化误差")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("n2一维门禁效率归因")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_1d_runtime_attribution_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c182 两点磁控一维门禁效率归因与缓存改造计划

## 实验目的

P2c179 已证明两点磁控第 1 主方向在 5 个工作点、40 个严格回环样本中全部通过。本轮不重跑求解，而是对 P2c179 的耗时结构做定量归因，为后续效率产品化提供改造优先级。

## 关键结果

| 指标 | 数值 |
|---|---:|
| 墙钟耗时 | {wall_s:.2f} s |
| 净重求解耗时 | {steady_s:.2f} s |
| 非重求解开销 | {overhead_s:.2f} s |
| 非重求解开销占比 | {overhead_ratio:.4%} |
| 重求解调用数 | {solve_calls} |
| 单次重求解平均耗时 | {mean_solve_s:.4f} s |
| 运行设备 | {', '.join(summary179['jax_devices'])} |

## 学术解释

当前瓶颈不在局部扰动重求解本身。严格回环中的正向、反向、多初值和回环求解累计只占约 {steady_s:.2f} s，而总墙钟约 {wall_s:.2f} s。这说明主要开销来自工作点分支恢复、目标分支推进、函数和约化基构造、JAX 首次编译，以及当前 CPU 后端执行。

因此，后续若要提高效率，不应降低一阶最优性、中心线漂移或局部线性化门限，而应减少重复构造和重复编译。

## 缓存改造计划

1. 对每个通过基线门禁的工作点，缓存物理解分支、约化基、局部灵敏度矩阵和基线输出。
2. 同一工作点的正向扰动、反向扰动、多初值和回环检查复用同一组函数和约化基。
3. 将四类初值作为同形状批量输入，减少 Python 循环和重复调度。
4. 明确区分首次编译耗时和稳定运行耗时，报告两类时间。
5. 先恢复 GPU 后端，再评估产品级耗时；CPU 结果只能作为保守参考。

## 可发布边界不变

效率改造不得扩大当前结论。当前仍只允许发布两点磁控第 1 主方向的一维分支条件局部灵敏度候选；三维末端位置、完整末端位姿和外层 IK 直接接入仍然禁止。

## 下一步计划

下一轮建议实现“只读缓存包”：把 P2c179 中每个工作点的分支、约化基、灵敏度矩阵和基线输出保存为可复用数据；随后运行同一门禁，比较冷启动、缓存启动和稳定运行时间。若可信性指标不变，再进入批量多初值求解改造。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c182_n2_1d_runtime_attribution_cache_plan.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
