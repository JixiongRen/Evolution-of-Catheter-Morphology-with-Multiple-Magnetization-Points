"""Create a dry-run cache schema for n2 branch-conditioned sensitivity gates."""

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
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write JSON to disk."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Write cache schema artifacts without running expensive branch recovery."""
    configure_matplotlib_cjk()
    p2c179 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705003000_p2c179_n2_mode1_cross_workpoint_branch_loop/summary.json"
    )
    p2c182 = load_json(
        ROOT / "experiment_records_ik_optimize/20260705014500_p2c182_n2_1d_runtime_attribution_cache_plan/summary.json"
    )
    workpoints = [item["workpoint"] for item in p2c179["workpoint_summaries"]]
    cache_fields = {
        "array_cache_npz": [
            "workpoint_name",
            "target_direction",
            "current_i0",
            "state_y",
            "reduced_basis",
            "terminal_sensitivity_S",
            "left_task_modes_U",
            "current_modes_Vt",
            "singular_values",
            "base_tip_position",
            "base_centerline",
        ],
        "json_metadata": [
            "source_experiment_id",
            "task_definition",
            "terminal_point_convention",
            "mode_index",
            "radius_A",
            "gate_thresholds",
            "workpoint_hash",
        ],
        "runtime_rebuild_required": [
            "JAX residual function",
            "JAX Jacobian function",
            "JAX output function",
            "compiled device executable",
        ],
    }
    replay_steps = [
        "Load cached state, basis, sensitivity matrix, local modes, baseline output.",
        "Rebuild only JAX functions from stored physical parameters and target convention.",
        "Reuse cached basis and local mode-1 current step for positive and negative perturbations.",
        "Run the same four-start perturbation and return-loop gate.",
        "Compare every gate metric against P2c179; any drift changes status to diagnostic only.",
    ]
    summary = {
        "experiment_id": EXP_DIR.name,
        "status": "cache_schema_ready_dryrun_only",
        "decision": "缓存接口已定义，但尚未生成真实分支缓存；下一轮需重跑一次分支恢复生成 npz/pickle 数据包并做回放对比。",
        "source_gate": p2c179["experiment_id"],
        "source_pass": f"{p2c179['pass_count']}/{p2c179['tested_count']}",
        "source_wall_s": p2c179["wall_s"],
        "source_steady_s": p2c179["total_solve_steady_s"],
        "source_overhead_ratio": p2c182["overhead_ratio"],
        "workpoints": workpoints,
        "cache_fields": cache_fields,
        "replay_steps": replay_steps,
        "acceptance": {
            "same_gate_pass_count": p2c179["pass_count"],
            "max_allowed_metric_drift": "1e-12 for loaded arrays; same gate thresholds for replay outputs",
            "required_reports": ["cold_build_time", "cache_load_time", "function_rebuild_time", "steady_gate_time"],
        },
        "not_claimed": [
            "No speedup is claimed in this dry run.",
            "No production cache is available until replay metrics match P2c179.",
            "No GPU efficiency is claimed while the latest backend remains TFRT_CPU_0.",
        ],
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_json(EXP_DIR / "cache_schema.json", cache_fields)

    failures = [
        {
            "case": "production_cache",
            "failure_class": "cache_not_built_yet",
            "evidence": "This run only defines schema and replay gates.",
            "decision": "needs_next_experiment",
        },
        {
            "case": "compiled_jax_functions",
            "failure_class": "not_serializable",
            "evidence": "JAX functions must be rebuilt from physical parameters and target convention.",
            "decision": "cache_arrays_not_executables",
        },
    ]
    (EXP_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in failures) + "\n",
        encoding="utf-8",
    )

    labels = ["真实重求解", "可缓存数组", "需重建函数", "可回放门禁"]
    values = [1.0, 1.0, 0.5, 0.5]
    colors = ["#2a9d8f", "#2a9d8f", "#f4a261", "#f4a261"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["未定义", "需重建", "可缓存"])
    ax.set_title("n2一维灵敏度缓存包状态")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_branch_cache_schema_cn.png", dpi=200)
    plt.close(fig)

    record = f"""# P2c183 两点磁控分支缓存包接口规范与干运行清单

## 实验目的

P2c182 表明 P2c179 的主要耗时不是扰动重求解，而是分支恢复、函数构造和编译等开销。本轮不重跑昂贵分支恢复，而是定义下一轮缓存包的接口规范、可落盘字段、不可落盘字段和回放验收标准。

## 可缓存内容

可直接保存为 `npz` 或 `json` 的内容包括工作点名称、目标方向、基线电流、物理状态向量、约化基、末端位置灵敏度矩阵、任务主方向、电流主方向、奇异值、基线末端点和基线中心线。

这些内容足以避免重复恢复物理解分支和重复计算局部灵敏度矩阵。

## 不能直接缓存的内容

JAX 残差函数、雅可比函数、输出函数和已编译设备执行体不能作为稳定缓存对象保存。后续回放仍需根据物理参数和任务约定重建函数。因此缓存方案不是“保存整个求解器”，而是“保存可信分支数组，运行时重建函数”。

## 回放验收

下一轮真实缓存实验必须完成以下检查：

1. 对 P2c179 的 {len(workpoints)} 个工作点生成缓存包。
2. 从缓存包加载分支数组，重建函数。
3. 重跑正负扰动、多初值和回环门禁。
4. 通过数必须保持 {p2c179['pass_count']} / {p2c179['tested_count']}。
5. 单独报告冷启动构建时间、缓存加载时间、函数重建时间和稳定门禁时间。

## 当前结论

本轮只完成缓存接口设计，不宣称速度提升。可信性边界不变：两点磁控仅第 1 主方向可作为一维分支条件候选；完整三维末端位置、完整末端位姿和直接外层 IK 仍不可发布。

## 下一步计划

下一轮执行真实缓存构建：重跑一次分支恢复生成缓存包，然后立即做缓存回放。若回放指标与 P2c179 一致，再比较冷启动与缓存启动耗时；若缓存回放失败，先定位是函数重建约定漂移、数组保存精度损失，还是终端点定义不一致。
"""
    (EXP_DIR / "record.md").write_text(record, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points
MPLCONFIGDIR=/tmp/matplotlib /home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/{EXP_DIR.name}/run_p2c183_n2_branch_cache_schema_dryrun.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")


if __name__ == "__main__":
    main()
