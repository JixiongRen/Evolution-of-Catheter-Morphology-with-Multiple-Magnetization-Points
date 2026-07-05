from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCES = {
    "P2c120": ROOT / "experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/summary.json",
    "P2c126": ROOT / "experiment_records_ik_optimize/20260629140000_p2c126_n1_failure_mechanism_repair_map/summary.json",
    "P2c128": ROOT / "experiment_records_ik_optimize/20260629160000_p2c128_n1_readonly_rigid_load_observer/summary.json",
    "P2c129": ROOT / "experiment_records_ik_optimize/20260629170000_p2c129_n1_explicit_rigid_load_residual/summary.json",
    "P2c130": ROOT / "experiment_records_ik_optimize/20260629190000_p2c130_n1_task_reduction_conservative_workpoint/summary.json",
    "P2c131": ROOT / "experiment_records_ik_optimize/20260629210000_p2c131_n1_configuration_boundary/summary.json",
    "P2c132": ROOT / "experiment_records_ik_optimize/20260629230000_p2c132_n1_mid_branch_consistency/summary.json",
    "P2c138": ROOT / "experiment_records_ik_optimize/20260630060000_p2c138_n1_effective_subspace_boundary_package/summary.json",
    "P2c139": ROOT / "experiment_records_ik_optimize/20260630063000_p2c139_n1_mid_branch_recovery_probe/summary.json",
    "P2c140": ROOT / "experiment_records_ik_optimize/20260629003000_p2c140_n1_continuation_branch_current_reopt/summary.json",
    "P2c159": ROOT / "experiment_records_ik_optimize/20260630160000_p2c159_n1_engineering_boundary_synthesis/summary.json",
}


def jsonable(value: Any) -> Any:
    """Convert values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_sources() -> dict[str, dict[str, Any]]:
    """Load source summaries."""
    data = {}
    for name, path in SOURCES.items():
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    return data


def build_summary(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the n1 release-boundary summary."""
    return {
        "experiment_id": EXP_DIR.name,
        "status": "n1_current_model_not_publishable_strict_boundary",
        "decision": "单点磁控当前模型不发布末端位置灵敏度，也不接入外层 IK；可作为模型失配分析和载荷传播修复对照。",
        "source_experiments": {name: str(path) for name, path in SOURCES.items()},
        "hard_evidence": {
            "p2c120_publishable_count": data["P2c120"].get("publishable_count"),
            "p2c126_primary_failure": data["P2c126"].get("primary_failure"),
            "p2c128_continuity_norm": data["P2c128"].get("final_gate", {}).get("continuity_norm"),
            "p2c129_pass_count": data["P2c129"].get("pass_count"),
            "p2c130_pass_count": data["P2c130"].get("pass_count"),
            "p2c131_pass_count": data["P2c131"].get("pass_count"),
            "p2c132_branch_status": data["P2c132"].get("branch_status"),
            "p2c138_release_status": data["P2c138"].get("release_status"),
            "p2c139_branch_conditioned_only": data["P2c139"].get("branch_conditioned_only"),
            "p2c140_release_status": data["P2c140"].get("release_status"),
            "p2c159_status": data["P2c159"].get("status"),
        },
        "failure_classes": [
            "刚性磁体段载荷传播失配",
            "条件数病态",
            "初始化敏感与多平衡竞争",
            "任务降维仍未闭合",
            "分支条件下证据不足",
        ],
        "allowed_use": [
            "模型失配分析",
            "分支恢复方法验证",
            "刚性磁体段载荷传播模型修复对照",
            "工作空间不可发布边界说明",
        ],
        "forbidden_use": [
            "发布单点磁控末端位置灵敏度",
            "发布单点磁控完整位姿灵敏度",
            "接入外层 IK",
            "批量扰动扫描作为控制律依据",
        ],
        "next_required_work": [
            "显式修复刚性磁体段载荷传播模型",
            "修复后先做基线门禁，不直接扰动扫描",
            "若仍失败，转为导管配置或磁化点设计不可发布建议",
        ],
        "publishable_effective_subspace": False,
        "publishable_full_3d_position": False,
        "publishable_full_pose": False,
        "outer_ik_allowed": False,
        "wall_s": 0.0,
    }


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese record."""
    evidence = summary["hard_evidence"]
    text = f"""# P2c177 单点磁控 n1 发布边界综合

## 实验目的

当前长期目标要求：若单点磁控无法获得可发布有效子空间灵敏度，必须给出严格证据说明当前模型或配置不可发布。本轮不重跑长耗时求解，而是综合 P2c120、P2c126、P2c128-P2c132、P2c138-P2c140 和 P2c159，形成单点磁控 n1 的发布边界结论。

## 证据摘要

- P2c120：可发布数量为 `{evidence['p2c120_publishable_count']}`，主结论为当前模型不可发布。
- P2c126：主失败机制为 `{evidence['p2c126_primary_failure']}`。
- P2c128：只读载荷观测中连续残差为 `{evidence['p2c128_continuity_norm']:.3e}`，超过 `1e-3` 门限。
- P2c129：显式刚段载荷残差最小原型通过数为 `{evidence['p2c129_pass_count']}`。
- P2c130：任务降维和保守工作点通过数为 `{evidence['p2c130_pass_count']}`。
- P2c131：配置扫描仅得到 `{evidence['p2c131_pass_count']}` 个基线候选，但未形成可发布灵敏度。
- P2c132：中间分支一致性状态为 `{evidence['p2c132_branch_status']}`。
- P2c138：有效子空间发布状态为 `{evidence['p2c138_release_status']}`。
- P2c139：虽出现分支条件下恢复，但标记为 `{evidence['p2c139_branch_conditioned_only']}`，不能视为全局可信。
- P2c140：连续分支电流重求解状态为 `{evidence['p2c140_release_status']}`。
- P2c159：工程边界状态为 `{evidence['p2c159_status']}`。

## 失败机制

单点磁控当前不是“缺少某个合适扰动半径”的问题，而是模型闭合链路未通过。主失败机制包括：

1. 刚性磁体段载荷传播失配。
2. 条件数病态。
3. 初始化敏感和多平衡竞争。
4. 任务降维后仍不能稳定闭合。
5. 分支条件下证据不足，不能推广为可发布灵敏度。

## 实验结论

单点磁控 n1 当前模型不发布末端位置灵敏度，不发布完整末端位姿灵敏度，不接入外层 IK。它只能用于模型失配分析、分支恢复方法验证、刚性磁体段载荷传播修复对照和工作空间不可发布边界说明。

## 工程建议

1. 不再对当前 n1 模型做批量扰动扫描。
2. 下一步若继续 n1，必须先显式修复刚性磁体段载荷传播模型。
3. 修复后先做基线门禁，再考虑有效子空间灵敏度。
4. 若修复后仍失败，应把单点磁控当前配置列为不可发布配置，而不是继续调权重。

## 耗时

- 本轮为证据综合：0 s。
- 是否重跑求解：否。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the n1 release-boundary synthesis."""
    data = load_sources()
    summary = build_summary(data)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for item in summary["failure_classes"]:
            f.write(json.dumps({"failure_class": item, "evidence": summary["hard_evidence"]}, ensure_ascii=False) + "\n")
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705035000_p2c177_n1_release_boundary_synthesis/run_p2c177_n1_release_boundary_synthesis.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
