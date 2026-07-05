from __future__ import annotations

import csv
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

import matplotlib.pyplot as plt
import numpy as np

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


P2C145 = ROOT / "experiment_records_ik_optimize/20260630090000_p2c145_n2_candidate_spectrum_audit"
P2C148 = ROOT / "experiment_records_ik_optimize/20260630103000_p2c148_n2_all_principal_safe_radius"
P2C174 = ROOT / "experiment_records_ik_optimize/20260705020500_p2c174_n2_1d_branch_loop_gate"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file with UTF-8 BOM tolerance."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build cross-workpoint release-boundary rows from existing evidence."""
    spectrum = {row["direction"]: row for row in read_csv(P2C145 / "spectrum_rows.csv")}
    principal = read_csv(P2C148 / "all_principal_rows.csv")
    strict = json.loads((P2C174 / "summary.json").read_text(encoding="utf-8"))
    strict_direction = "rand_05"
    rows = []
    for row in principal:
        direction = row["direction"]
        mode = int(row["mode_index"])
        linear_ok = row["release_class"] == "通过"
        strict_loop_ok = direction == strict_direction and mode == 1 and strict["status"] == "n2_1d_branch_loop_candidate"
        if mode == 3:
            release_level = "不可发布"
            reason = "第三主方向跨工作点线性化失败"
        elif strict_loop_ok:
            release_level = "一维候选"
            reason = "已通过正负扰动、多初值和回环门禁"
        elif linear_ok:
            release_level = "待复核候选"
            reason = "跨工作点线性化通过，但缺少严格分支回环"
        else:
            release_level = "不可发布"
            reason = str(row["reason"])
        rows.append(
            {
                "direction": direction,
                "mode_index": mode,
                "baseline_task_condition": float(spectrum[direction]["task_condition"]),
                "baseline_force_ratio": float(spectrum[direction]["force_ratio"]),
                "radius_A": float(row["radius_A"]),
                "linear_rel_err": float(row["linear_rel_err"]),
                "centerline_max_change_m": float(row["centerline_max_change_m"]),
                "continuity_norm": float(row["continuity_norm"]),
                "force_ratio": float(row["force_ratio"]),
                "state_radius": float(row["state_radius"]),
                "linear_gate_pass": bool(linear_ok),
                "strict_branch_loop_pass": bool(strict_loop_ok),
                "release_level": release_level,
                "reason": reason,
            }
        )
    return rows, strict


def summarize(rows: list[dict[str, Any]], strict: dict[str, Any]) -> dict[str, Any]:
    """Summarize cross-workpoint release boundary."""
    directions = sorted({row["direction"] for row in rows})
    mode_summary = {}
    for mode in (1, 2, 3):
        mode_rows = [row for row in rows if row["mode_index"] == mode]
        mode_summary[str(mode)] = {
            "tested_workpoints": len(mode_rows),
            "linear_gate_pass_count": sum(row["linear_gate_pass"] for row in mode_rows),
            "strict_branch_loop_pass_count": sum(row["strict_branch_loop_pass"] for row in mode_rows),
            "max_linear_error": max(float(row["linear_rel_err"]) for row in mode_rows),
            "release_level": (
                "一维候选" if any(row["strict_branch_loop_pass"] for row in mode_rows)
                else "待复核候选" if all(row["linear_gate_pass"] for row in mode_rows)
                else "不可发布"
            ),
        }
    return {
        "experiment_id": EXP_DIR.name,
        "status": "n2_cross_workpoint_boundary_synthesized",
        "decision": "n2 存在一维有效子空间候选；二维方向有跨工作点线性化证据但缺少严格回环；三维完整位置灵敏度不可发布。",
        "workpoint_count": len(directions),
        "row_count": len(rows),
        "mode_summary": mode_summary,
        "strict_source": P2C174.name,
        "strict_status": strict["status"],
        "publishable_full_3d": False,
        "publishable_1d_candidate": True,
        "outer_ik_allowed": False,
        "engineering_recommendation": [
            "短期只推进第 1 主方向一维有效子空间。",
            "第 2 主方向需补正负扰动、多初值和回环门禁。",
            "第 3 主方向跨工作点均线性化失败，不应接入外层 IK。",
            "完整三维末端位置灵敏度当前不可发布。"
        ],
        "wall_s": 0.0,
        "rows": rows,
    }


def plot(rows: list[dict[str, Any]]) -> None:
    """Plot cross-workpoint linearization errors."""
    configure_matplotlib_cjk()
    directions = sorted({row["direction"] for row in rows})
    x = np.arange(len(directions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for offset, mode in zip((-width, 0.0, width), (1, 2, 3)):
        vals = []
        for direction in directions:
            match = next(row for row in rows if row["direction"] == direction and row["mode_index"] == mode)
            vals.append(match["linear_rel_err"])
        ax.bar(x + offset, vals, width=width, label=f"主方向 {mode}")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(x, directions, rotation=25, ha="right")
    ax.set_ylabel("局部线性化误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.suptitle("n2 跨工作点主方向线性化边界")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "n2_cross_workpoint_boundary_cn.png", dpi=200)
    plt.close(fig)


def write_record(summary: dict[str, Any]) -> None:
    """Write Chinese experiment record."""
    table = [
        "| 工作点 | 主方向 | 线性误差 | 分支回环 | 发布层级 | 原因 |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in summary["rows"]:
        table.append(
            f"| {row['direction']} | {row['mode_index']} | {row['linear_rel_err']:.3e} | "
            f"{row['strict_branch_loop_pass']} | {row['release_level']} | {row['reason']} |"
        )
    text = f"""# P2c175 两点磁控 n2 跨工作点灵敏度边界综合

## 实验目的

P2c174 证明 `rand_05` 工作点第 1 主方向存在一维有效子空间候选。本轮不重跑长耗时求解，而是综合 P2c145、P2c148 和 P2c174，判断该结论能否扩展到多个工作点，并明确 n2 当前可发布边界。

## 证据来源

- P2c145：5 个 n2 工作点均通过基线谱门禁，完整三维谱看似可行。
- P2c148：5 个工作点的第 1、第 2 主方向在 `1e-3 A` 半径下通过局部线性化；第 3 主方向全部失败。
- P2c174：仅对 `rand_05` 第 1 主方向完成正负扰动、多初值和回环门禁。

## 结果

{chr(10).join(table)}

## 数据分析

第 1 主方向具备跨工作点线性化证据，并且已有一个工作点通过严格分支回环。第 2 主方向也具备跨工作点线性化证据，但尚未完成严格分支回环，因此只能列为待复核候选。第 3 主方向在所有工作点均出现线性化失败，说明当前 n2 配置不支持完整三维末端位置灵敏度发布。

## 实验结论

{summary['decision']}

## 工程建议

1. 短期只推进第 1 主方向一维有效子空间。
2. 第 2 主方向必须补做正负扰动、多初值和回环门禁。
3. 第 3 主方向不应接入外层 IK。
4. 当前 n2 不发布完整三维末端位置灵敏度。

## 下一步计划

下一轮应对第 2 主方向执行与 P2c174 相同的严格回环门禁；若通过，可形成二维有效子空间候选。若失败，则 n2 当前工程边界应收敛为一维有效子空间。

## 耗时

- 本轮为证据综合：0 s。
- 是否重跑求解：否。
"""
    (EXP_DIR / "record.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the cross-workpoint boundary synthesis."""
    rows, strict = build_rows()
    summary = summarize(rows, strict)
    write_csv(EXP_DIR / "cross_workpoint_boundary_rows.csv", rows)
    (EXP_DIR / "summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (EXP_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            if row["release_level"] == "不可发布":
                f.write(json.dumps({"failure_class": row["reason"], "evidence": jsonable(row)}, ensure_ascii=False) + "\n")
    plot(rows)
    write_record(summary)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python \\
  experiment_records_ik_optimize/20260705024000_p2c175_n2_cross_workpoint_boundary_synthesis/run_p2c175_n2_cross_workpoint_boundary_synthesis.py
"""
    (EXP_DIR / "commands.sh").write_text(commands, encoding="utf-8")
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
