"""Create an updated publish-boundary synthesis after P2c67."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

P2C59 = ROOT / "experiment_records_ik_optimize" / "20260607204500_p2c59_force_reg_local_grid"
P2C62 = ROOT / "experiment_records_ik_optimize" / "20260627213500_p2c62_slack_nullspace_attribution"
P2C66 = ROOT / "experiment_records_ik_optimize" / "20260627224102_p2c66_n1_load_gate_consistency_audit"
P2C67 = ROOT / "experiment_records_ik_optimize" / "20260627224507_p2c67_n1_hard_load_transfer_parameterization"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _classify_n2(p2c59: dict[str, Any], p2c62: str) -> dict[str, Any]:
    best = min(p2c59["summaries"], key=lambda row: row["gate_violation_score"])
    return {
        "object": "两点 n2",
        "task": "三维末端位置",
        "status": "近门但不可发布",
        "best_case": f"force={best['force_continuity_scale']}, reg={best['reg_scale']}",
        "position_error_m": best["final_position_error_m"],
        "stationarity_norm": best["final_stationarity_norm"],
        "condition": best["final_condition"],
        "continuity_norm": best["final_continuity_norm"],
        "force_ratio": best["final_force_ratio_to_rigid_load"],
        "hard_failure": "条件数偏高",
        "evidence": "P2c59 显示载荷传播和一阶最优性可过门但条件数约 4.9e7；P2c62 证明低维松弛路线存在不可辨识零空间。",
        "next_step": "围绕 P2c59 近门分支做物理坐标预条件或残差块正交化，不再引入自由松弛变量。",
    }


def _classify_n1(p2c66: dict[str, Any], p2c67: dict[str, Any]) -> dict[str, Any]:
    best66 = p2c66["best"]
    best67 = p2c67["best"]
    return {
        "object": "单点 n1",
        "task": "有效子空间末端位置",
        "status": "当前模型下不可发布",
        "best_soft_case": f"force={best66['force_continuity_scale']}, reg={best66['reg_scale']}",
        "soft_condition": best66["jacobian_condition"],
        "soft_continuity_norm": best66["continuity_norm"],
        "soft_force_ratio": best66["max_force_ratio_to_rigid_load"],
        "hard_position_error_m": best67["position_error_m"],
        "hard_stationarity_norm": best67["stationarity_norm"],
        "hard_condition": best67["jacobian_condition"],
        "hard_failure": best67["failure_class"],
        "evidence": "P2c66 显示软约束近门样本仍无发布门禁通过；P2c67 显示硬载荷传播闭合后任务和一阶最优性退化。",
        "next_step": "停止单点扰动灵敏度审计，转入模型耦合缺失或任务有效秩归因。",
    }


def _record(rows: list[dict[str, Any]]) -> str:
    n1 = next(row for row in rows if row["object"] == "单点 n1")
    n2 = next(row for row in rows if row["object"] == "两点 n2")
    return "\n".join(
        [
            "# P2c68 单点/两点发布边界更新与 n2 下一步计划",
            "",
            "## 实验目的",
            "",
            "吸收 P2c64-P2c67 的新证据，更新单点与两点磁控导管 position-only 灵敏度发布边界。该轮不做新求解，只做证据综合和下一步实验入口选择。",
            "",
            "## 单点 n1 结论",
            "",
            f"单点当前判定为：`{n1['status']}`。P2c66 中最优软约束样本条件数 `{n1['soft_condition']:.3e}`，接口连续误差 `{n1['soft_continuity_norm']:.3e}`，相对刚段外力跳变 `{n1['soft_force_ratio']:.3e}`，未达到发布门禁。P2c67 中硬载荷传播样本虽然把接口跳变消除，但末端误差 `{n1['hard_position_error_m']:.3e} m`，一阶最优性 `{n1['hard_stationarity_norm']:.3e}`，条件数 `{n1['hard_condition']:.3e}`，说明硬闭合后任务和边界难以相容。",
            "",
            "因此，单点方向目前不能进入扰动灵敏度审计。后续只应作为模型耦合缺失、有效秩不足和不可发布边界证据链继续整理。",
            "",
            "## 两点 n2 结论",
            "",
            f"两点当前判定为：`{n2['status']}`。P2c59 的近门样本 `{n2['best_case']}` 末端误差 `{n2['position_error_m']:.3e} m`，一阶最优性 `{n2['stationarity_norm']:.3e}`，接口连续误差 `{n2['continuity_norm']:.3e}`，相对刚段外力跳变 `{n2['force_ratio']:.3e}`，这些指标已经可接受；主要阻断项是条件数 `{n2['condition']:.3e}`，高于 `1e7` 门禁。",
            "",
            "P2c62 说明低维载荷定标松弛虽然能拟合末端，但存在不可辨识零空间，不能作为发布路线。因此下一步 n2 不应继续引入自由松弛变量，而应围绕 P2c59 近门分支做物理坐标预条件、残差块正交化或有效子空间投影，目标是降低条件数，同时保持载荷传播和一阶最优性不退化。",
            "",
            "## 下一步最小实验",
            "",
            "建议 P2c69 只围绕 `n2_z_pair_rand12, alpha=0.10, force=4, reg=3e-4` 做条件数修复。候选策略：",
            "",
            "1. 对 18 维分段内力变量做基于残差雅可比列范数的物理坐标预条件，比较物理坐标条件数和预条件坐标条件数。",
            "2. 对残差块做正交化或白化，仅用于数值求解；发布门禁仍用原物理残差重新评价。",
            "3. 若条件数仍不能低于 `1e7`，提取最弱奇异方向，判断它是否主要是任务不可辨识方向；若是，则转为 n2 有效子空间灵敏度，而不是完整三维发布。",
            "",
            "## 工程建议",
            "",
            "1. 单点 n1：当前不应接入外层 IK，不发布灵敏度，只保留诊断证据。",
            "2. 两点 n2：以 P2c59 近门分支作为主战场；禁止用自由松弛变量换取末端拟合。",
            "3. 后续所有可发布声明必须同时报告物理坐标条件数、预条件坐标条件数和最弱方向物理含义。",
            "",
        ]
    )


def main() -> None:
    p2c59 = _load_json(P2C59 / "summary.json")
    p2c66 = _load_json(P2C66 / "summary.json")
    p2c67 = _load_json(P2C67 / "summary.json")
    p2c62_record = (P2C62 / "record.md").read_text(encoding="utf-8")
    rows = [_classify_n1(p2c66, p2c67), _classify_n2(p2c59, p2c62_record)]
    summary = {
        "experiment": OUT_DIR.name,
        "sources": [str(P2C59), str(P2C62), str(P2C66), str(P2C67)],
        "rows": rows,
        "recommended_next": "P2c69 n2 condition repair around P2c59 force=4 reg=3e-4",
    }
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps({"object": "单点 n1", "failure": "当前模型下不可发布", "evidence": rows[0]}, ensure_ascii=False) + "\n"
        + json.dumps({"object": "两点 n2", "failure": "条件数偏高", "evidence": rows[1]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
