"""Synthesize publishability boundaries for n1 and n2 sensitivity studies."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

P2C56 = ROOT / "experiment_records_ik_optimize" / "20260607194500_p2c56_n1_combined_gate_subspace_audit" / "summary.json"
P2C59 = ROOT / "experiment_records_ik_optimize" / "20260607204500_p2c59_force_reg_local_grid" / "summary.json"
P2C60 = ROOT / "experiment_records_ik_optimize" / "20260627165000_p2c60_scaled_variable_condition_probe" / "summary.json"
P2C61 = ROOT / "experiment_records_ik_optimize" / "20260627210200_p2c61_lowdim_load_transfer_parameterization" / "summary.json"
P2C62 = ROOT / "experiment_records_ik_optimize" / "20260627213500_p2c62_slack_nullspace_attribution" / "summary.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return min(rows, key=lambda row: row[key])


def _record(summary: dict[str, Any]) -> str:
    rows = summary["boundary_rows"]
    table = [
        "| 对象 | 任务层级 | 当前判定 | 关键通过项 | 硬失败项 | 主要证据 | 工程建议 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['object']} | {row['task_level']} | {row['publish_status']} | {row['passed_items']} | "
            f"{row['failed_items']} | {row['main_evidence']} | {row['engineering_recommendation']} |"
        )
    return "\n".join(
        [
            "# P2c63 单点与两点灵敏度发布边界综合报告",
            "",
            "## 实验目的",
            "",
            "本报告汇总 P2c56、P2c59、P2c60、P2c61 和 P2c62 的证据，给出当前单点与两点磁控导管在 position-only 灵敏度研究中的阶段性发布边界。目标不是扩大样本数量，而是明确哪些结果可发布、哪些只能诊断、哪些路线应停止。",
            "",
            "## 判定原则",
            "",
            "灵敏度发布必须先满足物理闭合和局部可线性化门禁。末端位置误差小只说明零阶几何目标被命中，不能替代一阶最优性、条件数、载荷传播、分支一致性和扰动重求解一致性。",
            "",
            "## 边界结论",
            "",
            *table,
            "",
            "## 两点磁控 n2 证据链",
            "",
            f"P2c59 的最佳近门组合为力连续权重 `4`、正则 `3e-4`，末端误差 `{summary['n2']['p2c59_best_position_error_m']:.3e} m`，一阶最优性 `{summary['n2']['p2c59_best_stationarity']:.3e}`，接口连续 `{summary['n2']['p2c59_best_continuity']:.3e}`，载荷传播比例 `{summary['n2']['p2c59_best_force_ratio']:.3e}`，但条件数 `{summary['n2']['p2c59_best_condition']:.3e}` 超过 `1e7` 门禁。",
            "",
            f"P2c60 表明简单变量尺度重整不能解决该问题。最佳缩放坐标条件数约 `{summary['n2']['p2c60_best_scaled_condition']:.3e}`，仍未低于 `1e7`；物理坐标条件数仍约 `{summary['n2']['p2c60_best_physical_condition']:.3e}`。",
            "",
            f"P2c61 表明低维硬连续模型 `base_6d` 消除载荷跳变但任务不可达，末端误差 `{summary['n2']['p2c61_base_position_error_m']:.3e} m`；低维载荷定标模型 `load_scaled` 可将末端误差压至 `{summary['n2']['p2c61_load_position_error_m']:.3e} m`，但秩亏、条件数无穷大，载荷传播比例 `{summary['n2']['p2c61_load_force_ratio']:.3e}`。",
            "",
            f"P2c62 进一步证明 `load_scaled` 的末端拟合依赖不可辨识松弛方向：无正则样本最弱方向中松弛变量占比 `{summary['n2']['p2c62_null_slack_fraction']:.3f}`，沿该方向施加 `1e-3` 扰动时末端变化仅 `{summary['n2']['p2c62_null_tip_change_m']:.3e} m`。因此该路线不能用于发布局部灵敏度。",
            "",
            "## 单点磁控 n1 证据链",
            "",
            f"P2c56 显示单点工况存在二维诊断有效子空间，但基线未过门：一阶最优性 `{summary['n1']['stationarity']:.3e}`，条件数 `{summary['n1']['condition']:.3e}`，接口连续 `{summary['n1']['continuity']:.3e}`，相对刚段外力跳变 `{summary['n1']['force_ratio']:.3e}`。因此该二维结构只能作为可控方向诊断线索，不能作为可发布灵敏度。",
            "",
            "## 工程建议",
            "",
            "1. 两点 `n2_z_pair_rand12, alpha=0.10`：暂不接入外层 IK，不发布三维末端位置灵敏度；后续只应推进显式物理残差耦合或由物理方程确定松弛幅值的模型。",
            "2. 单点 `n1_tip_z_rand15, alpha=0.10`：暂不发布有效子空间灵敏度；后续先修复一阶最优性和载荷传播，再复核二维有效子空间。",
            "3. 当前阶段禁止用末端拟合成功作为灵敏度可信依据。若模型包含自由松弛变量，必须做零空间审计，证明任务残差能辨识这些变量。",
            "4. 下一轮优先级：单点方向做有效子空间的物理闭合修复；两点方向停止简单权重、尺度和低维松弛扫描。",
            "",
            "## 下一步计划",
            "",
            "1. 以单点 n1 为主线，构造不依赖自由松弛的有效子空间门禁实验。",
            "2. 对两点 n2 只保留不可发布边界和模型修复议题，不再进入扰动灵敏度扫描。",
            "3. 若后续两点模型重新引入松弛变量，必须同时报告零空间、有效秩和载荷传播比例。",
            "",
        ]
    )


def main() -> None:
    p2c56 = _load(P2C56)
    p2c59 = _load(P2C59)
    p2c60 = _load(P2C60)
    p2c61 = _load(P2C61)
    p2c62 = _load(P2C62)

    p2c59_best = _best_by(p2c59["summaries"], "gate_violation_score")
    p2c60_best = _best_by(p2c60["summaries"], "gate_violation_score")
    p2c61_base = next(row for row in p2c61["summaries"] if row["mode"] == "base_6d")
    p2c61_load = next(row for row in p2c61["summaries"] if row["mode"] == "load_scaled")
    p2c62_task = next(row for row in p2c62["analyses"] if row["label"] == "task_fit_rank_deficient")
    n1_summary = p2c56["summary"]

    boundary_rows = [
        {
            "object": "单点 n1",
            "task_level": "有效子空间末端位置",
            "publish_status": "不可发布，仅诊断",
            "passed_items": "有效秩约为 2，提示存在二维可控方向",
            "failed_items": "一阶最优性、条件数、接口连续、载荷传播",
            "main_evidence": f"stationarity={n1_summary['stationarity_norm']:.3e}, condition={n1_summary['jacobian_condition']:.3e}, force_ratio={n1_summary['max_force_ratio_to_rigid_load']:.3e}",
            "engineering_recommendation": "先修复物理闭合，再复核二维有效子空间；不得接入外层 IK",
        },
        {
            "object": "两点 n2",
            "task_level": "三维末端位置",
            "publish_status": "不可发布，已形成失败边界",
            "passed_items": "末端位置拟合可达；部分显式连续样本一阶最优性和载荷传播可接近门禁",
            "failed_items": "条件数病态、秩亏、不可辨识松弛方向、载荷传播失配",
            "main_evidence": f"null_slack={p2c62_task['null_slack_fraction']:.3f}, null_tip_change={p2c62_task['null_tip_change_per_1e3']:.3e} m, force_ratio={p2c62_task['max_interface_force_jump_over_rigid_force']:.3e}",
            "engineering_recommendation": "停止扰动灵敏度扫描；转入显式物理耦合或物理方程确定松弛幅值",
        },
    ]

    summary = {
        "experiment": OUT_DIR.name,
        "boundary_rows": boundary_rows,
        "n2": {
            "p2c59_best_condition": p2c59_best["final_condition"],
            "p2c59_best_stationarity": p2c59_best["final_stationarity_norm"],
            "p2c59_best_continuity": p2c59_best["final_continuity_norm"],
            "p2c59_best_force_ratio": p2c59_best["final_force_ratio_to_rigid_load"],
            "p2c59_best_position_error_m": p2c59_best["final_position_error_m"],
            "p2c60_best_physical_condition": p2c60_best["final_physical_condition"],
            "p2c60_best_scaled_condition": p2c60_best["final_scaled_condition"],
            "p2c61_base_position_error_m": p2c61_base["best_position_error_m"],
            "p2c61_load_position_error_m": p2c61_load["best_position_error_m"],
            "p2c61_load_force_ratio": p2c61_load["best_force_ratio_to_rigid_load"],
            "p2c62_null_slack_fraction": p2c62_task["null_slack_fraction"],
            "p2c62_null_tip_change_m": p2c62_task["null_tip_change_per_1e3"],
        },
        "n1": {
            "stationarity": n1_summary["stationarity_norm"],
            "condition": n1_summary["jacobian_condition"],
            "continuity": n1_summary["continuity_norm"],
            "force_ratio": n1_summary["max_force_ratio_to_rigid_load"],
            "diagnostic_rank": n1_summary["diagnostic_subspace_rank"],
        },
        "git": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "status_short": _run(["git", "status", "--short"]),
        },
    }
    OUT_DIR.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_DIR / "boundary_summary.csv", boundary_rows)
    failures = OUT_DIR / "failure_events.jsonl"
    failures.write_text("", encoding="utf-8")
    for row in boundary_rows:
        _append_jsonl(failures, {"failure": row["publish_status"], "object": row["object"], "evidence": row})
    OUT_DIR.joinpath("record.md").write_text(_record(summary), encoding="utf-8")
    print(json.dumps({"boundary_rows": boundary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
