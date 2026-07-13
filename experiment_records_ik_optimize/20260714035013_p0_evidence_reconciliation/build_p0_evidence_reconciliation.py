"""Build the P0 evidence reconciliation report for A1/A2 supervised IK."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

MERGED_RUN = ROOT / (
    "DELIVER_TO_EXP/formal_experiments/"
    "20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40"
)
DIAG_DIR = ROOT / (
    "DELIVER_TO_EXP/formal_diagnostics/full40_merged/"
    "20260713060401_A1_A2_failure_diagnostics"
)
REACH_DIR = ROOT / (
    "DELIVER_TO_EXP/formal_diagnostics/reachability_sampling/"
    "20260713123103_A1_A2_reachability_sampling"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def bool_text(value: str) -> bool:
    return str(value).strip().lower() == "true"


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_mode(row: dict[str, str]) -> str:
    source = row.get("warm_start_source", "")
    run_dir = row.get("run_dir", "")
    if source == "manual_initial_state":
        return "人工分支种子"
    if "20260713063500_A1_A2_5mm_supervised_IK_no_gravity_merged_complete40" in run_dir:
        return "补跑合并"
    if source.startswith("last_success"):
        return "顺序热启动"
    if source.startswith("cold_start"):
        return "独立或冷启动"
    return "未分类"


def main() -> None:
    summary_rows = read_csv(MERGED_RUN / "summary.csv")
    diag_rows = read_csv(DIAG_DIR / "diagnostic_summary.csv")
    reach_rows = read_csv(REACH_DIR / "target_reachability_summary.csv")

    diag_by_key = {
        (row["catheter_type"], row["target_index"]): row for row in diag_rows
    }
    reach_by_key = {
        (row["catheter_type"], row["target_index"]): row for row in reach_rows
    }

    point_rows: list[dict[str, Any]] = []
    discrepancy_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        key = (row["catheter_type"], row["target_index"])
        diag = diag_by_key.get(key, {})
        reach = reach_by_key.get(key, {})
        mode = classify_mode(row)
        success_summary = bool_text(row.get("success_5mm", ""))
        success_diag = bool_text(diag.get("success_true_eps", ""))
        err_summary = float_or_none(row.get("err_true_mm", ""))
        err_diag = float_or_none(diag.get("err_true_mm", ""))
        err_delta = ""
        if err_summary is not None and err_diag is not None:
            err_delta = abs(err_summary - err_diag)
        consistent = success_summary == success_diag and (
            err_delta == "" or err_delta < 1e-6
        )
        out = {
            "catheter_type": row["catheter_type"],
            "target_index": row["target_index"],
            "err_true_mm": row["err_true_mm"],
            "success_5mm": row["success_5mm"],
            "experiment_mode": mode,
            "warm_start_source": row.get("warm_start_source", ""),
            "stop_reason": row.get("stop_reason", ""),
            "cls": row.get("cls", ""),
            "root_cause": diag.get("main_root_cause", ""),
            "diagnostic_confidence": diag.get("confidence_level", ""),
            "d0_gate_valid": diag.get("d0_gate_valid", ""),
            "reachability_conclusion": reach.get("d4_conclusion", ""),
            "history_nearest_mm": reach.get("history_nearest_mm", ""),
            "sample_nearest_mm": reach.get("sample_nearest_mm", ""),
            "summary_diagnostic_consistent": consistent,
            "run_dir": row.get("run_dir", ""),
        }
        point_rows.append(out)
        if not consistent:
            discrepancy_rows.append(out)

    by_catheter: dict[str, dict[str, Any]] = {}
    for catheter in sorted({row["catheter_type"] for row in point_rows}):
        rows = [row for row in point_rows if row["catheter_type"] == catheter]
        successes = [row for row in rows if row["success_5mm"] == "True"]
        by_catheter[catheter] = {
            "count": len(rows),
            "success_5mm_count": len(successes),
            "success_5mm_points": [int(row["target_index"]) for row in successes],
            "mode_counts": dict(Counter(row["experiment_mode"] for row in rows)),
            "success_by_mode": dict(
                Counter(row["experiment_mode"] for row in successes)
            ),
            "root_cause_counts": dict(Counter(row["root_cause"] for row in rows)),
            "stop_reason_counts": dict(Counter(row["stop_reason"] for row in rows)),
        }

    report_text = (MERGED_RUN / "report.md").read_text(encoding="utf-8")
    report_known_conflicts = []
    if "| A2 |" in report_text and "| A2 | 0/0 | formal_20pt_forced | 20 | 3 |" in report_text:
        report_known_conflicts.append(
            "合并目录 report.md 仍写 A2 为 3/20，未吸收 A2-pt06 补跑成功。"
        )
    if "auto_timeout" in report_text and "| 6 | formal_20pt | nan | nan |" in report_text:
        report_known_conflicts.append(
            "合并目录 report.md 仍保留 A2-pt06/07/08 的原始超时行。"
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "merged_run": str(MERGED_RUN),
        "diagnostic_dir": str(DIAG_DIR),
        "reachability_dir": str(REACH_DIR),
        "point_count": len(point_rows),
        "summary_diagnostic_discrepancy_count": len(discrepancy_rows),
        "by_catheter": by_catheter,
        "report_known_conflicts": report_known_conflicts,
        "authoritative_statement_zh": (
            "当前 P0 对账以 merged_complete40 的 summary.csv/summary.json "
            "和 full40_merged 诊断表为准；原 report.md 存在口径漂移，"
            "不得单独作为 A2 成功率或补跑点位结论来源。"
        ),
    }

    fieldnames = [
        "catheter_type",
        "target_index",
        "err_true_mm",
        "success_5mm",
        "experiment_mode",
        "warm_start_source",
        "stop_reason",
        "cls",
        "root_cause",
        "diagnostic_confidence",
        "d0_gate_valid",
        "reachability_conclusion",
        "history_nearest_mm",
        "sample_nearest_mm",
        "summary_diagnostic_consistent",
        "run_dir",
    ]
    write_csv(OUT_DIR / "p0_point_reconciliation.csv", point_rows, fieldnames)
    write_csv(OUT_DIR / "p0_discrepancies.csv", discrepancy_rows, fieldnames)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_markdown_report(summary, point_rows)
    (OUT_DIR / "record.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "commands.sh").write_text(
        "XLA_PYTHON_CLIENT_PREALLOCATE=false "
        "/home/ub22/miniconda3/envs/evolution_of_mcrs/bin/python "
        "build_p0_evidence_reconciliation.py\n",
        encoding="utf-8",
    )


def build_markdown_report(
    summary: dict[str, Any],
    point_rows: list[dict[str, Any]],
) -> str:
    by_catheter = summary["by_catheter"]
    lines = [
        "# A1/A2 有监督逆运动学 P0 证据对账记录",
        "",
        "## 一、实验目的",
        "",
        "本轮不重新运行逆运动学，而是对现有 A1/A2 40 点正式结果、诊断结论和可达性采样进行口径对账。目标是固定后续修正所依据的可信基线，并明确哪些旧报告不能单独作为结论来源。",
        "",
        "## 二、输入数据",
        "",
        f"- 合并正式结果：`{summary['merged_run']}`",
        f"- 合并诊断结果：`{summary['diagnostic_dir']}`",
        f"- 可达性采样结果：`{summary['reachability_dir']}`",
        "",
        "## 三、对账结论",
        "",
        f"- 点位总数：{summary['point_count']}。",
        f"- 正式汇总表与诊断表逐点成功状态不一致数量：{summary['summary_diagnostic_discrepancy_count']}。",
        f"- A1 当前合并口径为 {by_catheter['A1']['success_5mm_count']}/20 进入 5 mm，成功点为 {by_catheter['A1']['success_5mm_points']}。",
        f"- A2 当前合并口径为 {by_catheter['A2']['success_5mm_count']}/20 进入 5 mm，成功点为 {by_catheter['A2']['success_5mm_points']}。",
        "- A2-pt06 来自补跑并使用人工初始状态，应与独立求解成功率分账统计。",
        "- 原合并目录中的 `report.md` 存在口径漂移，不得单独作为 A2 成功率结论来源。",
        "",
        "## 四、报告冲突",
        "",
    ]
    if summary["report_known_conflicts"]:
        for item in summary["report_known_conflicts"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 未检测到已知文本冲突。")

    lines.extend(
        [
            "",
            "## 五、分导管摘要",
            "",
            "| 导管 | 5 mm 成功数 | 成功点 | 实验模式统计 | 失败主因统计 |",
            "|---|---:|---|---|---|",
        ]
    )
    for catheter in ["A1", "A2"]:
        data = by_catheter[catheter]
        lines.append(
            "| {cat} | {succ}/20 | {points} | {modes} | {causes} |".format(
                cat=catheter,
                succ=data["success_5mm_count"],
                points=data["success_5mm_points"],
                modes=json.dumps(data["mode_counts"], ensure_ascii=False),
                causes=json.dumps(data["root_cause_counts"], ensure_ascii=False),
            )
        )

    lines.extend(
        [
            "",
            "## 六、后续使用规则",
            "",
            "- 后续 P1/P2/P3 只能引用本目录 `p0_point_reconciliation.csv`、合并 `summary.csv/summary.json` 和 `full40_merged` 诊断表作为正式基线。",
            "- 若需要引用旧 `report.md`，必须同时说明它未吸收 A2-pt06/07/08 补跑结果。",
            "- 独立运行、顺序热启动、人工分支种子和补跑合并必须分账统计。",
            "- 对 A2 的阶段性结论必须同时给出“合并成功率”和“不含人工初始状态的成功率”。",
            "",
            "## 七、输出文件",
            "",
            "- `p0_point_reconciliation.csv`：逐点对账表。",
            "- `p0_discrepancies.csv`：正式汇总表与诊断表不一致的点位。本轮为空表示两者逐点一致。",
            "- `summary.json`：机器可读摘要。",
            "- `commands.sh`：生成本记录的命令。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

