"""Build a provenance-aware success ledger for supervised IK runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


REPO = Path(__file__).resolve().parents[1]
FORMAL_ROOT = REPO / "DELIVER_TO_EXP" / "formal_experiments"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _font() -> FontProperties:
    if FONT_PATH.exists():
        return FontProperties(fname=str(FONT_PATH))
    return FontProperties()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _experiment_root(summary_path: Path) -> Path:
    parts = summary_path.parts
    if "formal_experiments" not in parts:
        return summary_path.parent
    idx = parts.index("formal_experiments")
    if idx + 1 < len(parts):
        return Path(*parts[: idx + 2])
    return summary_path.parent


def _classify_row(row: dict[str, Any]) -> tuple[str, str, str]:
    """Classify one IK result by its evidence role and initialization source."""
    warm = str(row.get("warm_start_source") or "").strip()
    initial_branch_enabled = _truthy(row.get("initial_branch_state_enabled"))
    branch_warm = _truthy(row.get("branch_state_warm_start"))
    zero_iter = _truthy(row.get("zero_iter_pass") or row.get("no_op_pass"))
    auxiliary = _truthy(row.get("auxiliary_row")) or str(row.get("phase", "")).endswith("_basin")

    if auxiliary:
        return "stage_auxiliary", "阶段辅助", "阶段中间结果，不能进入最终成功率"
    if initial_branch_enabled or branch_warm or warm == "manual_branch_state":
        return "branch_state_reload", "分支状态重放", "加载历史内部平衡状态，只能作为分支存在性证据"
    if zero_iter:
        return "zero_iter_record_validation", "零迭代记录验证", "零迭代通过不能证明外层算法能力"
    if warm in {"cold_start_formal", "cold_start_gate", "cold_start_diagnostic"}:
        return "cold_start", "冷启动", "标准初始状态或冷启动诊断"
    if warm == "cold_start_after_failure":
        return "cold_restart_after_failure", "失败后冷重启", "前一点失败后重新冷启动，需单独统计"
    if warm.startswith("last_success_pt_"):
        return "neighbor_continuation", "邻点延拓", "使用上一成功点作为初值，不是独立冷启动"
    if warm == "last_success_before_failed_point":
        return "stale_success_continuation", "旧成功点延拓", "前一点失败后仍继承上一成功点，路径依赖更强"
    if warm == "manual_initial_state":
        return "manual_initial_state", "人工初值", "使用人工指定控制初值，不能计入冷启动成功率"
    if warm.startswith("two_stage_"):
        return "two_stage_flow_validation", "两阶段流程验证", "两阶段流程结果需与算法统计分账"
    if warm:
        return "other_warm_start", "其他热启动", f"未注册热启动来源：{warm}"
    return "unknown_source", "来源不明", "缺少初值来源字段，不能计入严格冷启动成功率"


def _evidence_quality(row: dict[str, Any], category: str) -> tuple[str, str]:
    """Return conservative evidence quality labels."""
    required = ["warm_start_source", "err_true_mm", "success_5mm", "catheter_type", "target_index"]
    missing = [name for name in required if name not in row]
    if missing:
        return "旧格式或证据不足", "缺少字段：" + ",".join(missing)
    if category in {"branch_state_reload", "stage_auxiliary", "zero_iter_record_validation"}:
        return "流程证据", "不能作为算法成功率"
    if category in {"neighbor_continuation", "stale_success_continuation", "manual_initial_state", "two_stage_flow_validation"}:
        return "路径依赖证据", "可证明延拓或候选有效，不能证明冷启动"
    if category in {"cold_start", "cold_restart_after_failure"}:
        return "算法证据", "可按对应类别单独统计"
    return "证据不足", "需要人工复核"


def _row_to_record(summary_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    category, category_zh, category_reason = _classify_row(row)
    quality, quality_reason = _evidence_quality(row, category)
    success_5mm = _truthy(row.get("success_5mm"))
    algorithm_stats = _truthy(row.get("algorithm_statistics_included"))
    formal_stats = _truthy(row.get("formal_statistics_included"))
    timed_out = _truthy(row.get("timed_out"))
    gpu_valid = str(row.get("runtime_platform") or "").lower() in {"gpu", "cuda"} or "cuda" in str(
        row.get("runtime_device_repr") or ""
    ).lower()
    branch_loaded = _truthy(row.get("initial_branch_state_enabled")) or _truthy(row.get("branch_state_warm_start"))
    strict_cold = bool(
        category == "cold_start"
        and success_5mm
        and not timed_out
        and not branch_loaded
        and quality == "算法证据"
    )
    cold_restart = bool(
        category == "cold_restart_after_failure"
        and success_5mm
        and not timed_out
        and not branch_loaded
        and quality == "算法证据"
    )
    continuation_success = bool(
        category in {"neighbor_continuation", "stale_success_continuation"}
        and success_5mm
        and not timed_out
        and not branch_loaded
    )
    branch_reload_success = bool(category == "branch_state_reload" and success_5mm and not timed_out)

    exp_root = _experiment_root(summary_path)
    run_dir = str(row.get("run_dir") or "")
    branch_loop = _branch_loop_summary(Path(run_dir), row) if run_dir else {}
    return {
        "experiment_id": exp_root.name,
        "summary_path": str(summary_path.resolve()),
        "catheter_type": row.get("catheter_type"),
        "phase": row.get("phase"),
        "experiment_type": row.get("experiment_type"),
        "target_index": row.get("target_index"),
        "err_true_mm": row.get("err_true_mm"),
        "err_flex_mm": row.get("err_flex_mm"),
        "success_5mm": success_5mm,
        "success_10mm": _truthy(row.get("success_10mm")),
        "classification": category,
        "classification_zh": category_zh,
        "classification_reason": category_reason,
        "evidence_quality": quality,
        "evidence_quality_reason": quality_reason,
        "strict_cold_start_success": strict_cold,
        "cold_restart_success": cold_restart,
        "continuation_success": continuation_success,
        "branch_reload_success": branch_reload_success,
        "algorithm_statistics_included_raw": algorithm_stats,
        "formal_statistics_included_raw": formal_stats,
        "warm_start_source": row.get("warm_start_source"),
        "branch_state_warm_start": _truthy(row.get("branch_state_warm_start")),
        "initial_branch_state_enabled": _truthy(row.get("initial_branch_state_enabled")),
        "zero_iter_pass": _truthy(row.get("zero_iter_pass") or row.get("no_op_pass")),
        "auxiliary_row": _truthy(row.get("auxiliary_row")),
        "report_stop_reason": row.get("report_stop_reason") or row.get("stop_reason"),
        "ik_stop_reason": row.get("ik_stop_reason"),
        "outer_iter_count": row.get("outer_iter_count"),
        "accept_count": row.get("accept_count"),
        "reject_count": row.get("reject_count"),
        "runtime_platform": row.get("runtime_platform"),
        "runtime_device_kind": row.get("runtime_device_kind"),
        "gpu_valid": gpu_valid,
        "centerline_hash": row.get("centerline_hash"),
        "z_star_hash": row.get("z_star_hash"),
        "branch_npz_path": row.get("branch_npz_path"),
        "branch_solver_hash": row.get("branch_solver_hash"),
        "branch_sensitivity_hash": row.get("branch_sensitivity_hash"),
        "branch_loop_stall_detected": branch_loop.get("detected", False),
        "branch_loop_reason": branch_loop.get("reason"),
        "branch_loop_repeated_hash_count": branch_loop.get("repeated_hash_count"),
        "branch_loop_final_error_mm": branch_loop.get("final_error_mm"),
        "branch_loop_best_error_mm": branch_loop.get("best_error_mm"),
        "branch_loop_error_gap_mm": branch_loop.get("error_gap_mm"),
        "run_dir": run_dir,
    }


def _branch_loop_summary(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Detect repeated-branch stagnation from saved branch-state manifests."""
    manifest_path = run_dir / "branch_states" / "branch_state_manifest.jsonl"
    if not manifest_path.exists():
        return {
            "detected": False,
            "reason": "no_branch_manifest",
            "repeated_hash_count": 0,
            "final_error_mm": _float_or_none(row.get("err_true_mm")),
            "best_error_mm": _float_or_none(row.get("best_err_true_mm") or row.get("err_true_mm")),
            "error_gap_mm": None,
        }
    entries: list[dict[str, Any]] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        return {
            "detected": False,
            "reason": f"manifest_parse_failed:{exc}",
            "repeated_hash_count": 0,
            "final_error_mm": _float_or_none(row.get("err_true_mm")),
            "best_error_mm": _float_or_none(row.get("best_err_true_mm") or row.get("err_true_mm")),
            "error_gap_mm": None,
        }
    final_hash = str(row.get("centerline_hash") or "")
    hash_counts = Counter(str(item.get("centerline_hash") or "") for item in entries if item.get("centerline_hash"))
    repeated_count = int(hash_counts.get(final_hash, 0)) if final_hash else 0
    errors = [
        value
        for value in (_float_or_none(item.get("terminal_error_mm")) for item in entries)
        if value is not None
    ]
    final_error = _float_or_none(row.get("err_true_mm"))
    best_error = min(errors) if errors else _float_or_none(row.get("best_err_true_mm") or row.get("err_true_mm"))
    error_gap = None
    if final_error is not None and best_error is not None:
        error_gap = max(0.0, float(final_error) - float(best_error))
    stop_reason = str(row.get("ik_stop_reason") or row.get("report_stop_reason") or row.get("stop_reason") or "")
    detected = bool(
        repeated_count >= 2
        and error_gap is not None
        and error_gap <= 0.1
        and stop_reason == "effective_step_below_tolerance"
    )
    reason = "branch_hash_repeats_with_step_collapse" if detected else "not_detected"
    return {
        "detected": detected,
        "reason": reason,
        "repeated_hash_count": repeated_count,
        "final_error_mm": final_error,
        "best_error_mm": best_error,
        "error_gap_mm": error_gap,
    }


def _read_summary_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_tables(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec.get("experiment_id")), str(rec.get("catheter_type")))].append(rec)

    rows: list[dict[str, Any]] = []
    for (experiment_id, catheter), items in sorted(grouped.items()):
        classes = Counter(str(item["classification"]) for item in items)
        rows.append(
            {
                "experiment_id": experiment_id,
                "catheter_type": catheter,
                "n_rows": len(items),
                "n_success_5mm_raw": sum(1 for item in items if item["success_5mm"]),
                "n_strict_cold_start_success": sum(1 for item in items if item["strict_cold_start_success"]),
                "n_cold_restart_success": sum(1 for item in items if item["cold_restart_success"]),
                "n_continuation_success": sum(1 for item in items if item["continuation_success"]),
                "n_branch_reload_success": sum(1 for item in items if item["branch_reload_success"]),
                "n_branch_loop_stall": sum(1 for item in items if item["branch_loop_stall_detected"]),
                "n_algorithm_evidence": sum(1 for item in items if item["evidence_quality"] == "算法证据"),
                "n_path_dependent_evidence": sum(1 for item in items if item["evidence_quality"] == "路径依赖证据"),
                "n_process_evidence": sum(1 for item in items if item["evidence_quality"] == "流程证据"),
                "class_counts_json": json.dumps(dict(classes), ensure_ascii=False, sort_keys=True),
            }
        )

    aggregate = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_records": len(records),
        "n_experiment_catheter_groups": len(grouped),
        "classification_counts": dict(Counter(str(item["classification"]) for item in records)),
        "raw_success_5mm": sum(1 for item in records if item["success_5mm"]),
        "strict_cold_start_success": sum(1 for item in records if item["strict_cold_start_success"]),
        "cold_restart_success": sum(1 for item in records if item["cold_restart_success"]),
        "continuation_success": sum(1 for item in records if item["continuation_success"]),
        "branch_reload_success": sum(1 for item in records if item["branch_reload_success"]),
        "branch_loop_stall": sum(1 for item in records if item["branch_loop_stall_detected"]),
    }
    return rows, aggregate


def _write_record(path: Path, summary: dict[str, Any], source_paths: list[Path]) -> None:
    lines = [
        "# A1/A2 有监督逆运动学成功台账重分类记录",
        "",
        "## 一、目的",
        "",
        "本轮审计用于把历史正式实验和诊断实验中的成功结果重新分账，区分冷启动、失败后冷重启、邻点延拓、分支状态重放、零迭代记录验证和阶段辅助结果，避免把路径依赖结果误写成独立冷启动算法成功。",
        "",
        "## 二、输入范围",
        "",
        f"- 扫描 summary.csv 数量：{len(source_paths)}",
        f"- 结果行数量：{summary['n_records']}",
        "",
        "## 三、总览结果",
        "",
        f"- 原始 5 mm 成功行数：{summary['raw_success_5mm']}",
        f"- 严格冷启动成功行数：{summary['strict_cold_start_success']}",
        f"- 失败后冷重启成功行数：{summary['cold_restart_success']}",
        f"- 邻点或旧成功点延拓成功行数：{summary['continuation_success']}",
        f"- 分支状态重放成功行数：{summary['branch_reload_success']}",
        f"- 后验分支回环停滞行数：{summary['branch_loop_stall']}",
        "",
        "分类计数：",
        "",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        lines.append(f"- `{key}`：{value}")
    lines.extend(
        [
            "",
            "## 四、解释原则",
            "",
            "1. 只有未加载外部分支状态、初值来源为标准冷启动，并且真实末端误差进入 5 mm 的结果，才计入严格冷启动成功。",
            "2. `last_success_*` 和 `last_success_before_failed_point` 只证明邻点延拓或路径依赖有效，不证明独立点冷启动可达。",
            "3. 加载历史分支状态得到的结果只说明该平衡分支存在，不能计入冷启动成功率。",
            "4. 缺少关键字段的旧格式结果被标为证据不足或旧格式推断，不作为严格成功证据。",
            "",
            "## 五、输出文件",
            "",
            "- `success_ledger.csv`：逐点重分类台账。",
            "- `success_ledger_summary.csv`：按实验和导管类型聚合的摘要。",
            "- `success_ledger_summary.json`：机器可读总览。",
            "- `success_ledger_overview.png/svg`：成功证据类别概览图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_overview(out_dir: Path, summary: dict[str, Any]) -> None:
    """Plot a Chinese summary chart for success evidence classes."""
    labels = ["原始5mm成功", "严格冷启动", "失败后冷重启", "邻点/旧成功点延拓", "分支状态重放"]
    values = [
        summary["raw_success_5mm"],
        summary["strict_cold_start_success"],
        summary["cold_restart_success"],
        summary["continuation_success"],
        summary["branch_reload_success"],
    ]
    colors = ["#6B7280", "#2563EB", "#059669", "#D97706", "#DC2626"]
    font = _font()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontproperties=font, rotation=15, ha="right")
    ax.set_ylabel("结果行数量", fontproperties=font)
    ax.set_title("A1/A2 有监督逆运动学成功证据重分类", fontproperties=font, fontsize=15)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontproperties=font,
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "success_ledger_overview.png", dpi=180)
    fig.savefig(out_dir / "success_ledger_overview.svg")
    plt.close(fig)


def build_ledger(summary_paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(summary_paths):
        for row in _read_summary_rows(summary_path):
            records.append(_row_to_record(summary_path, row))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=FORMAL_ROOT,
        help="Root directory that contains formal experiment outputs.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        default=[],
        help="Specific summary.csv file to include. May be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the audit ledger outputs.")
    args = parser.parse_args()

    summary_paths = list(args.summary_csv)
    if not summary_paths:
        summary_paths = sorted(args.formal_root.glob("*/summary.csv"))
    summary_paths = [path.resolve() for path in summary_paths if path.exists()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = build_ledger(summary_paths)
    summary_rows, aggregate = _summary_tables(records)

    _write_csv(args.out_dir / "success_ledger.csv", records)
    _write_csv(
        args.out_dir / "branch_loop_stall_rows.csv",
        [record for record in records if record.get("branch_loop_stall_detected")],
    )
    _write_csv(args.out_dir / "success_ledger_summary.csv", summary_rows)
    (args.out_dir / "success_ledger_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_record(args.out_dir / "record.md", aggregate, summary_paths)
    _plot_overview(args.out_dir, aggregate)
    (args.out_dir / "source_summaries.txt").write_text(
        "\n".join(str(path) for path in summary_paths) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
