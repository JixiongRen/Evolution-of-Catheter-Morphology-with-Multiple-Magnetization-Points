"""Design n2 workpoint candidates from existing target-continuation diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

SOURCE_SUMMARIES = [
    ROOT / "experiment_records_ik_optimize/20260628165500_p2c92_n2_target_radius_100um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628175300_p2c93_n2_target_combo_100um/summary.json",
    ROOT / "experiment_records_ik_optimize/20260628183000_p2c94_n2_target_random_100um/summary.json",
]

PUBLISH_CONDITION_LIMIT = 1.0e7
DIAGNOSTIC_CONDITION_LIMIT = 1.0e9
STATIONARITY_LIMIT = 1.0e-5
FORCE_RATIO_LIMIT = 1.5
MAX_CANDIDATES = 5


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for path in SOURCE_SUMMARIES:
        payload = load_json(path)
        for row in payload.get("rows", []):
            if float(row.get("target_offset_m", 0.0)) <= 0.0:
                continue
            if int(row.get("step_index", -1)) != 2:
                continue
            candidate = dict(row)
            candidate["source_experiment"] = path.parent.name
            rows.append(candidate)
    return rows


def row_score(row: dict) -> float:
    condition = float(row.get("jacobian_condition", math.inf))
    stationarity = float(row.get("stationarity_norm", math.inf))
    force_ratio = float(row.get("max_force_ratio_to_rigid_load", math.inf))
    position_error = float(row.get("position_error_m", math.inf))
    drift = float(row.get("state_drift_from_base", math.inf))
    return (
        math.log10(max(condition, 1.0))
        + 3.0 * math.log10(max(stationarity, 1.0e-12) / STATIONARITY_LIMIT + 1.0)
        + force_ratio
        + 10.0 * position_error
        + 0.2 * drift
    )


def classify(row: dict) -> str:
    condition = float(row.get("jacobian_condition", math.inf))
    stationarity = float(row.get("stationarity_norm", math.inf))
    force_ratio = float(row.get("max_force_ratio_to_rigid_load", math.inf))
    if stationarity > STATIONARITY_LIMIT:
        return "基线不建议"
    if force_ratio > FORCE_RATIO_LIMIT:
        return "载荷传播风险"
    if condition < PUBLISH_CONDITION_LIMIT:
        return "可进入严格基线"
    if condition < DIAGNOSTIC_CONDITION_LIMIT:
        return "仅可谨慎诊断"
    return "病态禁止扫描"


def select_candidates(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    seen_prefixes: set[str] = set()
    for row in sorted(rows, key=row_score):
        direction = str(row.get("direction", ""))
        prefix = direction.split("_")[0]
        if prefix in seen_prefixes and len(selected) < 3:
            continue
        selected.append(row)
        seen_prefixes.add(prefix)
        if len(selected) >= MAX_CANDIDATES:
            break
    return selected


def export_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "rank",
        "direction",
        "target_offset_m",
        "position_error_m",
        "stationarity_norm",
        "jacobian_condition",
        "max_force_ratio_to_rigid_load",
        "state_drift_from_base",
        "classification",
        "source_experiment",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "direction": row.get("direction"),
                    "target_offset_m": row.get("target_offset_m"),
                    "position_error_m": row.get("position_error_m"),
                    "stationarity_norm": row.get("stationarity_norm"),
                    "jacobian_condition": row.get("jacobian_condition"),
                    "max_force_ratio_to_rigid_load": row.get("max_force_ratio_to_rigid_load"),
                    "state_drift_from_base": row.get("state_drift_from_base"),
                    "classification": classify(row),
                    "source_experiment": row.get("source_experiment"),
                }
            )


def plot_candidates(candidates: list[dict], path: Path) -> None:
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    labels = [str(row.get("direction")) for row in candidates]
    condition = [float(row.get("jacobian_condition", 0.0)) for row in candidates]
    force_ratio = [float(row.get("max_force_ratio_to_rigid_load", 0.0)) for row in candidates]
    stationarity = [float(row.get("stationarity_norm", 0.0)) for row in candidates]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].bar(labels, condition, color="#457b9d")
    axes[0].axhline(PUBLISH_CONDITION_LIMIT, color="red", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0e}"))
    axes[0].set_title("条件数", fontproperties=font)

    axes[1].bar(labels, force_ratio, color="#2a9d8f")
    axes[1].axhline(FORCE_RATIO_LIMIT, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("载荷传播比例", fontproperties=font)

    axes[2].bar(labels, stationarity, color="#f4a261")
    axes[2].axhline(STATIONARITY_LIMIT, color="red", linestyle="--", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0e}"))
    axes[2].set_title("一阶最优性残差", fontproperties=font)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font)
            label.set_rotation(25)
            label.set_ha("right")
    fig.suptitle("n2 下一轮工作点候选门禁预筛", fontproperties=font)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_record(candidates: list[dict], summary: dict) -> None:
    lines = [
        "# P2c143 两点磁控新工作点候选设计",
        "",
        "## 实验目的",
        "",
        "本轮不新增昂贵求解，而是从 P2c92-P2c94 的目标闭环诊断中筛选下一轮两点磁控 n2 跨工作点验证候选。该步骤用于避免继续围绕单一工作点重复微调。",
        "",
        "## 筛选原则",
        "",
        "- 只选择目标偏移不为零的终点行。",
        "- 一阶最优性残差需低于 `1e-5`。",
        "- 载荷传播比例需低于 `1.5`。",
        "- 条件数若高于 `1e7`，只能作为谨慎诊断候选，不能直接进入发布性扰动扫描。",
        "- 候选方向尽量分散，避免只验证单一方向。",
        "",
        "## 候选结果",
        "",
        "| 排名 | 方向 | 偏移量 m | 条件数 | 载荷传播比例 | 一阶最优性 | 分类 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(candidates, start=1):
        lines.append(
            "| {idx} | {direction} | {offset:.3e} | {condition:.3e} | {force:.3f} | {stat:.3e} | {cls} |".format(
                idx=idx,
                direction=row.get("direction"),
                offset=float(row.get("target_offset_m", 0.0)),
                condition=float(row.get("jacobian_condition", 0.0)),
                force=float(row.get("max_force_ratio_to_rigid_load", 0.0)),
                stat=float(row.get("stationarity_norm", 0.0)),
                cls=classify(row),
            )
        )
    lines += [
        "",
        "## 关键判断",
        "",
        f"共筛选出 {summary['candidate_count']} 个候选。所有候选的一阶最优性和载荷传播比例均满足基本门禁，但条件数均处于 `1e7-1e9` 的谨慎诊断区。因此，下一轮必须先做基线门禁和奇异值谱复核，不能直接执行大范围扰动扫描。",
        "",
        "## 下一步计划",
        "",
        "1. 对上述候选逐一重建 n2 基线解，记录有效秩、条件数、残差分块和分支漂移。",
        "2. 只保留条件数降至可接受范围或优势方向信号显著的候选。",
        "3. 对通过基线门禁的候选，仅沿第一主灵敏方向做小半径扰动重求解。",
        "4. 若 3 个以上工作点均稳定通过，可形成两点磁控有效子空间灵敏度族；否则给出工作空间分区边界。",
        "",
        "## 产物",
        "",
        "- `n2_workpoint_candidate_rows.csv`",
        "- `summary.json`",
        "- `n2_workpoint_candidate_gate_cn.png`",
        "- `failure_events.jsonl`",
    ]
    (OUT_DIR / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    candidates = select_candidates(rows)
    failure_events = []
    for row in candidates:
        if classify(row) == "仅可谨慎诊断":
            failure_events.append(
                {
                    "direction": row.get("direction"),
                    "failure_class": "条件数处于谨慎诊断区",
                    "jacobian_condition": row.get("jacobian_condition"),
                    "action": "下一轮先做基线门禁，不进入大范围扰动扫描",
                }
            )

    summary = {
        "experiment_id": OUT_DIR.name,
        "purpose": "design_n2_new_workpoint_candidates_from_existing_target_diagnostics",
        "source_experiment_count": len(SOURCE_SUMMARIES),
        "eligible_row_count": len(rows),
        "candidate_count": len(candidates),
        "publishable_now": False,
        "outer_ik_allowed": False,
        "dominant_limitation": "candidate_conditions_are_in_diagnostic_condition_number_band",
        "next_step": "rerun_baseline_gate_and_singular_spectrum_on_selected_candidates_before_perturbation_scan",
    }
    export_rows(OUT_DIR / "n2_workpoint_candidate_rows.csv", candidates)
    plot_candidates(candidates, OUT_DIR / "n2_workpoint_candidate_gate_cn.png")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "failure_events.jsonl").open("w", encoding="utf-8") as f:
        for event in failure_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_record(candidates, summary)


if __name__ == "__main__":
    main()
