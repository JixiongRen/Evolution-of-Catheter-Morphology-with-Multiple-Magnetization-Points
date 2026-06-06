"""Synthesize sensitivity reliability evidence from recent IK experiments."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT / "experiment_records_ik_optimize"

EXPERIMENTS = {
    "P2c15": "20260605125315_p2c15_slack_suppressed_sensitivity_audit",
    "P2c16": "20260605132600_p2c16_alpha_boundary_phase_map",
    "P2c17": "20260605145500_p2c17_refined_boundary_path_dependence",
    "P2c18": "20260605153000_p2c18_lower_alpha_breakpoint_load_audit",
    "P2c19": "20260606103000_p2c19_ultrafine_lower_alpha_breakpoint",
    "P2c20": "20260606143000_p2c20_alpha0375_branch_clustering_roundtrip",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"true", "1", "yes"}


def _as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _failure_modes(row: dict[str, Any]) -> list[str]:
    modes: list[str] = []
    if _as_float(row, "max_stage_stationarity", _as_float(row, "stationarity_norm")) > 1.0e-5:
        modes.append("一阶最优性退化")
    if _as_float(row, "max_stage_condition", _as_float(row, "jacobian_condition")) > 1.0e10:
        modes.append("数值病态或局部不可辨识")
    if _as_float(row, "max_linearization_error") > 5.0e-2:
        modes.append("局部线性近似失效")
    if _as_float(row, "branch_rms_max") > 1.0e-3:
        modes.append("分支不一致或分支切换")
    if _as_float(row, "max_slack_force_over_rigid_force") > 2.0:
        modes.append("刚性磁体段载荷传播失配")
    if _as_float(row, "position_error_m", _as_float(row, "max_position_error_m")) > 0.05 * 0.11:
        modes.append("可达域或末端约束失配")
    return modes or ["证据不足或边界临界"]


def _primary_mode(modes: list[str]) -> str:
    priority = [
        "分支不一致或分支切换",
        "一阶最优性退化",
        "刚性磁体段载荷传播失配",
        "局部线性近似失效",
        "数值病态或局部不可辨识",
        "可达域或末端约束失配",
    ]
    for item in priority:
        if item in modes:
            return item
    return modes[0]


def _load_case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in ["P2c15", "P2c16", "P2c17", "P2c18", "P2c19"]:
        exp_dir = BASE_DIR / EXPERIMENTS[tag]
        data = _read_json(exp_dir / "summary.json")
        for row in data["case_summaries"]:
            publishable = _as_bool(row.get("publishable_gate"))
            modes = [] if publishable else _failure_modes(row)
            rows.append(
                {
                    "source_experiment": tag,
                    "source_dir": str(exp_dir),
                    "case": row.get("case", "n3_mixed_rand20"),
                    "alpha": _as_float(row, "alpha"),
                    "eps_alpha": _as_float(row, "eps_alpha"),
                    "classification": row.get("classification", "trusted" if publishable else "diagnostic"),
                    "publishable_gate": publishable,
                    "strict_independent_gate": _as_bool(row.get("strict_independent_gate", row.get("independent_seed_gate"))),
                    "continuation_gate": _as_bool(row.get("continuation_gate")),
                    "sensitivity_norm_m_per_alpha": _as_float(row, "sensitivity_norm_m_per_alpha"),
                    "max_linearization_error": _as_float(row, "max_linearization_error"),
                    "branch_rms_max": _as_float(row, "branch_rms_max"),
                    "max_stage_stationarity": _as_float(row, "max_stage_stationarity"),
                    "max_stage_condition": _as_float(row, "max_stage_condition"),
                    "max_slack_force_over_rigid_force": _as_float(row, "max_slack_force_over_rigid_force"),
                    "max_position_error_m": _as_float(row, "max_position_error_m"),
                    "failure_modes": ";".join(modes),
                    "primary_failure_mode": "" if publishable else _primary_mode(modes),
                    "evidence_type": "high_confidence_sensitivity" if publishable else "diagnostic_only",
                }
            )
    return rows


def _load_seed_rows() -> list[dict[str, Any]]:
    exp_dir = BASE_DIR / EXPERIMENTS["P2c20"]
    rows = []
    for row in _read_csv(exp_dir / "seed_rows.csv"):
        gate = _as_bool(row.get("baseline_gate"))
        modes = [] if gate else _failure_modes(row)
        rows.append(
            {
                "source_experiment": "P2c20",
                "source_dir": str(exp_dir),
                "alpha": _as_float(row, "alpha"),
                "seed_name": row["seed_name"],
                "seed_class": row["seed_class"],
                "physical_branch_gate": gate,
                "stationarity_norm": _as_float(row, "stationarity_norm"),
                "jacobian_condition": _as_float(row, "jacobian_condition"),
                "position_error_m": _as_float(row, "position_error_m"),
                "max_slack_force_over_rigid_force": _as_float(row, "max_slack_force_over_rigid_force"),
                "failure_modes": ";".join(modes),
                "primary_failure_mode": "" if gate else _primary_mode(modes),
                "evidence_type": "physical_branch_candidate" if gate else "diagnostic_only",
            }
        )
    return rows


def _recommendations(case_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    trusted = sorted({round(row["alpha"], 6) for row in case_rows if row["publishable_gate"]})
    diagnostic = sorted({round(row["alpha"], 6) for row in case_rows if not row["publishable_gate"]})
    physical_seeds = [row for row in seed_rows if row["physical_branch_gate"]]
    strict_physical = [row for row in physical_seeds if row["seed_class"] == "strict_independent"]
    return {
        "verified_conclusions": [
            "在 n3_mixed_rand20 工况下，alpha=0.3752 至 0.425 附近已有高置信敏感度证据，但不同实验覆盖密度不同。",
            "alpha=0.3750 处存在物理分支，但该分支吸引域很窄，不能作为高置信敏感度点直接发布。",
            "末端位置误差本身不足以判定敏感度质量；必须同时检查一阶最优性、分支一致性和刚性磁体段载荷传播一致性。",
        ],
        "working_alpha_guidance": {
            "trusted_alpha_samples": trusted,
            "diagnostic_alpha_samples": diagnostic,
            "lower_boundary_statement": "独立高置信下界目前定位在 0.3750 < alpha <= 0.3752；alpha=0.3750 属于临界分支吸引域过窄点。",
            "upper_boundary_statement": "已有证据显示 0.425 仍可信，0.435 及以上表现为路径依赖或临界诊断，应继续加密验证。",
        },
        "configuration_guidance": [
            "低电流或弱磁驱动区域需要先做分支可进入性判定，再计算敏感度。",
            "三磁化段混合磁化构型在边界区可能出现多解竞争，应优先采用逐级增载和多初值审计。",
            "若界面力松弛与刚段外力比值超过 2.0，应将结果归为载荷传播失配，不发布敏感度。",
            "若必须穿越 alpha=0.3750 附近，应使用路径延续控制，不建议独立点求解。",
        ],
        "p2c20_branch_evidence": {
            "seed_count": len(seed_rows),
            "physical_branch_count": len(physical_seeds),
            "strict_independent_physical_count": len(strict_physical),
        },
    }


def _plot_alpha_phase(case_rows: list[dict[str, Any]]) -> None:
    order = sorted(case_rows, key=lambda r: (r["alpha"], r["source_experiment"]))
    color = {
        "high_confidence_sensitivity": "green",
        "diagnostic_only": "orange",
    }
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for row in order:
        c = color[row["evidence_type"]]
        axes[0].scatter(row["alpha"], row["sensitivity_norm_m_per_alpha"], color=c, s=60)
        axes[1].scatter(row["alpha"], row["max_linearization_error"], color=c, s=60)
        axes[2].scatter(row["alpha"], row["max_stage_stationarity"], color=c, s=60)
        axes[3].scatter(row["alpha"], row["max_slack_force_over_rigid_force"], color=c, s=60)
    axes[1].axhline(5.0e-2, color="black", ls="--", lw=1)
    axes[2].axhline(1.0e-5, color="black", ls="--", lw=1)
    axes[3].axhline(2.0, color="black", ls="--", lw=1)
    axes[0].set_ylabel("敏感度范数 m/alpha")
    axes[1].set_ylabel("线性化误差")
    axes[2].set_ylabel("一阶最优性")
    axes[3].set_ylabel("载荷传播失配指标")
    axes[3].set_xlabel("电流缩放系数 alpha")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_reliability_phase_synthesis.png", dpi=180)
    plt.close(fig)


def _plot_failure_counts(case_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    counter = Counter()
    for row in case_rows + seed_rows:
        mode = row.get("primary_failure_mode")
        if mode:
            counter[mode] += 1
    labels = list(counter)
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(len(labels)), values, color="#9c4a1a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("样本数量")
    ax.set_title("失败归因主类型统计")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "failure_mode_counts.png", dpi=180)
    plt.close(fig)


def _plot_seed_branch(seed_rows: list[dict[str, Any]]) -> None:
    labels = [row["seed_name"] for row in seed_rows]
    colors = ["green" if row["physical_branch_gate"] else "red" for row in seed_rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    specs = [
        ("stationarity_norm", "一阶最优性", 1.0e-5),
        ("max_slack_force_over_rigid_force", "界面力松弛/刚段外力", 2.0),
        ("position_error_m", "末端位置误差 m", 0.05 * 0.11),
    ]
    x = np.arange(len(labels))
    for ax, (key, ylabel, gate) in zip(axes, specs):
        ax.bar(x, [row[key] for row in seed_rows], color=colors)
        ax.axhline(gate, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2c20_seed_branch_gate_synthesis.png", dpi=180)
    plt.close(fig)


def _plot_time_costs() -> None:
    labels = []
    values = []
    for tag, dirname in EXPERIMENTS.items():
        p = BASE_DIR / dirname / "summary.json"
        if not p.exists():
            continue
        data = _read_json(p)
        labels.append(tag)
        values.append(float(data.get("wall_s", 0.0)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3f6c8f")
    ax.set_ylabel("墙钟时间 s")
    ax.set_title("近期敏感度实验时间成本")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "time_cost_synthesis.png", dpi=180)
    plt.close(fig)


def _plot_representative_centerlines() -> None:
    specs = [
        (
            "临界 alpha=0.3750",
            BASE_DIR / EXPERIMENTS["P2c19"] / "alpha_0.3750_ultrafine_lower_breakpoint.npz",
            "orange",
        ),
        (
            "可信下界 alpha=0.3752",
            BASE_DIR / EXPERIMENTS["P2c19"] / "alpha_0.3752_ultrafine_lower_breakpoint.npz",
            "green",
        ),
        (
            "可信中区 alpha=0.400",
            BASE_DIR / EXPERIMENTS["P2c16"] / "alpha_0.400_eps_0.010_phase_map.npz",
            "blue",
        ),
        (
            "可信上界样本 alpha=0.425",
            BASE_DIR / EXPERIMENTS["P2c16"] / "alpha_0.425_eps_0.010_phase_map.npz",
            "purple",
        ),
    ]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, path, color in specs:
        data = np.load(path)
        pts = np.asarray(data["base_centerline"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=label, color=color)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, s=35)
    first = np.asarray(np.load(specs[0][1])["base_centerline"], dtype=float)[0]
    ax.scatter(first[0], first[1], first[2], color="black", s=45, label="基点")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("代表性中心线形态对比")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_centerline_synthesis.png", dpi=180)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    case_rows = _load_case_rows()
    seed_rows = _load_seed_rows()
    recommendations = _recommendations(case_rows, seed_rows)
    _write_csv(OUT_DIR / "case_reliability_rows.csv", case_rows)
    _write_csv(OUT_DIR / "seed_branch_rows.csv", seed_rows)

    failure_counter = Counter()
    for row in case_rows + seed_rows:
        if row.get("primary_failure_mode"):
            failure_counter[row["primary_failure_mode"]] += 1
    failure_rows = [{"failure_mode": k, "count": v} for k, v in sorted(failure_counter.items())]
    _write_csv(OUT_DIR / "failure_taxonomy_counts.csv", failure_rows)
    _plot_alpha_phase(case_rows)
    _plot_failure_counts(case_rows, seed_rows)
    _plot_seed_branch(seed_rows)
    _plot_time_costs()
    _plot_representative_centerlines()

    summary = {
        "experiment": OUT_DIR.name,
        "wall_s": float(time.perf_counter() - t0),
        "source_experiments": EXPERIMENTS,
        "overall": {
            "case_row_count": len(case_rows),
            "seed_row_count": len(seed_rows),
            "publishable_case_count": sum(row["publishable_gate"] for row in case_rows),
            "diagnostic_case_count": sum(not row["publishable_gate"] for row in case_rows),
            "physical_seed_count": sum(row["physical_branch_gate"] for row in seed_rows),
            "diagnostic_seed_count": sum(not row["physical_branch_gate"] for row in seed_rows),
            "failure_mode_count": dict(failure_counter),
        },
        "recommendations": recommendations,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in failure_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary["overall"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
