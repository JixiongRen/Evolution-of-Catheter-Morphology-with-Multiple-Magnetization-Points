"""Summarize n1 failure mechanisms and repair boundaries."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk  # noqa: E402


SOURCE_PATHS = {
    "P2c99": ROOT / "experiment_records_ik_optimize/20260628215500_p2c99_n1_nonpublishable_boundary_synthesis/summary.json",
    "P2c110": ROOT / "experiment_records_ik_optimize/20260629030000_p2c110_n1_hard_force_scaled_continuity/summary.json",
    "P2c111": ROOT / "experiment_records_ik_optimize/20260629033000_p2c111_n1_beta_load_transfer/summary.json",
    "P2c112": ROOT / "experiment_records_ik_optimize/20260629040000_p2c112_n1_nonpublishable_boundary_recommendation/summary.json",
    "P2c120": ROOT / "experiment_records_ik_optimize/20260629083000_p2c120_n1_publishability_boundary_package/summary.json",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_mechanism_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build evidence rows for the n1 failure mechanism map."""
    p99_rows = sources["P2c99"]["rows"]
    p110_final = sources["P2c110"]["final_rows"]
    p111_final = sources["P2c111"]["final_rows"]
    p112_rows = sources["P2c112"]["evidence_rows"]
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "source": "P2c99/P2c84",
            "attempt": "原始物理门禁",
            "best_position_error_m": None,
            "stationarity_norm": next(row for row in p99_rows if row["experiment"] == "P2c84")["stationarity"],
            "condition_number": next(row for row in p99_rows if row["experiment"] == "P2c84")["condition"],
            "force_ratio": next(row for row in p99_rows if row["experiment"] == "P2c84")["load_ratio"],
            "continuity_norm": None,
            "beta_deviation": None,
            "publishable": False,
            "dominant_failure": "载荷传播失配",
            "interpretation": "一阶最优性和条件数接近可用，但力链传播比例约 2.769，超过硬门禁。",
        }
    )
    for row in p110_final:
        rows.append(
            {
                "source": "P2c110",
                "attempt": f"硬力连续 force_scale={row['force_scale']}",
                "best_position_error_m": row["position_error_m"],
                "stationarity_norm": row["stationarity_norm"],
                "condition_number": row["jacobian_condition"],
                "force_ratio": row["max_interface_force_jump_over_rigid_force"],
                "continuity_norm": row["continuity_norm"],
                "beta_deviation": None,
                "publishable": False,
                "dominant_failure": row["failure_class"],
                "interpretation": "强行压低接口力跳变会转化为一阶最优性退化、条件数病态或接口连续残差超限。",
            }
        )
    for row in p111_final:
        rows.append(
            {
                "source": "P2c111",
                "attempt": f"beta 载荷传递 beta_reg={row['beta_reg_scale']}",
                "best_position_error_m": row["position_error_m"],
                "stationarity_norm": row["stationarity_norm"],
                "condition_number": row["jacobian_condition"],
                "force_ratio": row["max_interface_force_jump_over_rigid_force"],
                "continuity_norm": row["continuity_norm"],
                "beta_deviation": row["beta_deviation_from_one"],
                "publishable": False,
                "dominant_failure": row["failure_class"],
                "interpretation": "自由比例参数可改善部分指标，但会破坏物理闭合或造成一阶最优性退化。",
            }
        )
    for row in p112_rows:
        rows.append(
            {
                "source": row["source"],
                "attempt": row["test"],
                "best_position_error_m": row["position_error_m"],
                "stationarity_norm": row["stationarity_norm"],
                "condition_number": row["condition_number"],
                "force_ratio": row["force_ratio"],
                "continuity_norm": None,
                "beta_deviation": row["beta_deviation"],
                "publishable": bool(row["publishable"]),
                "dominant_failure": row["failure_class"],
                "interpretation": "边界建议层面的证据复核。",
            }
        )
    return rows


def build_repair_rows() -> list[dict[str, Any]]:
    """Build repair-route recommendations."""
    return [
        {
            "priority": 1,
            "route": "显式刚性磁体段载荷传播模型",
            "action": "把磁体段外力和外力矩沿刚段几何传播到柔性段接口，而不是只用自由比例或残差权重吸收。",
            "success_gate": "载荷传播比例 <= 1.5，且一阶最优性 <= 1e-5，条件数 < 1e7。",
            "risk": "需要重审刚段端点、磁体中心、柔性节点和终端点约定。",
        },
        {
            "priority": 2,
            "route": "任务降维与保守工作点",
            "action": "只保留一维或二维有效子空间末端位置任务，并选择载荷传播自然较小的目标。",
            "success_gate": "有效秩与任务维度一致，扰动重求解线性误差 < 0.05。",
            "risk": "只能发布有效子空间导数，不能发布完整三维位置或位姿。",
        },
        {
            "priority": 3,
            "route": "模型对照基线",
            "action": "保留当前 n1_tip_z_rand15 作为负样本，对所有修复模型复测同一指标。",
            "success_gate": "同一工作点至少一个模型同时通过物理闭合和局部线性化门禁。",
            "risk": "若连续三轮仍同类失败，应固定为工作空间不可发布边界。",
        },
        {
            "priority": 4,
            "route": "效率优化",
            "action": "只有在可信门禁通过后再做缓存、批量门禁或隐式灵敏度。",
            "success_gate": "可信性不下降且完整审计耗时下降。",
            "risk": "当前不应以速度优化替代物理模型修复。",
        },
    ]


def plot_mechanisms(rows: list[dict[str, Any]]) -> None:
    """Plot core n1 failure metrics."""
    configure_matplotlib_cjk()
    compact = [
        row
        for row in rows
        if row["source"] in {"P2c99/P2c84", "P2c110", "P2c111"}
    ]
    labels = [row["attempt"] for row in compact]
    stationarity = [row["stationarity_norm"] or 0.0 for row in compact]
    condition = [row["condition_number"] or 0.0 for row in compact]
    force = [row["force_ratio"] or 0.0 for row in compact]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].bar(labels, stationarity, color="#356f8f")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("一阶最优性残差")
    axes[0].set_title("n1 当前模型失败机制矩阵")

    axes[1].bar(labels, condition, color="#8f4f35")
    axes[1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("条件数")

    axes[2].bar(labels, force, color="#4f8f35")
    axes[2].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("载荷传播或力跳变指标")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n1_failure_mechanism_matrix_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], mechanism_rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]]) -> None:
    """Write the Chinese experiment record."""
    evidence_lines = "\n".join(
        f"- {row['attempt']}：不可发布；主失败为{row['dominant_failure']}；解释：{row['interpretation']}"
        for row in mechanism_rows[:8]
    )
    repair_lines = "\n".join(
        f"- 优先级 {row['priority']}：{row['route']}。措施：{row['action']} 验收：{row['success_gate']}"
        for row in repair_rows
    )
    record = f"""# P2c126 单点磁控失败机制与修复路线图

## 实验目的

本轮不新增扰动扫描，而是综合 P2c99、P2c110、P2c111、P2c112 和 P2c120 的证据，形成单点磁控导管当前工作点的失败机制图谱和后续修复路线。目标是避免在基线门禁失败时继续计算无物理可信基础的灵敏度。

## 证据链

{evidence_lines}

## 机制解释

当前 n1 失败不是单一权重参数造成的。原始物理门禁中，一阶最优性和条件数曾接近可接受，但载荷传播比例约 2.769，超过硬门禁。硬力连续模型可以压低接口力跳变，却会带来一阶最优性退化、条件数病态或接口连续残差超限。beta 载荷传递模型可以改善拟合，但会把刚性磁体段载荷传播改写为自由比例拟合，因此不能作为可发布灵敏度依据。

这说明主矛盾位于刚性磁体段外力向柔性段内力链传播的物理耦合模型，而不是末端位置误差本身。末端拟合成功不能证明灵敏度可信。

## 工程边界

- 当前工作点：`n1_tip_z_rand15`。
- 当前判定：不可发布。
- 允许用途：模型失配分析、工作空间边界说明、后续模型修复对照。
- 禁止用途：发布末端位置灵敏度、接入外层 IK、扰动扫描、完整末端位姿求解。
- 重新测试前置条件：显式修复刚性磁体段载荷传播模型，或降低任务维度并更换保守工作点。

## 修复路线

{repair_lines}

## 结论

单点磁控当前模型下尚未获得可发布有效子空间灵敏度。最优策略不是继续扰动扫描，而是将其固定为负样本边界，并以此检验后续刚性磁体段载荷传播模型修复是否真正改善物理闭合。

## 下一步计划

若继续 n1，应优先实现“显式刚性磁体段载荷传播模型”的最小原型，并只跑基线门禁；若基线仍失败，不进入扰动灵敏度扫描。若近期目标是工程可用 IK，应继续沿 n2 分支条件位置灵敏度推进。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Generate the n1 failure mechanism package."""
    t0 = time.perf_counter()
    sources = {key: read_json(path) for key, path in SOURCE_PATHS.items()}
    mechanism_rows = build_mechanism_rows(sources)
    repair_rows = build_repair_rows()
    publishable_count = sum(1 for row in mechanism_rows if row["publishable"])
    failure_counts: dict[str, int] = {}
    for row in mechanism_rows:
        failure_counts[row["dominant_failure"]] = failure_counts.get(row["dominant_failure"], 0) + 1
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "n1_failure_mechanism_mapped_not_publishable",
        "case": "n1_tip_z_rand15",
        "publishable_count": publishable_count,
        "evidence_count": len(mechanism_rows),
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "primary_failure": "刚性磁体段载荷传播失配与模型耦合缺失",
        "failure_counts": failure_counts,
        "recommended_next": "显式刚性磁体段载荷传播模型最小原型；只跑基线门禁，不直接扰动扫描",
        "source_experiments": {key: str(path.relative_to(ROOT)) for key, path in SOURCE_PATHS.items()},
        "wall_s": time.perf_counter() - t0,
    }
    write_csv(OUT_DIR / "mechanism_rows.csv", mechanism_rows)
    write_csv(OUT_DIR / "repair_route_rows.csv", repair_rows)
    write_csv(OUT_DIR / "summary.csv", [{"status": summary["status"], "publishable_count": publishable_count, "evidence_count": len(mechanism_rows), "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "source": row["source"],
                    "attempt": row["attempt"],
                    "failure_class": row["dominant_failure"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in mechanism_rows
            if not row["publishable"]
        ),
        encoding="utf-8",
    )
    plot_mechanisms(mechanism_rows)
    write_record(summary, mechanism_rows, repair_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
