"""Package the n1 non-publishable sensitivity boundary evidence."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


SOURCES = {
    "P2c99": ROOT / "experiment_records_ik_optimize/20260628215500_p2c99_n1_nonpublishable_boundary_synthesis/summary.json",
    "P2c110": ROOT / "experiment_records_ik_optimize/20260629030000_p2c110_n1_hard_force_scaled_continuity/summary.json",
    "P2c111": ROOT / "experiment_records_ik_optimize/20260629033000_p2c111_n1_beta_load_transfer/summary.json",
    "P2c112": ROOT / "experiment_records_ik_optimize/20260629040000_p2c112_n1_nonpublishable_boundary_recommendation/summary.json",
}


def read_json(path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def collect_evidence(summaries: dict[str, dict]) -> list[dict]:
    """Collect the most relevant n1 publishability evidence."""
    rows: list[dict] = []
    for row in summaries["P2c99"]["rows"]:
        if row["experiment"] in {"P2c84", "P2c87"}:
            rows.append(
                {
                    "source": row["experiment"],
                    "test": row["evidence"],
                    "position_error_m": None,
                    "stationarity_norm": row.get("stationarity"),
                    "condition_number": row.get("condition"),
                    "force_ratio": row.get("load_ratio"),
                    "beta_deviation": None,
                    "publishable": False,
                    "failure_class": row["failure_class"],
                    "interpretation": row["interpretation"],
                }
            )
    for row in summaries["P2c110"]["final_rows"]:
        rows.append(
            {
                "source": "P2c110",
                "test": f"硬力连续 force_scale={row['force_scale']}",
                "position_error_m": row["position_error_m"],
                "stationarity_norm": row["stationarity_norm"],
                "condition_number": row["jacobian_condition"],
                "force_ratio": row["max_interface_force_jump_over_rigid_force"],
                "beta_deviation": None,
                "publishable": bool(row["publish_gate"]),
                "failure_class": row["failure_class"],
                "interpretation": "强制力连续后，任务拟合、一阶最优性、条件数或接口残差不能同时过门。",
            }
        )
    for row in summaries["P2c111"]["final_rows"]:
        rows.append(
            {
                "source": "P2c111",
                "test": f"beta 载荷传递 beta_reg={row['beta_reg_scale']}",
                "position_error_m": row["position_error_m"],
                "stationarity_norm": row["stationarity_norm"],
                "condition_number": row["jacobian_condition"],
                "force_ratio": row["max_interface_force_jump_over_rigid_force"],
                "beta_deviation": row["beta_deviation_from_one"],
                "publishable": bool(row["publish_gate"]),
                "failure_class": row["failure_class"],
                "interpretation": "允许载荷传递比例后，物理闭合和一阶最优性仍不能同时满足。",
            }
        )
    return rows


def decision_interface() -> dict:
    """Return a read-only boundary decision for the n1 case."""
    return {
        "case": "n1_tip_z_rand15",
        "decision": "not_publishable_current_model",
        "position_sensitivity_allowed": False,
        "disturbance_scan_allowed": False,
        "outer_ik_allowed": False,
        "allowed_use": "模型失配分析、工作空间边界说明、后续模型修复对照",
        "forbidden_use": "发布灵敏度、接入外层 IK、扰动扫描、完整位姿或位置任务默认求解",
        "primary_failure": "刚性磁体段载荷传播失配与模型耦合缺失",
        "secondary_failures": ["一阶最优性退化", "条件数病态", "接口连续残差超限", "载荷传递比例偏离物理闭合"],
        "repair_required_before_retest": "显式修复刚性磁体段外力向柔性段内力链的传播模型，或降低任务维度并选择更保守工作点。",
    }


def plot_evidence(rows: list[dict]) -> None:
    """Plot n1 publishability gate evidence."""
    configure_matplotlib_cjk()
    labels = [f"{row['source']}\n{row['test'][:12]}" for row in rows]
    stationarity = [row["stationarity_norm"] or 0.0 for row in rows]
    condition = [row["condition_number"] or 0.0 for row in rows]
    force = [row["force_ratio"] or 0.0 for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(labels, stationarity, color="#b33a3a")
    axes[0].axhline(1.0e-5, color="black", linestyle="--", label="一阶最优性门限")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("一阶最优性")
    axes[0].legend()
    axes[1].bar(labels, condition, color="#c58a19")
    axes[1].axhline(1.0e7, color="black", linestyle="--", label="条件数建议门限")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("条件数")
    axes[1].legend()
    axes[2].bar(labels, force, color="#457b9d")
    axes[2].axhline(1.5, color="black", linestyle="--", label="载荷传播门限")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("载荷传播比例")
    axes[2].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("n1 当前模型不可发布证据包")
    fig.savefig(OUT_DIR / "n1_publishability_boundary_package_cn.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """Build the n1 publishability boundary package."""
    t0 = time.perf_counter()
    summaries = {key: read_json(path) for key, path in SOURCES.items()}
    evidence_rows = collect_evidence(summaries)
    interface = decision_interface()
    publishable_count = sum(bool(row["publishable"]) for row in evidence_rows)
    failure_classes = sorted({row["failure_class"] for row in evidence_rows})
    summary = {
        "experiment_id": OUT_DIR.name,
        "status": "n1_not_publishable_current_model",
        "publishable_count": publishable_count,
        "evidence_count": len(evidence_rows),
        "failure_classes": failure_classes,
        "decision_interface": interface,
        "source_experiments": {key: str(path.relative_to(ROOT)) for key, path in SOURCES.items()},
        "wall_s": time.perf_counter() - t0,
        "interpretation": "该证据包将 n1 当前工作点固定为不可发布边界案例；除非修复载荷传播模型或更换任务口径，否则不得进入扰动灵敏度扫描。",
    }
    write_csv(OUT_DIR / "evidence_rows.csv", evidence_rows)
    write_csv(OUT_DIR / "summary.csv", [{"publishable_count": publishable_count, "evidence_count": len(evidence_rows), "wall_s": summary["wall_s"]}])
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "n1_boundary_decision_interface.json", interface)
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps({"source": row["source"], "failure_class": row["failure_class"], "evidence": row}, ensure_ascii=False) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    plot_evidence(evidence_rows)
    row_lines = "\n".join(
        f"- {row['source']} / {row['test']}：{row['failure_class']}。"
        for row in evidence_rows
    )
    record = f"""# P2c120 n1 当前模型不可发布边界证据包

## 实验目的

本轮不重新求解，而是将 P2c99、P2c110、P2c111 和 P2c112 的单点磁控导管证据整理为可审计边界包。目标是判断当前 `n1_tip_z_rand15` 工作点是否能发布有效子空间末端位置灵敏度，或是否应被固定为当前模型下的不可发布案例。

## 证据摘要

{row_lines}

## 判定

- 可发布证据数：{publishable_count} / {len(evidence_rows)}。
- 当前判定：`{summary["status"]}`。
- 允许用途：{interface["allowed_use"]}。
- 禁止用途：{interface["forbidden_use"]}。
- 主失败机制：{interface["primary_failure"]}。

## 解释

历史样本显示，n1 在某些口径下可以得到较小末端位置误差或较小载荷跳变，但这些性质不能同时满足硬门禁。硬力连续会引入一阶最优性退化、条件数病态或接口残差超限；beta 载荷传递松弛会把物理闭合变成自由比例拟合，不能作为可发布灵敏度依据。因此当前失败不是单纯调权重可解决的问题，而是刚性磁体段外力向柔性内力链传播的模型耦合缺口。

## 工程建议

当前 n1 不应进行扰动灵敏度扫描，不应接入外层 IK，也不应发布位置或完整位姿灵敏度。后续若继续 n1，应先修复刚性磁体段载荷传播模型，或选择更保守目标并降低任务维度，再重新从基线门禁开始。

## 下一步计划

主线可回到 n2 诊断模式外层 IK 的工作空间扫描；n1 只作为模型失配边界案例保留，直到有新的物理模型修复方案。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
