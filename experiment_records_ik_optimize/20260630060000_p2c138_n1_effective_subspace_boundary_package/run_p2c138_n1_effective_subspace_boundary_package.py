"""Build the n1 effective-subspace boundary package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


configure_matplotlib_cjk()

OUT_DIR = Path(__file__).resolve().parent
P2C126_DIR = ROOT / "experiment_records_ik_optimize" / "20260629140000_p2c126_n1_failure_mechanism_repair_map"
P2C131_DIR = ROOT / "experiment_records_ik_optimize" / "20260629210000_p2c131_n1_configuration_boundary"
P2C132_DIR = ROOT / "experiment_records_ik_optimize" / "20260629230000_p2c132_n1_mid_branch_consistency"


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON data."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def jsonable(value: Any) -> Any:
    """Convert nested values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_boundary_rows(summary131: dict[str, Any], summary132: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact n1 boundary rows."""
    rows: list[dict[str, Any]] = []
    for item in summary131.get("rows", summary131.get("final_rows", [])):
        rows.append(
            {
                "source": "P2c131",
                "case": item.get("case"),
                "task": item.get("task", item.get("task_mode")),
                "baseline_gate": item.get("baseline_gate"),
                "failure_class": item.get("failure_class"),
                "position_error_m": item.get("position_error_m"),
                "stationarity_norm": item.get("stationarity_norm"),
                "effective_condition": item.get("effective_condition", item.get("jacobian_condition")),
                "force_ratio": item.get("max_interface_force_jump_over_rigid_force"),
                "release_class": "候选需复核" if item.get("baseline_gate") else "不可发布",
            }
        )
    for item in summary132.get("final_rows", []):
        rows.append(
            {
                "source": "P2c132",
                "case": "n1_mid_z_rand15_a010",
                "task": "position_3d",
                "init_mode": item.get("init_mode"),
                "baseline_gate": item.get("baseline_gate"),
                "failure_class": item.get("failure_class"),
                "position_error_m": item.get("position_error_m"),
                "stationarity_norm": item.get("stationarity_norm"),
                "effective_condition": item.get("effective_condition", item.get("jacobian_condition")),
                "force_ratio": item.get("max_interface_force_jump_over_rigid_force"),
                "release_class": "孤立基线通过" if item.get("baseline_gate") else "不可发布",
            }
        )
    return rows


def plot_boundary(rows: list[dict[str, Any]]) -> None:
    """Plot n1 gate evidence."""
    p2c131 = [row for row in rows if row["source"] == "P2c131"]
    p2c132 = [row for row in rows if row["source"] == "P2c132"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(
        [f"{row.get('case')}\n{row.get('task')}" for row in p2c131],
        [row["effective_condition"] for row in p2c131],
        color=["#2a9d8f" if row["baseline_gate"] else "#b24b38" for row in p2c131],
    )
    axes[0].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_title("配置筛选条件数")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        [row.get("init_mode", "") for row in p2c132],
        [row["effective_condition"] for row in p2c132],
        color=["#2a9d8f" if row["baseline_gate"] else "#b24b38" for row in p2c132],
    )
    axes[1].axhline(1.0e7, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("中段候选多初值复核条件数")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "n1_effective_subspace_boundary_cn.png", dpi=220)
    plt.close(fig)


def write_record(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the Chinese boundary report."""
    table = "\n".join(
        f"| {row.get('source')} | {row.get('case')} | {row.get('task')} | {row.get('init_mode', '')} | {row.get('baseline_gate')} | {row.get('failure_class')} | {row.get('release_class')} |"
        for row in rows
    )
    record = f"""# P2c138 单点磁控有效子空间边界包

## 实验目的

本轮综合 P2c126、P2c131 和 P2c132 的证据，给出单点磁控 n1 的阶段性发布边界。目标不是寻找新的拟合解，而是判断当前证据是否足以支持有效子空间末端位置灵敏度发布。

## 证据来源

- P2c126：原始单点末端磁体工作点 `n1_tip_z_rand15` 存在刚性磁体段载荷传播失配；硬连续和自由比例修复均不能形成可发布物理闭合。
- P2c131：更换磁体位置后，中段单磁体 `n1_mid_z_rand15_a010` 的三维末端位置任务存在一个基线通过样本。
- P2c132：该中段候选只在 FK oracle 初值下通过；半 oracle、零状态和零电流 oracle 初值均因条件数病态失败，且中心线漂移约 `7-9 mm`，分支一致性未通过。

## 证据表

| 来源 | 工况 | 任务 | 初值 | 基线门禁 | 失败分类 | 发布类别 |
| --- | --- | --- | --- | --- | --- | --- |
{table}

## 判定

当前 n1 尚无可发布有效子空间末端位置灵敏度。原因不是所有配置都不能拟合末端位置，而是唯一基线通过的中段候选缺少分支一致性证据；该解对初值敏感，无法证明代表稳定物理解分支。

## 失败机制

1. 末端单磁体工作点：主失败为刚性磁体段载荷传播失配和模型耦合缺失。
2. 基端与末端替代配置：主失败为条件数病态或任务误差超限。
3. 中段替代配置：存在孤立基线通过，但多初值复核失败，主失败为初始化敏感和多平衡竞争风险。

## 工程建议

- 当前 n1 不应接入外层逆运动学，也不应发布末端位置灵敏度。
- `n1_mid_z_rand15_a010` 可保留为模型修复后的正向复核候选，但不能直接扰动扫描。
- 若继续 n1，应优先改善分支恢复策略或改变工作点，使至少两条合理初值路径收敛到同一中心线分支。
- 若目标是近期工程可用，应优先使用 n2 的优势方向局部导数，而不是强推 n1。

## 结论

P2c138 给出 n1 阶段性不可发布边界：当前模型和当前测试配置下，没有满足“基线门禁、分支一致性、局部线性化、载荷传播、扰动重求解一致性”的单点磁控有效子空间灵敏度。

## 下一步计划

建议 P2c139 执行 n1 分支恢复修复小实验：只针对 `n1_mid_z_rand15_a010`，尝试 continuation 初值路径和更小任务步长。如果仍只有 oracle 初值通过，则固定为不可发布结论，转向更换工作点或模型修复。

## 汇总

- 发布状态：{summary['release_status']}。
- 外层 IK 允许：{summary['outer_ik_allowed']}。
- 可发布方向数：{summary['release_direction_count']}。
"""
    (OUT_DIR / "record.md").write_text(record, encoding="utf-8")


def main() -> None:
    """Build n1 boundary artifacts."""
    summary126 = load_json(P2C126_DIR / "summary.json")
    summary131 = load_json(P2C131_DIR / "summary.json")
    summary132 = load_json(P2C132_DIR / "summary.json")
    rows = build_boundary_rows(summary131, summary132)
    payload = {
        "experiment_id": OUT_DIR.name,
        "release_status": "not_publishable",
        "release_direction_count": 0,
        "full_3d_position_publishable": False,
        "full_pose_publishable": False,
        "outer_ik_allowed": False,
        "disturbance_scan_allowed": False,
        "dominant_failure_classes": [
            "刚性磁体段载荷传播失配",
            "条件数病态",
            "初始化敏感",
            "多平衡竞争风险",
        ],
        "boundary_rows": rows,
        "evidence": {
            "p2c126_summary": str(P2C126_DIR / "summary.json"),
            "p2c131_summary": str(P2C131_DIR / "summary.json"),
            "p2c132_summary": str(P2C132_DIR / "summary.json"),
            "p2c126_release_status": summary126.get("release_status"),
            "p2c132_branch_status": summary132.get("branch_status"),
        },
    }
    write_csv(OUT_DIR / "boundary_rows.csv", rows)
    write_csv(
        OUT_DIR / "summary.csv",
        [
            {
                "release_status": payload["release_status"],
                "release_direction_count": payload["release_direction_count"],
                "outer_ik_allowed": payload["outer_ik_allowed"],
                "disturbance_scan_allowed": payload["disturbance_scan_allowed"],
            }
        ],
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "failure_events.jsonl").write_text(
        json.dumps(
            {
                "failure_class": "单点磁控有效子空间灵敏度不可发布",
                "reason": "唯一基线通过样本缺少分支一致性证据，原始末端磁体工作点存在刚性磁体段载荷传播失配。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_boundary(rows)
    write_record(payload, rows)
    print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
