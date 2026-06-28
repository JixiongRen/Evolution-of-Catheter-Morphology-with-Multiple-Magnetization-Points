"""Synthesize the current n1 non-publishable boundary."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_records_ik_optimize.plotting_fonts import configure_matplotlib_cjk


OUT_DIR = Path(__file__).resolve().parent
configure_matplotlib_cjk()

SOURCES = {
    "P2c82": ROOT / "experiment_records_ik_optimize/20260628120813_p2c82_n1_effective_subspace_boundary/summary.json",
    "P2c84": ROOT / "experiment_records_ik_optimize/20260628121600_p2c84_block_stationarity_load_closure_audit/summary.json",
    "P2c85": ROOT / "experiment_records_ik_optimize/20260628124937_p2c85_n1_boundary_component_conflict_synthesis/summary.json",
    "P2c86": ROOT / "experiment_records_ik_optimize/20260628133120_p2c86_n1_boundary_relaxation_load_balance/summary.json",
    "P2c87": ROOT / "experiment_records_ik_optimize/20260628140500_p2c87_n1_rigid_endpoint_load_closure/summary.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the evidence table."""
    p84_n1 = data["P2c84"]["gate_rows"][0]
    p85 = data["P2c85"]["gate_status"]
    p86_rows = data["P2c86"]["rows"]
    p87 = data["P2c87"]
    force_ratio = max(row["force_jump_over_rigid_force"] for row in p87["force_rows"])
    min_torque_ratio = min(row["torque_jump_over_rigid_torque"] for row in p87["torque_rows"])
    return [
        {
            "experiment": "P2c82",
            "evidence": "历史边界综合",
            "stationarity": 1.065e-6,
            "condition": 8.113e6,
            "load_ratio": 2.769,
            "status": "不可发布",
            "failure_class": "载荷传播失配",
            "interpretation": "单点近门样本已有二维有效秩线索，但原物理门禁不通过。",
        },
        {
            "experiment": "P2c84",
            "evidence": "分块最优性与载荷闭合",
            "stationarity": p84_n1["stationarity_norm"],
            "condition": p84_n1["jacobian_condition"],
            "load_ratio": p84_n1["max_interface_force_jump_over_rigid_force"],
            "status": "不可发布",
            "failure_class": p84_n1["failure_class"],
            "interpretation": "一阶最优性和条件数已过门，但载荷传播比例仍明显超限。",
        },
        {
            "experiment": "P2c85",
            "evidence": "边界分量冲突综合",
            "stationarity": p85["stationarity"],
            "condition": p85["condition"],
            "load_ratio": p85["load_ratio"],
            "status": "不可发布",
            "failure_class": "边界-任务-接口相容性冲突",
            "interpretation": "失败不是简单边界权重误设，也不是删除边界分量即可解决。",
        },
        {
            "experiment": "P2c86",
            "evidence": "边界弱化与载荷重标定反事实",
            "stationarity": p86_rows[0]["stationarity_norm"],
            "condition": p86_rows[0]["jacobian_condition"],
            "load_ratio": p86_rows[0]["original_load_ratio"],
            "status": "不可发布",
            "failure_class": "反事实不能替代物理闭合",
            "interpretation": "边界弱化、载荷重标定和联合白化只改变诊断口径，不能修复原物理载荷链。",
        },
        {
            "experiment": "P2c87",
            "evidence": "刚性磁体段端点载荷闭合",
            "stationarity": None,
            "condition": None,
            "load_ratio": force_ratio,
            "torque_ratio": min_torque_ratio,
            "status": "不可发布",
            "failure_class": "刚段外力与柔性段内力链不相容",
            "interpretation": "中点等效载荷可修正力矩跳变，但不能修正力跳变；主缺口是力连续性。",
        },
    ]


def _plot(rows: list[dict[str, Any]]) -> None:
    """Plot the n1 non-publishable evidence chain."""
    labels = [row["experiment"] for row in rows]
    load = [row["load_ratio"] for row in rows]
    stationarity = [row["stationarity"] or float("nan") for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(labels, load, color="#e76f51")
    axes[0].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("载荷传播比例")
    axes[1].bar(labels, stationarity, color="#457b9d")
    axes[1].axhline(1e-5, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("一阶最优性")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("P2c99 n1 当前模型不可发布边界")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "n1_nonpublishable_boundary_cn.png", dpi=180)
    plt.close(fig)


def _record(rows: list[dict[str, Any]]) -> str:
    """Render the Chinese synthesis report."""
    table = [
        "| 实验 | 证据 | 状态 | 载荷比 | 一阶最优性 | 条件数 | 失败分类 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table.append(
            f"| {row['experiment']} | {row['evidence']} | {row['status']} | {row['load_ratio']:.3e} | "
            f"{row['stationarity'] if row['stationarity'] is not None else float('nan'):.3e} | "
            f"{row['condition'] if row['condition'] is not None else float('nan'):.3e} | {row['failure_class']} |"
        )
    return "\n".join(
        [
            "# P2c99 n1 当前模型不可发布边界综合",
            "",
            "## 实验目的",
            "",
            "本轮不重新求解，而是综合 P2c82、P2c84 至 P2c87 的单点磁控证据，明确 `n1_tip_z_rand15, alpha=0.10` 在当前模型下是否具备有效子空间末端位置灵敏度发布条件，并给出下一步模型修复边界。",
            "",
            "## 证据表",
            "",
            *table,
            "",
            "## 阶段性判定",
            "",
            "n1 当前模型下不能发布有效子空间末端位置灵敏度。关键原因不是末端位置拟合不足，也不是一阶最优性整体失败；P2c84 已显示近门样本的一阶最优性和条件数达到继续分析要求。真正阻断项是刚性磁体段外力与相邻柔性段内力链不相容，表现为载荷传播比例稳定高于 1.5，最高约 2.769。",
            "",
            "P2c85 和 P2c86 排除了简单边界权重、边界分量删除、接口载荷重标定和联合白化等低风险修复。P2c87 进一步说明中点等效载荷可以使力矩传播近似闭合，但无法降低力跳变。因此，n1 的主缺口应归因为刚段-柔性段力连续性模型缺失，而不是力矩作用点选择错误。",
            "",
            "## 可用范围",
            "",
            "- 可用于诊断：二维有效子空间线索、近门软约束样本、边界与接口耦合分析。",
            "- 可用于模型修复：刚性磁体段接口力连续方程、外力传播残差尺度、冗余内力自由度分析。",
            "- 不可用于发布：单点有效子空间灵敏度、外层 IK 默认灵敏度、完整末端位姿灵敏度。",
            "",
            "## 失败机制",
            "",
            "当前失败机制应归类为“刚性磁体段载荷传播失配”与“模型耦合缺失”。软约束近门解可以拟合末端位置，但该拟合依赖分段内力自由度，未能由唯一物理载荷链闭合。因此，即便末端误差达到微米级，也不能说明该解的局部导数具有物理可信性。",
            "",
            "## 下一步计划",
            "",
            "1. 若继续 n1，应先重写接口力连续残差或显式引入有物理含义的刚段力传递参数。",
            "2. 任何新模型都必须回到原门禁评价：一阶最优性、条件数、接口连续、载荷传播和扰动重求解。",
            "3. 在模型修复前，不再做 n1 扰动灵敏度扫描，也不做完整姿态任务。",
            "",
        ]
    )


def main() -> None:
    data = {name: _read_json(path) for name, path in SOURCES.items()}
    rows = _rows(data)
    payload = {
        "experiment": OUT_DIR.name,
        "status": "n1_current_model_not_publishable",
        "sources": {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
        "rows": rows,
        "decision": {
            "n1_effective_subspace_position": "not_publishable_current_model",
            "primary_failure": "rigid_magnet_force_propagation_mismatch",
            "recommended_next": "interface_force_continuity_model_repair",
        },
    }
    _write_csv(OUT_DIR / "summary.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [{"failure": row["failure_class"], "evidence": row} for row in rows if row["status"] != "通过"]
    (OUT_DIR / "failure_events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
        encoding="utf-8",
    )
    _plot(rows)
    (OUT_DIR / "record.md").write_text(_record(rows), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
